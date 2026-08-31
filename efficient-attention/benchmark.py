"""
Step 19 — Putting numbers on all of it
=======================================
Three methods, one table. The analytic counts are what you would use to decide
whether a method can possibly help before writing a kernel; the measurements
are what tell you whether the constants ruin it.

Effort: small, and the most useful half-hour in the directory. Do not skip it —
a technique you cannot cost is a technique you cannot choose.

What you build:
  count_flops            -> leading-order arithmetic per forward pass
  peak_extra_elements    -> the largest intermediate that must exist at once
  count_hbm_bytes        -> measured on the toy device, not derived
  fit_slope              -> the empirical exponent from a handful of points
  measure_time           -> wall clock
  scaling_report         -> all of the above across sequence lengths
"""

from typing import Dict, Optional, Sequence

import numpy as np

from baseline import attention
from clustered import clustered_attention, improved_clustered_attention
from common import human_bytes, random_qkv, rel_error, timeit
from flash import flash_attention
from linear import causal_linear_attention, linear_attention

KINDS = ("softmax", "clustered", "improved", "linear", "causal_linear", "flash")


# ---------------------------------------------------------------------------
# Analytic counts
# ---------------------------------------------------------------------------

def count_flops(kind: str, n: int, d: int, dv: Optional[int] = None,
                clusters: int = 32, top_k: int = 32, bits: int = 16,
                iters: int = 10) -> float:
    """Multiply-adds (2 flops each) for one forward pass, leading terms only.

    TODO — return, for each kind:

      softmax        2 n^2 d + 5 n^2 + 2 n^2 dv                    quadratic
      flash          the same; see below
      clustered      2 C n d + 5 C n + 2 C n dv                    linear
                     + 2 n d B          (the LSH projection)
                     + 2 n C B T        (T K-means iterations)
      improved       clustered + 2 n k d + 2 n k dv + n C
      linear         4 n d dv + 2 n d                              linear
      causal_linear  the same leading term

    Background — read the `flash` row twice:
      FlashAttention does not reduce flops AT ALL. It does slightly more
      arithmetic than standard attention once the backward pass recomputes P.
      It is faster because attention at these shapes is memory bound, not
      compute bound, so the currency that matters is `count_hbm_bytes`, not
      this. Two of the three papers in this directory attack flops; the third
      attacks bytes; conflating the two is the most common way to reason
      wrongly about which one to reach for.
    """
    raise NotImplementedError


def peak_extra_elements(kind: str, n: int, d: int, dv: Optional[int] = None,
                        clusters: int = 32, top_k: int = 32,
                        sram_bytes: int = 64 * 1024, itemsize: int = 8) -> int:
    """Largest intermediate that must exist at once, in elements.

    Not counting Q, K, V and O themselves, which every method needs.

    TODO:
      softmax        n * n            the score matrix — and in training it
                                      must survive until the backward pass
      clustered      C * n
      improved       C * n + n * k
      linear         d * dv           the state; INDEPENDENT OF n
      causal_linear  d * dv           when written as a scan (see linear.py)
      flash          B_r * B_c        one tile — and it lives in SRAM, not HBM
    """
    raise NotImplementedError


def count_hbm_bytes(kind: str, n: int, d: int, sram_bytes: int = 64 * 1024,
                    seed: int = 0) -> int:
    """Measured, not derived: run it on the toy Device and read the counter.

    TODO: build a Device, put random Q, K, V in its HBM, run
    standard_attention_io / flash_forward / flash2_forward according to `kind`,
    and return dev.total_bytes.
    """
    raise NotImplementedError


def fit_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Slope of log(y) against log(x): the empirical exponent.

    2.0 means quadratic, 1.0 linear, 0.0 constant. Fitting in log space is the
    only honest way to read an asymptotic off a handful of measurements.

    TODO: np.polyfit(log(xs), log(ys), 1)[0].
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_time(kind: str, n: int, d: int, clusters: int = 32, top_k: int = 32,
                 seed: int = 0, repeats: int = 3) -> float:
    """Wall-clock seconds for one forward pass at this size.

    TODO: dispatch on `kind` to the implementation from the earlier files and
    time it with common.timeit.

    Expect a surprise: the clustered implementations will be SLOWER than plain
    attention here, because their K-means loop runs in Python while numpy's
    matmul runs in BLAS. That is a fact about this implementation, not about
    the method — which is exactly why the flop column exists next to it.
    """
    raise NotImplementedError


def scaling_report(ns: Sequence[int] = (128, 256, 512, 1024), d: int = 32,
                   kinds: Sequence[str] = ("softmax", "clustered", "improved",
                                           "linear"),
                   clusters: int = 32, top_k: int = 32) -> Dict[str, Dict]:
    """Time, flops, memory and accuracy for each method across sequence lengths.

    TODO: for each kind and each n, collect time / flops / peak / relative
    error against exact attention, and add the three fitted exponents
    (time_slope, flop_slope, peak_slope). Return a dict keyed by kind.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------

def _demo():
    ns = (128, 256, 512, 1024)
    d = 32
    report = scaling_report(ns, d=d)

    print(f"d={d}, clusters=32, top_k=32, iid gaussian Q/K/V "
          f"(the WORST case for clustering)\n")
    for kind, r in report.items():
        print(f"{kind}")
        print(f"  {'N':>6} {'time':>10} {'flops':>12} {'peak extra':>12} "
              f"{'rel error':>10}")
        for i, n in enumerate(r["ns"]):
            print(f"  {n:>6} {r['time'][i] * 1e3:>9.2f}ms {r['flops'][i] / 1e6:>10.1f}M"
                  f" {human_bytes(r['peak'][i] * 8):>12} {r['error'][i]:>10.4f}")
        print(f"  measured exponent: time {r['time_slope']:.2f}, "
              f"flops {r['flop_slope']:.2f}, memory {r['peak_slope']:.2f}\n")

    print("read those error columns carefully:")
    print("  * clustered/improved: a genuine approximation error, and this is")
    print("    their worst case — iid gaussian queries do not cluster at all.")
    print("  * linear: not an error. Linear attention computes a DIFFERENT")
    print("    function; the number is how far that function is from softmax")
    print("    attention on untrained random inputs, which is not a defect.")
    print("  * clustered wall-clock is worse than softmax here only because the")
    print("    K-means loop is Python; the flop column is the portable claim.\n")

    from common import clustered_qkv
    print("the same two approximations on queries that actually cluster")
    print(f"  {'N':>6} {'clustered':>11} {'improved':>11}")
    for n in ns:
        Q, K, V = clustered_qkv(n, d, groups=16, seed=0, spread=0.1, scale=2.0)
        exact = attention(Q, K, V)
        print(f"  {n:>6}"
              f" {rel_error(clustered_attention(Q, K, V, n_clusters=32), exact):>11.4f}"
              f" {rel_error(improved_clustered_attention(Q, K, V, n_clusters=32, top_k=32), exact):>11.4f}")
    print()

    print("HBM traffic, measured on the toy device (SRAM 64 KB, d=32)\n")
    print(f"  {'N':>6} {'standard':>12} {'flash-1':>12} {'flash-2':>12} {'ratio':>8}")
    for n in (128, 256, 512):
        std = count_hbm_bytes("softmax", n, d)
        f1 = count_hbm_bytes("flash", n, d)
        f2 = count_hbm_bytes("flash2", n, d)
        print(f"  {n:>6} {human_bytes(std):>12} {human_bytes(f1):>12}"
              f" {human_bytes(f2):>12} {std / f1:>7.2f}x")

    print("\nwhat each method costs at N = 65536, d = 64 (fp16), per head per layer")
    for kind in ("softmax", "clustered", "improved", "linear"):
        peak = peak_extra_elements(kind, 65536, 64, clusters=32, top_k=32)
        flops = count_flops(kind, 65536, 64, clusters=32, top_k=32)
        print(f"  {kind:<14} {human_bytes(peak * 2):>10} intermediate, "
              f"{flops / 1e9:>9.1f} GFLOP")


if __name__ == "__main__":
    _demo()
