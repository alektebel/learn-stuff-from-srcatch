"""
Step 19 — Putting numbers on all of it. Complete solution.

Three methods, one table. The analytic counts are what you would use to decide
whether a method can possibly help before writing a kernel; the measurements are
what tells you whether the constants ruin it.
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
    """Multiply-adds (counted as 2 flops each) for one forward pass.

    Only the leading terms; the point is the exponent of n, not the constant.

      softmax        2 n^2 d + 5 n^2 + 2 n^2 dv           quadratic
      clustered      2 C n d + 5 C n + 2 C n dv           linear, plus clustering
                     + 2 n d B (hashing) + 2 n C B T      (K-means iterations)
      improved       clustered + 2 n k d + 2 n k dv       linear
      linear         4 n d dv + 2 n d                     linear
      causal_linear  same leading term, one state update per position
      flash          softmax's arithmetic, unchanged      quadratic

    Read the last row twice. FlashAttention does not reduce flops at all — it
    does slightly *more* arithmetic than standard attention once the backward
    pass recomputes P. It is faster because attention at these shapes is memory
    bound, not compute bound, so the currency that matters is the byte count in
    `count_hbm_bytes`, not this.
    """
    dv = d if dv is None else dv
    if kind == "softmax":
        return 2 * n * n * d + 5 * n * n + 2 * n * n * dv
    if kind == "flash":
        return 2 * n * n * d + 5 * n * n + 2 * n * n * dv
    if kind in ("clustered", "improved"):
        c = min(clusters, n)
        total = (2 * c * n * d + 5 * c * n + 2 * c * n * dv      # centroid attention
                 + 2 * n * d * bits                             # LSH projection
                 + 2 * n * c * bits * iters)                    # K-means
        if kind == "improved":
            total += 2 * n * top_k * d + 2 * n * top_k * dv + n * c
        return total
    if kind in ("linear", "causal_linear"):
        return 4 * n * d * dv + 2 * n * d
    raise ValueError(f"unknown kind {kind!r}")


def peak_extra_elements(kind: str, n: int, d: int, dv: Optional[int] = None,
                        clusters: int = 32, top_k: int = 32,
                        sram_bytes: int = 64 * 1024, itemsize: int = 8) -> int:
    """Largest intermediate that has to exist at once, in elements.

    Not counting Q, K, V and O themselves, which every method needs.

      softmax        n * n            the score matrix, and in training it must
                                      survive until the backward pass
      clustered      C * n            centroid attention rows
      improved       C * n + n * k
      linear         d * dv           the state; INDEPENDENT OF n
      causal_linear  d * dv           if written as a scan (see linear.py)
      flash          B_r * B_c        one tile, and it lives in SRAM, not HBM
    """
    dv = d if dv is None else dv
    if kind in ("softmax",):
        return n * n
    if kind == "clustered":
        return min(clusters, n) * n
    if kind == "improved":
        return min(clusters, n) * n + n * min(top_k, n)
    if kind in ("linear", "causal_linear"):
        return d * dv
    if kind == "flash":
        from flash import flash_block_sizes
        br, bc = flash_block_sizes(sram_bytes, max(d, dv), itemsize)
        return br * bc
    raise ValueError(f"unknown kind {kind!r}")


def count_hbm_bytes(kind: str, n: int, d: int, sram_bytes: int = 64 * 1024,
                    seed: int = 0) -> int:
    """Measured, not derived: run it on the toy Device and read the counter."""
    from io_model import Device
    from flash import flash_forward, standard_attention_io
    from flash2 import flash2_forward

    Q, K, V = random_qkv(n, d, seed=seed)
    dev = Device(sram_bytes=sram_bytes)
    args = (dev, dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V"))
    if kind == "softmax":
        standard_attention_io(*args)
    elif kind == "flash":
        flash_forward(*args)
    elif kind == "flash2":
        flash2_forward(*args)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return dev.total_bytes


def fit_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Slope of log(y) against log(x): the empirical exponent.

    2.0 means quadratic, 1.0 linear. Fitting in log space is the only honest
    way to read an asymptotic off a handful of measurements.
    """
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    return float(np.polyfit(lx, ly, 1)[0])


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_time(kind: str, n: int, d: int, clusters: int = 32, top_k: int = 32,
                 seed: int = 0, repeats: int = 3) -> float:
    """Wall-clock seconds for one forward pass at this size."""
    Q, K, V = random_qkv(n, d, seed=seed)
    runners = {
        "softmax": lambda: attention(Q, K, V),
        "clustered": lambda: clustered_attention(Q, K, V, n_clusters=clusters),
        "improved": lambda: improved_clustered_attention(Q, K, V, n_clusters=clusters,
                                                         top_k=top_k),
        "linear": lambda: linear_attention(Q, K, V),
        "causal_linear": lambda: causal_linear_attention(Q, K, V, method="cumsum"),
        "flash": lambda: flash_attention(Q, K, V),
    }
    if kind not in runners:
        raise ValueError(f"unknown kind {kind!r}")
    return timeit(runners[kind], repeats=repeats)


def scaling_report(ns: Sequence[int] = (128, 256, 512, 1024), d: int = 32,
                   kinds: Sequence[str] = ("softmax", "clustered", "improved",
                                           "linear"),
                   clusters: int = 32, top_k: int = 32) -> Dict[str, Dict]:
    """Time, flops, memory and accuracy for each method across sequence lengths."""
    out: Dict[str, Dict] = {}
    for kind in kinds:
        times, flops, peaks, errors = [], [], [], []
        for n in ns:
            times.append(measure_time(kind, n, d, clusters=clusters, top_k=top_k))
            flops.append(count_flops(kind, n, d, clusters=clusters, top_k=top_k))
            peaks.append(peak_extra_elements(kind, n, d, clusters=clusters,
                                             top_k=top_k))
            Q, K, V = random_qkv(n, d, seed=0)
            exact = attention(Q, K, V)
            approx = {
                "softmax": lambda: exact,
                "clustered": lambda: clustered_attention(Q, K, V, n_clusters=clusters),
                "improved": lambda: improved_clustered_attention(
                    Q, K, V, n_clusters=clusters, top_k=top_k),
                "linear": lambda: linear_attention(Q, K, V),
                "causal_linear": lambda: exact,
                "flash": lambda: flash_attention(Q, K, V)[0],
            }[kind]()
            errors.append(rel_error(approx, exact))
        out[kind] = {"ns": list(ns), "time": times, "flops": flops,
                     "peak": peaks, "error": errors,
                     "time_slope": fit_slope(ns, times),
                     "flop_slope": fit_slope(ns, flops),
                     "peak_slope": fit_slope(ns, [max(p, 1) for p in peaks])}
    return out


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
