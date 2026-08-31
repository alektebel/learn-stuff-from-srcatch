"""
Shared helpers — GIVEN. You do not implement anything in this file.

Everything here is measurement apparatus and toy data, deliberately kept out of
the exercises so the files you do write contain only attention.

Conventions used everywhere in this directory
---------------------------------------------
  N   query length          Q : (N, d)
  M   key/value length      K : (M, d)      V : (M, dv)
  d   head dimension        output          : (N, dv)

Single head, no batch. Multi-head attention is this, run H times on H slices of
the projected tensors; adding that dimension would triple the index bookkeeping
in every file and teach nothing new. Self-attention means N == M.

float64 is the default dtype for correctness work. FlashAttention is exact in
exact arithmetic, and float64 lets the checker demand agreement at 1e-12 rather
than the 1e-6 that float32 rounding would force — the difference between "the
algorithm is right" and "the algorithm is roughly right" is worth 4 bytes.
"""

from typing import Callable, Optional, Tuple

import numpy as np

__all__ = [
    "random_qkv",
    "max_abs_error",
    "rel_error",
    "numerical_gradient",
    "human_bytes",
    "causal_mask",
    "timeit",
]


def random_qkv(
    n: int,
    d: int,
    dv: Optional[int] = None,
    m: Optional[int] = None,
    seed: int = 0,
    scale: float = 1.0,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random Q, K, V for tests and demos.

    `scale` multiplies the queries and keys only. It is the knob that controls
    how *peaked* the attention distributions are: with scale=1 and d=32 the
    logits have standard deviation ~1 after the 1/sqrt(d) scaling and attention
    is nearly uniform; with scale=3 it is close to a hard argmax. Several
    exercises behave completely differently in those two regimes, which is the
    point of exposing it.
    """
    rng = np.random.default_rng(seed)
    m = n if m is None else m
    dv = d if dv is None else dv
    q = (rng.standard_normal((n, d)) * scale).astype(dtype)
    k = (rng.standard_normal((m, d)) * scale).astype(dtype)
    v = rng.standard_normal((m, dv)).astype(dtype)
    return q, k, v


def clustered_qkv(
    n: int,
    d: int,
    groups: int = 4,
    seed: int = 0,
    spread: float = 0.15,
    scale: float = 1.0,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q, K, V where the queries genuinely fall into `groups` tight clusters.

    Clustered attention is only a good approximation when queries actually
    group; random Gaussian queries in high dimension are all roughly equidistant
    and it does badly on them. Having both generators available lets you measure
    the difference instead of taking it on faith.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((groups, d)) * scale
    which = rng.integers(0, groups, size=n)
    q = (centres[which] + rng.standard_normal((n, d)) * spread).astype(dtype)
    k = (rng.standard_normal((n, d)) * scale).astype(dtype)
    v = rng.standard_normal((n, d)).astype(dtype)
    return q, k, v


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute disagreement between two arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative Frobenius error ||a - b|| / ||b||."""
    b = np.asarray(b)
    denom = float(np.linalg.norm(b))
    return float(np.linalg.norm(np.asarray(a) - b) / (denom + 1e-30))


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function, for gradient checks.

    O(x.size) evaluations of f, so only ever use it on tiny inputs.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f(x)
        x[idx] = original - eps
        minus = f(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def human_bytes(n: float) -> str:
    """1536 -> '1.5 KB'. Byte counts are the currency of the FlashAttention half."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def causal_mask(n: int, m: Optional[int] = None) -> np.ndarray:
    """Boolean (n, m) mask, True where attention is ALLOWED (j <= i)."""
    m = n if m is None else m
    return np.tril(np.ones((n, m), dtype=bool))


def timeit(fn: Callable[[], object], repeats: int = 3) -> float:
    """Best-of-`repeats` wall-clock seconds. Best-of, not mean: we are timing a
    deterministic computation, so the spread is noise from the machine and the
    minimum is the least contaminated estimate."""
    import time

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best
