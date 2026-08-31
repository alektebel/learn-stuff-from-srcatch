"""
Steps 16-18 — FlashAttention-2. Complete solution.

Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work
Partitioning", 2023.

FlashAttention-1 was already IO-optimal to within constants, yet reached only
~25-40% of the GPU's peak matmul throughput. FlashAttention-2 does not change
the algorithm's asymptotics at all. It changes three things about *how the work
is arranged*:

  1. fewer non-matmul operations   — a GPU's tensor cores do matmul ~16x faster
                                      than its ALUs do anything else, so every
                                      rescale of the output accumulator is
                                      expensive out of all proportion to its
                                      flop count;
  2. parallelism over the sequence — FA-1 parallelises over batch x heads, which
                                      is not enough blocks to fill a GPU when
                                      the batch is small and the sequence long;
  3. work partitioning inside a block, so that warps share less through shared
     memory (not modelled here — it lives below this level of abstraction).

This file makes 1 and 2 measurable. 3 is discussed in the README.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from flash import flash_block_sizes, online_softmax_update
from io_model import Device, HBMTensor


# ---------------------------------------------------------------------------
# Step 16 — swap the loops, defer the division
# ---------------------------------------------------------------------------

def flash2_forward(dev: Device, Qh: HBMTensor, Kh: HBMTensor, Vh: HBMTensor,
                   causal: bool = False, scale: Optional[float] = None,
                   block_q: Optional[int] = None, block_k: Optional[int] = None,
                   only_rows: Optional[Sequence[int]] = None
                   ) -> Tuple[HBMTensor, HBMTensor]:
    """FlashAttention-2 forward: OUTER loop over Q blocks, INNER over K/V.

    Two consequences of the swap, both of which the counters will show:

    * The running state (O_i, l_i, m_i) for a query block never leaves SRAM.
      FA-1 reads and writes O_i once per K/V block; here it is written exactly
      once, at the end. Nothing else about the arithmetic changes.

    * The accumulator is kept UNNORMALISED and divided by l_i once, at the end,
      instead of being un-normalised and re-normalised on every inner step.
      Three elementwise passes over the (B_r, d) accumulator become one — and
      elementwise work is the expensive kind on a tensor-core GPU.

    And one consequence the counters cannot show, which is the more important
    one: iterations of the outer loop are now INDEPENDENT. Query block i needs
    nothing from query block i'. So the outer loop can be spread across
    streaming multiprocessors, and a single long sequence with batch size 1
    still fills the GPU. Under FA-1's loop order every row block shares the
    same O accumulator, so the outer loop cannot be parallelised at all.
    `only_rows` exists to make that concrete: compute any subset of the row
    blocks, in any order, and get identical answers.
    """
    n, d = Qh.shape
    m_len = Kh.shape[0]
    dv = Vh.shape[1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    itemsize = np.dtype(Qh.dtype).itemsize
    br, bc = flash_block_sizes(dev.sram_bytes, max(d, dv), itemsize, "fa2")
    br = max(1, min(block_q or br, n))
    bc = max(1, min(block_k or bc, m_len))

    Oh = dev.hbm_zeros((n, dv), dtype=Qh.dtype, name="O")
    Lh = dev.hbm_zeros((n,), dtype=Qh.dtype, name="L")

    row_starts = list(range(0, n, br))
    if only_rows is not None:
        row_starts = [row_starts[i] for i in only_rows]

    for i0 in row_starts:
        ri = slice(i0, min(i0 + br, n))
        rows = ri.stop - ri.start
        with dev.scope() as sram:
            q = sram.load(Qh, rows=ri)
            acc = sram.alloc((rows, dv), dtype=Qh.dtype)      # UNnormalised
            l = sram.alloc((rows,), dtype=Qh.dtype)
            mm = sram.alloc((rows,), dtype=Qh.dtype, fill=-np.inf)

            for j0 in range(0, m_len, bc):
                cj = slice(j0, min(j0 + bc, m_len))
                if causal and cj.start > ri.stop - 1:
                    dev.count("blocks_skipped")
                    continue
                dev.count("blocks_computed")

                with dev.scope() as tile:
                    k = tile.load(Kh, rows=cj)
                    v = tile.load(Vh, rows=cj)
                    s = tile.keep(q @ k.T * scale)
                    if causal:
                        qi = np.arange(ri.start, ri.stop)[:, None]
                        kj = np.arange(cj.start, cj.stop)[None, :]
                        s = np.where(kj <= qi, s, -np.inf)

                    mm, l, acc = online_softmax_update(mm, l, acc, s, v)
                    dev.count("o_elementwise_ops", rows * dv)   # one rescale

            safe_l = np.where(l > 0, l, 1.0)
            dev.count("o_elementwise_ops", rows * dv)           # one division
            sram.store(Oh, acc / safe_l[:, None], rows=ri)
            sram.store(Lh, np.where(l > 0, mm + np.log(safe_l), -np.inf), rows=ri)

    return Oh, Lh


# ---------------------------------------------------------------------------
# Step 17 — splitting the KEY dimension, and merging partial softmaxes
# ---------------------------------------------------------------------------

def flash2_partial(dev: Device, Qh: HBMTensor, Kh: HBMTensor, Vh: HBMTensor,
                   key_range: slice, causal: bool = False,
                   scale: Optional[float] = None, block_k: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attend to only keys[key_range]. Returns (O_partial, m_partial, l_partial).

    Row blocks give parallelism proportional to N/B_r, which is plenty when
    training. It is *not* plenty when generating one token at a time: N = 1,
    one row block, one SM busy and the rest idle, on a problem that is entirely
    memory bound. Splitting the KEY axis instead gives back as much parallelism
    as you want. This is what FlashAttention-2's work partitioning buys, and
    what flash-decoding does at inference time.

    The partial results cannot simply be added: each is normalised by its own
    local softmax denominator. Return the local (m, l) alongside, and the merge
    in `combine_partials` is exact.
    """
    n, d = Qh.shape
    dv = Vh.shape[1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    itemsize = np.dtype(Qh.dtype).itemsize
    br, bc = flash_block_sizes(dev.sram_bytes, max(d, dv), itemsize, "fa2")
    bc = max(1, min(block_k or bc, key_range.stop - key_range.start))

    O = np.zeros((n, dv), dtype=Qh.dtype)
    M = np.full(n, -np.inf, dtype=Qh.dtype)
    L = np.zeros(n, dtype=Qh.dtype)

    for i0 in range(0, n, br):
        ri = slice(i0, min(i0 + br, n))
        rows = ri.stop - ri.start
        with dev.scope() as sram:
            q = sram.load(Qh, rows=ri)
            acc = sram.alloc((rows, dv), dtype=Qh.dtype)
            l = sram.alloc((rows,), dtype=Qh.dtype)
            mm = sram.alloc((rows,), dtype=Qh.dtype, fill=-np.inf)

            for j0 in range(key_range.start, key_range.stop, bc):
                cj = slice(j0, min(j0 + bc, key_range.stop))
                with dev.scope() as tile:
                    k = tile.load(Kh, rows=cj)
                    v = tile.load(Vh, rows=cj)
                    s = tile.keep(q @ k.T * scale)
                    if causal:
                        qi = np.arange(ri.start, ri.stop)[:, None]
                        kj = np.arange(cj.start, cj.stop)[None, :]
                        s = np.where(kj <= qi, s, -np.inf)
                    mm, l, acc = online_softmax_update(mm, l, acc, s, v)

            safe_l = np.where(l > 0, l, 1.0)
            O[ri] = acc / safe_l[:, None]
            M[ri], L[ri] = mm, l

    return O, M, L


def combine_partials(partials: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Merge independently computed softmax pieces into the exact whole.

        m = max_p m_p
        w_p = exp(m_p - m) * l_p            the true weight of piece p
        O   = sum_p w_p O_p / sum_p w_p
        L   = m + log(sum_p w_p)

    This is the online-softmax update of step 12, applied to whole partial
    results rather than to a single block — and it is what makes attention
    *associative* over key ranges. The same merge underlies flash-decoding,
    ring attention across devices, and paged/chunked KV caches: every one of
    them is "compute attention over a slice, keep (m, l), combine later".
    """
    ms = np.stack([p[1] for p in partials])                  # (P, N)
    ls = np.stack([p[2] for p in partials])                  # (P, N)
    os_ = np.stack([p[0] for p in partials])                 # (P, N, dv)

    m = np.max(np.where(ls > 0, ms, -np.inf), axis=0)        # (N,)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    w = np.where(ls > 0, np.exp(ms - safe_m) * ls, 0.0)      # (P, N)
    total = w.sum(axis=0)                                    # (N,)
    safe_total = np.where(total > 0, total, 1.0)
    O = (w[:, :, None] * os_).sum(axis=0) / safe_total[:, None]
    L = np.where(total > 0, safe_m + np.log(safe_total), -np.inf)
    return O, L


# ---------------------------------------------------------------------------
# Step 18 — the backward pass without read-modify-write on dQ
# ---------------------------------------------------------------------------

def flash2_backward(dev: Device, Qh, Kh, Vh, Oh, Lh, dOh, causal: bool = False,
                    scale: Optional[float] = None, block_q: Optional[int] = None,
                    block_k: Optional[int] = None):
    """Two independent passes, so no gradient is ever accumulated through HBM.

    The problem with FA-1's backward: its outer loop is over K/V blocks, so
    dQ_i is touched by every one of them. Each touch is a read-modify-write of
    HBM — an atomicAdd on a real GPU, or a pre-split buffer plus a reduction.
    dK_j and dV_j have no such problem: they stay in SRAM for the whole inner
    loop.

    The fix is to run the loop in both orders, once each:

      pass A (outer over Q blocks): dQ_i = sum_j dS_ij K_j       -> one store
      pass B (outer over K/V blocks): dK_j, dV_j                 -> one store

    Both passes recompute S and P from Q, K and L. Recomputing the score tiles
    twice sounds wasteful and is not: the tiles never leave SRAM, both passes
    are matmul-heavy, and what is bought is that every gradient is written to
    HBM exactly once and every block of both loops is independent — which is
    the same parallelism argument as step 16, now for the backward pass.

    Counting `dq_stores` is the whole point of this step: FA-1 does one per
    (i, j) pair, this does one per row block.
    """
    n, d = Qh.shape
    m_len = Kh.shape[0]
    dv = Vh.shape[1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    itemsize = np.dtype(Qh.dtype).itemsize
    br, bc = flash_block_sizes(dev.sram_bytes, max(d, dv), itemsize, "backward")
    br = max(1, min(block_q or br, n))
    bc = max(1, min(block_k or bc, m_len))

    dQh = dev.hbm_zeros((n, d), dtype=Qh.dtype, name="dQ")
    dKh = dev.hbm_zeros((m_len, d), dtype=Qh.dtype, name="dK")
    dVh = dev.hbm_zeros((m_len, dv), dtype=Qh.dtype, name="dV")
    Dh = dev.hbm_zeros((n,), dtype=Qh.dtype, name="D")

    for i0 in range(0, n, br):
        ri = slice(i0, min(i0 + br, n))
        with dev.scope() as sram:
            sram.store(Dh, np.sum(sram.load(dOh, rows=ri) * sram.load(Oh, rows=ri),
                                  axis=-1), rows=ri)

    def masked_scores(q, ri, cj, k):
        s = q @ k.T * scale
        if causal:
            qi = np.arange(ri.start, ri.stop)[:, None]
            kj = np.arange(cj.start, cj.stop)[None, :]
            s = np.where(kj <= qi, s, -np.inf)
        return s

    # ---- pass A: dQ, outer loop over query blocks --------------------------
    for i0 in range(0, n, br):
        ri = slice(i0, min(i0 + br, n))
        with dev.scope() as sram:
            q = sram.load(Qh, rows=ri)
            do = sram.load(dOh, rows=ri)
            lse = sram.load(Lh, rows=ri)
            D = sram.load(Dh, rows=ri)
            dq = sram.alloc((ri.stop - ri.start, d), dtype=Qh.dtype)

            for j0 in range(0, m_len, bc):
                cj = slice(j0, min(j0 + bc, m_len))
                if causal and cj.start > ri.stop - 1:
                    continue
                with dev.scope() as tile:
                    k = tile.load(Kh, rows=cj)
                    v = tile.load(Vh, rows=cj)
                    p = np.exp(tile.keep(masked_scores(q, ri, cj, k)) - lse[:, None])
                    ds = p * (tile.keep(do @ v.T) - D[:, None])
                    dq += ds @ k * scale

            sram.store(dQh, dq, rows=ri)
            dev.count("dq_stores")

    # ---- pass B: dK, dV, outer loop over key blocks ------------------------
    for j0 in range(0, m_len, bc):
        cj = slice(j0, min(j0 + bc, m_len))
        with dev.scope() as sram:
            k = sram.load(Kh, rows=cj)
            v = sram.load(Vh, rows=cj)
            dk = sram.alloc((cj.stop - cj.start, d), dtype=Qh.dtype)
            dv_acc = sram.alloc((cj.stop - cj.start, dv), dtype=Qh.dtype)

            for i0 in range(0, n, br):
                ri = slice(i0, min(i0 + br, n))
                if causal and cj.start > ri.stop - 1:
                    continue
                with dev.scope() as tile:
                    q = tile.load(Qh, rows=ri)
                    do = tile.load(dOh, rows=ri)
                    lse = tile.load(Lh, rows=ri)
                    D = tile.load(Dh, rows=ri)
                    p = np.exp(tile.keep(masked_scores(q, ri, cj, k)) - lse[:, None])
                    dv_acc += p.T @ do
                    ds = p * (tile.keep(do @ v.T) - D[:, None])
                    dk += ds.T @ q * scale

            sram.store(dKh, dk, rows=cj)
            sram.store(dVh, dv_acc, rows=cj)

    return dQh, dKh, dVh


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def flash2_attention(Q, K, V, causal=False, scale=None, sram_bytes=64 * 1024,
                     return_device=False):
    dev = Device(sram_bytes=sram_bytes)
    Oh, Lh = flash2_forward(dev, dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V"),
                            causal=causal, scale=scale)
    out = (Oh.to_numpy(), Lh.to_numpy())
    return (out, dev) if return_device else out


def flash2_attention_backward(dO, Q, K, V, O, L, causal=False, scale=None,
                              sram_bytes=64 * 1024, return_device=False):
    dev = Device(sram_bytes=sram_bytes)
    grads = flash2_backward(dev, dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V"),
                            dev.hbm(O, "O"), dev.hbm(L, "L"), dev.hbm(dO, "dO"),
                            causal=causal, scale=scale)
    out = tuple(g.to_numpy() for g in grads)
    return (out, dev) if return_device else out


# ---------------------------------------------------------------------------

def _demo():
    from baseline import attention, attention_backward
    from common import human_bytes, max_abs_error, random_qkv
    from flash import flash_attention, flash_attention_backward

    n, d = 256, 32
    Q, K, V = random_qkv(n, d, seed=0)
    reference, extras = attention(Q, K, V, return_extras=True)

    (O1, L1), dev1 = flash_attention(Q, K, V, return_device=True)
    (O2, L2), dev2 = flash2_attention(Q, K, V, return_device=True)

    print(f"N={n}, d={d}\n")
    print(f"{'':<22}{'FlashAttention-1':>18}{'FlashAttention-2':>18}")
    print(f"{'HBM traffic':<22}{human_bytes(dev1.total_bytes):>18}"
          f"{human_bytes(dev2.total_bytes):>18}")
    print(f"{'elementwise ops on O':<22}{dev1.counters['o_elementwise_ops']:>18,}"
          f"{dev2.counters['o_elementwise_ops']:>18,}")
    print(f"{'error vs numpy':<22}{max_abs_error(O1, reference):>18.2e}"
          f"{max_abs_error(O2, reference):>18.2e}")
    print(f"\nsame answer as each other: {np.allclose(O1, O2)}  "
          f"(same L: {np.allclose(L1, L2)})")

    # the outer loop is parallel: compute row blocks in any order, alone
    dev = Device()
    Qh, Kh, Vh = dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V")
    n_blocks = len(range(0, n, flash_block_sizes(dev.sram_bytes, d, 8, "fa2")[0]))
    pieces = [flash2_forward(Device(), Qh, Kh, Vh, only_rows=[i])[0].to_numpy()
              for i in reversed(range(n_blocks))]
    stitched = sum(pieces)      # each piece is zero outside its own row block
    print(f"row blocks computed separately, in reverse: "
          f"{max_abs_error(stitched, reference):.2e}")

    # splitting the key axis, flash-decoding style
    splits = [slice(0, 64), slice(64, 128), slice(128, 256)]
    partials = [flash2_partial(Device(), Qh, Kh, Vh, s) for s in splits]
    Oc, Lc = combine_partials(partials)
    print(f"3 uneven key splits, merged: {max_abs_error(Oc, reference):.2e}"
          f"  (L: {max_abs_error(Lc, extras['lse']):.2e})")

    dO = np.random.default_rng(3).standard_normal(V.shape)
    (dQ1, dK1, dV1), b1 = flash_attention_backward(dO, Q, K, V, O1, L1,
                                                   return_device=True)
    (dQ2, dK2, dV2), b2 = flash2_attention_backward(dO, Q, K, V, O2, L2,
                                                    return_device=True)
    rQ, rK, rV = attention_backward(dO, Q, K, V, extras["P"])
    print(f"\nbackward errors vs standard: dQ {max_abs_error(dQ2, rQ):.2e}"
          f"  dK {max_abs_error(dK2, rK):.2e}  dV {max_abs_error(dV2, rV):.2e}")
    print(f"dQ written to HBM: {b1.counters.get('dq_stores', 0)} times (FA-1: "
          f"once per (i,j) pair = {b1.counters['blocks_computed']}), "
          f"{b2.counters['dq_stores']} times (FA-2: once per row block)")

    causal_ref = attention(Q, K, V, causal=True)
    (Oc2, _), devc = flash2_attention(Q, K, V, causal=True, return_device=True)
    print(f"causal: {max_abs_error(Oc2, causal_ref):.2e}, "
          f"{devc.counters['blocks_computed']} blocks computed, "
          f"{devc.counters['blocks_skipped']} skipped")


if __name__ == "__main__":
    _demo()
