"""
Steps 16-18 — FlashAttention-2
===============================
Paper: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
Work Partitioning", 2023 (arXiv:2307.08691).

FlashAttention-1 was already IO-optimal to within constants, yet reached only
~25-40% of a GPU's peak matmul throughput. FlashAttention-2 does not change the
algorithm's asymptotics at all. It changes how the work is arranged:

  1. fewer non-matmul operations   — tensor cores do matmul roughly 16x faster
                                     than the ALUs do anything else, so every
                                     rescale of the output accumulator costs
                                     far more than its flop count suggests;
  2. parallelism over the sequence — FA-1 parallelises over batch x heads,
                                     which is not enough blocks to fill a GPU
                                     when the batch is small and the sequence
                                     long;
  3. better work partitioning inside a block, so warps share less through
     shared memory (below the level of abstraction modelled here — see
     THEORY.md).

This file makes 1 and 2 measurable.

Effort: if step 13 works, step 16 is a rearrangement of it. Step 17 is short
and is the most reusable idea in the directory.

What you build:
  flash2_forward     -> loops swapped, one division at the end, parallel blocks
  flash2_partial     -> attention over a slice of the KEYS
  combine_partials   -> merge partial softmaxes exactly
  flash2_backward    -> two passes, so no gradient accumulates through HBM
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

    Background — three consequences of the swap:

      * The running state (O_i, l_i, m_i) for a query block never leaves SRAM.
        FA-1 reads and writes O_i once per K/V block; here it is written
        exactly once, at the end.

      * The accumulator stays UNNORMALISED and is divided by l_i once, at the
        end, instead of being un-normalised and re-normalised on every inner
        step. Three elementwise passes over the (B_r, dv) accumulator become
        one — and elementwise work is the expensive kind on a tensor-core GPU.

      * Iterations of the outer loop are now INDEPENDENT. Query block i needs
        nothing from query block i', so the outer loop can be spread across
        streaming multiprocessors and a single long sequence with batch size 1
        still fills the GPU. Under FA-1's loop order every row block shares the
        same O accumulator, so its outer loop cannot be parallelised at all.
        This is the change that matters most, and it is invisible in a flop
        count.

    TODO:
    1. B_r, B_c from flash_block_sizes(..., stage="fa2").
    2. for each Q block i (or only those in `only_rows`, which exists to make
       the independence testable — compute any subset, in any order, and the
       answers must be identical):
         with dev.scope() as sram:
           load q once; allocate acc (rows, dv), l (rows,), m (rows,) = -inf
           for each K/V block j (skipping fully-masked tiles when causal):
             nested scope: load k, v; s = q @ k.T * scale, masked if causal;
             (m, l, acc) = online_softmax_update(m, l, acc, s, v)
             dev.count("o_elementwise_ops", rows * dv)      # one rescale
           store O_i = acc / l  and  L_i = m + log(l)
           dev.count("o_elementwise_ops", rows * dv)        # the one division
    3. Return (O, L).

    Test: identical numbers to flash_forward, strictly fewer elementwise ops,
    strictly fewer bytes written.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 17 — splitting the KEY dimension, and merging partial softmaxes
# ---------------------------------------------------------------------------

def flash2_partial(dev: Device, Qh: HBMTensor, Kh: HBMTensor, Vh: HBMTensor,
                   key_range: slice, causal: bool = False,
                   scale: Optional[float] = None, block_k: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attend to only keys[key_range]. Returns (O_partial, m_partial, l_partial).

    Background — why the key axis, when step 16 already gave parallelism:
      Row blocks give N/B_r independent units, which is plenty when training.
      It is not plenty when GENERATING one token at a time: N = 1, one row
      block, one SM busy and the rest of the GPU idle, on a problem that is
      entirely memory bound. Splitting the KEY axis instead gives back as much
      parallelism as you want. This is flash-decoding, and it is the same
      partitioning idea FlashAttention-2 applies inside a block.

    TODO: exactly flash2_forward's body, but with the inner loop restricted to
    `key_range`, and returning plain numpy (O, m, l) instead of writing to HBM.
    Block over query rows as usual so the SRAM budget still holds.

    Do NOT normalise away m and l — they are what makes the pieces mergeable.
    """
    raise NotImplementedError


def combine_partials(partials: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Merge independently computed softmax pieces into the exact whole.

        m   = max_p m_p
        w_p = exp(m_p - m) * l_p            the true weight of piece p
        O   = sum_p w_p O_p / sum_p w_p
        L   = m + log(sum_p w_p)

    Background:
      This is step 12's update applied to whole partial results rather than to
      single blocks, and it is what makes attention ASSOCIATIVE over key
      ranges. The same merge underlies flash-decoding, ring attention across
      devices, and paged/chunked KV caches: every one of them is "compute
      attention over a slice, keep (m, l), combine later".

    TODO: stack the pieces, then the four lines above.

    Edge case the checker exercises: under a causal mask some pieces are
    entirely in the future for some queries, so they arrive with l = 0 and
    m = -inf. They must contribute nothing and must not produce nan.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 18 — the backward pass without read-modify-write on dQ
# ---------------------------------------------------------------------------

def flash2_backward(dev: Device, Qh, Kh, Vh, Oh, Lh, dOh, causal: bool = False,
                    scale: Optional[float] = None, block_q: Optional[int] = None,
                    block_k: Optional[int] = None):
    """Two independent passes, so no gradient is ever accumulated through HBM.

    Background — the problem with FA-1's backward:
      Its outer loop is over K/V blocks, so dQ_i is touched by every one of
      them. Each touch is a read-modify-write of HBM: an atomicAdd on a real
      GPU, or a pre-split buffer plus a reduction. dK_j and dV_j have no such
      problem — they stay in SRAM for the whole inner loop.

      The fix is to run the loop in both orders, once each:

        pass A (outer over Q blocks):    dQ_i = sum_j dS_ij K_j   -> one store
        pass B (outer over K/V blocks):  dK_j, dV_j               -> one store

      Both passes recompute S and P from Q, K and L. Recomputing the score
      tiles twice sounds wasteful and is not: the tiles never leave SRAM, both
      passes are matmul-heavy, and what is bought is that every gradient is
      written to HBM exactly once and every block of both loops is independent.

    TODO:
    1. D = rowsum(dO * O) as before.
    2. Pass A: for each Q block, accumulate dq over all K/V blocks in SRAM,
       store once, and count dev.count("dq_stores").
    3. Pass B: for each K/V block, accumulate dk and dv over all Q blocks in
       SRAM, store once.
    4. Return (dQ, dK, dV).

    Test: identical gradients to flash_backward (causal and not), and
    dq_stores one per ROW BLOCK rather than one per (i, j) PAIR.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Convenience wrappers — GIVEN
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

    dev = Device()
    Qh, Kh, Vh = dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V")
    n_blocks = math.ceil(n / flash_block_sizes(dev.sram_bytes, d, 8, "fa2")[0])
    pieces = [flash2_forward(Device(), Qh, Kh, Vh, only_rows=[i])[0].to_numpy()
              for i in reversed(range(n_blocks))]
    print(f"row blocks computed separately, in reverse: "
          f"{max_abs_error(sum(pieces), reference):.2e}")

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
