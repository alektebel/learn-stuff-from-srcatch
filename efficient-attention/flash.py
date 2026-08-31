"""
Steps 11-15 — FlashAttention
=============================
Paper: Dao, Fu, Ermon, Rudra, Re, "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness", NeurIPS 2022 (arXiv:2205.14135).

Nothing in this file changes what attention computes. Every function here
returns, up to floating-point reassociation, exactly softmax(QK^T/sqrt(d))V.
What changes is how many bytes cross the boundary between big-slow memory
(HBM) and small-fast memory (SRAM).

That is why this half of the directory runs on the `Device` from io_model.py.
There is no GPU here, and a profiler would tell you what happened without
telling you why; the Device charges for every transfer and refuses to hold more
than its SRAM budget, so the claim "flash moves fewer bytes" becomes a number
you produce rather than a sentence you read.

Effort: step 11 is long but mechanical, step 12 is short and is the idea, step
13 is where they combine. Steps 14-15 are variations once 13 works.

What you build:
  standard_attention_io   -> the baseline, with its IO counted
  online_softmax_update   -> the streaming rescale, the heart of the method
  online_softmax          -> one pass over a row, in blocks
  flash_block_sizes       -> tile sizes DERIVED from the SRAM budget
  flash_forward           -> the tiled forward pass, exact
  flash_backward          -> gradients by recomputation, no P stored
  (causal support in both, with whole tiles skipped)
"""

import math
from typing import Optional, Tuple

import numpy as np

from io_model import Device, HBMTensor


# ---------------------------------------------------------------------------
# Step 11 — standard attention, written so that its IO can be counted
# ---------------------------------------------------------------------------

def standard_attention_io(dev: Device, Qh: HBMTensor, Kh: HBMTensor,
                          Vh: HBMTensor, block: Optional[int] = None,
                          scale: Optional[float] = None) -> HBMTensor:
    """Exactly what a pre-Flash framework does: three kernels, N x M in HBM.

    Background:
      matmul -> softmax -> matmul. Separate kernel launches cannot keep
      anything on chip between them, so the score matrix is written to HBM by
      one and read back by the next. Writing this out is what makes the
      FlashAttention comparison honest: you will have measured the baseline
      rather than quoted it.

    The passes, and what each costs in HBM traffic:

      1. S = Q K^T / sqrt(d)          write N*M     (+ re-read K per row block)
      2. row maxima of S              read  N*M
      3. row sums of exp(S - m)       read  N*M
      4. P = exp(S - m) / l           read  N*M, write N*M
      5. O = P V                      read  N*M     (+ re-read V per row block)

    Passes 2 and 3 are separate because doing them in one pass needs the
    rescaling trick of step 12 — which is precisely the observation
    FlashAttention is built on. And the quadratic term is not only traffic: S
    is ALLOCATED in HBM, so the footprint is quadratic too, and in training it
    stays allocated until the backward pass consumes it.

    TODO:
    1. Pick `block` so that a row block of Q, a column block of K and the score
       tile fit in dev.sram_bytes. Something like
       sram_elements // (4 * max(d, dv)) is fine — this is the baseline, not
       the thing being optimised.
    2. Allocate in HBM: S (n, m) — the quadratic one — plus O, row_max, row_sum.
    3. Write the five passes above. Inside each, use
           with dev.scope() as sram:
               ... sram.load(tensor, rows=..., cols=...) ...
               ... sram.store(tensor, value, rows=..., cols=...) ...
       and open a NESTED scope inside the inner loop, so each tile is freed
       when the iteration ends. Everything loaded in a scope stays resident
       until it exits; forget the nesting and you will get SRAMOverflow, which
       is the accountant telling you something true about the kernel you just
       described.
    4. Return the output tensor.

    The checker asserts that you moved more than 3 N^2-sized blocks and that
    the largest tensor you allocated is at least N^2. Both are properties of
    standard attention, not mistakes.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 12 — online softmax
# ---------------------------------------------------------------------------

def online_softmax_update(m_prev, l_prev, acc_prev, s_block, v_block):
    """Fold one block of scores into a running softmax-weighted sum.

    This is the whole idea. Everything else in FlashAttention is scheduling.

    Background:
      Softmax needs the maximum over a row before it can exponentiate
      anything, and the maximum is only known after the whole row has been
      seen. That is why standard attention makes several passes. The way out:
      keep a PROVISIONAL maximum and correct the accumulation when it changes.

      State per row: the running max m, the running sum of exponentials l, and
      the running UNNORMALISED output acc = sum_j exp(s_j - m) v_j.

      The identity:

          sum exp(s - m_old) = exp(m_old - m_new) ... rescales to m_new

      so when a new block raises the maximum, everything accumulated so far is
      corrected by ONE scalar multiply per row. Nothing is recomputed, and
      every exponent stays <= 0, so nothing can overflow.

    TODO:
    1. m_block = max of s_block over the last axis; m_new = max(m_prev, m_block).
    2. correction = exp(m_prev - m_new).
    3. p = exp(s_block - m_new[:, None]).
    4. l_new = correction * l_prev + p.sum(-1)
       acc_new = correction[:, None] * acc_prev + p @ v_block
    5. Return (m_new, l_new, acc_new). The caller divides acc by l ONCE, at the
       very end.

    Watch the fully-masked case: if m_prev and m_new are both -inf,
    exp(-inf - -inf) is nan. Replace a non-finite m_new by 0.0 when forming the
    exponents, so an all -inf row keeps correction 0 and acc 0.
    """
    raise NotImplementedError


def online_softmax(scores, values, block: int = 8):
    """Stream a full row of scores in blocks; return (output, logsumexp).

    TODO: initialise m = -inf, l = 0, acc = 0; loop over column blocks calling
    online_softmax_update; return acc / l and m + log(l).

    Test: the answer must be identical for block = 1, 5, 7 and the full width,
    must not change when 800 is added to every score, and must not depend on
    the ORDER the blocks arrive in.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 13 — the forward pass
# ---------------------------------------------------------------------------

def flash_block_sizes(sram_bytes: int, d: int, itemsize: int = 8,
                      stage: str = "forward") -> Tuple[int, int]:
    """Block sizes DERIVED from the SRAM budget — not chosen, derived.

    Background:
      Forward (paper, Algorithm 1): K_j and V_j (2 B_c d), Q_i and O_i
      (2 B_r d) and the score tile (B_r B_c) must all be resident at once, so
      with M the SRAM capacity in elements

          B_c = ceil(M / 4d),   B_r = min(B_c, d)

      B_r is capped at d so the score tile stays within M/4. This is the one
      place where a hardware number enters the algorithm, and it is why
      FlashAttention is a *co-designed* algorithm rather than a better formula.

    TODO — return (B_r, B_c) for each stage:
      "forward"   as above.
      "fa2"       FlashAttention-2 keeps the accumulator on chip for the whole
                  inner loop and wants square tiles: the largest B with
                  4Bd + B^2 <= M, i.e. floor(sqrt(4d^2 + M) - 2d).
      "backward"  twice as many tiles are live (dQ, dK, dV, dO and a second
                  score-shaped tile for dS), so use M / 10d in place of M / 4d.
                  Real kernels do the same: backward tiles are smaller.

    Keep about an eighth of the budget in reserve for the per-row m and l and
    for temporaries — the paper's formula is asymptotic, io_model.py is not.
    """
    raise NotImplementedError


def flash_forward(dev: Device, Qh: HBMTensor, Kh: HBMTensor, Vh: HBMTensor,
                  causal: bool = False, scale: Optional[float] = None,
                  block_q: Optional[int] = None, block_k: Optional[int] = None
                  ) -> Tuple[HBMTensor, HBMTensor]:
    """FlashAttention-1 forward. Returns (O, L) in HBM; L is the row logsumexp.

    Background — the loop order is the paper's, and it matters:
      OUTER over K/V blocks, INNER over Q blocks. K_j and V_j are therefore
      loaded once each, while the running state for every query block — O_i,
      l_i, m_i — is read from HBM and written back on EVERY outer iteration.
      Counting those round trips is what step 16 is about, where
      FlashAttention-2 swaps the loops to remove them.

      HBM traffic comes out at Theta(N^2 d^2 / M) against standard attention's
      Theta(N^2): the same asymptotic in N, smaller by the factor d^2/M. The
      other win is not asymptotic at all and matters more in practice — the
      N x N matrix is never allocated, so memory goes from quadratic to linear.

      Store L = m + log(l), one number per query. It is everything the backward
      pass needs to reconstruct P exactly, and it is O(N) rather than O(N^2).

    TODO:
    1. B_r, B_c from flash_block_sizes(dev.sram_bytes, max(d, dv), itemsize).
    2. Allocate O (n, dv), L (n,), and running m (init -inf) and l (init 0).
    3. for each K/V block j:
         with dev.scope() as sram: load k, v ONCE for the whole inner loop
         for each Q block i:
           if causal and the whole tile is in the future (cj.start > ri.stop-1):
               dev.count("blocks_skipped"); continue
           dev.count("blocks_computed")
           with a NESTED scope: load q, o, l_i, m_i; s = q @ k.T * scale;
           if causal, mask entries with key index > query index to -inf;
           update with online_softmax_update, remembering that FA-1 keeps O
           NORMALISED in HBM: pass it o * l_i, and divide the result by l_new
           before storing.
           Count dev.count("o_elementwise_ops", 3 * rows * dv): un-normalise,
           rescale, re-normalise. Step 16 removes two of the three, and the
           comparison only exists if you count them here.
    4. Afterwards write L = m + log(l), guarding rows where l == 0.

    Test: error vs plain attention below 1e-12 (this is EXACT, not an
    approximation), L equal to the logsumexp, and no HBM tensor bigger than
    O(N*d).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 14 — the backward pass, by recomputation
# ---------------------------------------------------------------------------

def flash_backward(dev: Device, Qh, Kh, Vh, Oh, Lh, dOh, causal: bool = False,
                   scale: Optional[float] = None, block_q: Optional[int] = None,
                   block_k: Optional[int] = None):
    """Gradients without ever storing P. Returns (dQ, dK, dV) in HBM.

    Background:
      Standard backward reads the saved N x N probability matrix. Flash
      recomputes each tile from Q, K and the stored L:

          P_ij = exp(Q_i K_j^T / sqrt(d) - L_i)

      which is EXACT, because L_i is precisely the log normaliser of row i.
      The trade is arithmetic for memory — a second QK^T over the whole matrix
      — and on a memory-bound kernel it is a good trade: the recomputed tile
      never leaves SRAM, while the stored one would have cost two N^2 HBM round
      trips.

      The second half of the trick is D_i = sum_k dO_ik O_ik, which equals
      sum_k P_ik dP_ik (you proved this in step 2). One vector of length N
      replaces a second N x N intermediate.

      Note what the loop order costs here: dK_j and dV_j accumulate entirely in
      SRAM across the inner loop, but dQ_i is read and written once per (i, j)
      pair, because the outer loop is over j. On a GPU that is an atomicAdd.
      Step 18 gets rid of it.

    TODO:
    1. B_r, B_c from flash_block_sizes(..., stage="backward").
    2. Allocate dQ, dK, dV and D in HBM.
    3. Pass 0: D = rowsum(dO * O), one row block at a time.
    4. for each K/V block j:  load k, v; zero dk, dv accumulators in SRAM
         for each Q block i (skipping fully-masked tiles when causal):
           load q, dO_i, L_i, D_i, and dQ_i;
           s = q @ k.T * scale, masked if causal;  p = exp(s - L_i[:, None])
           dv_acc += p.T @ dO_i
           dp = dO_i @ v.T
           ds = p * (dp - D_i[:, None])
           dk += ds.T @ q * scale
           store dQ_i + ds @ k * scale     and count dev.count("dq_stores")
         store dk, dv
    5. Return (dQ, dK, dV).

    Test: matches baseline.attention_backward to ~1e-10, and no HBM tensor
    larger than O(N*d) — if P is anywhere in HBM, the exercise failed.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Convenience wrappers (plain numpy in, plain numpy out) — GIVEN
# ---------------------------------------------------------------------------

def flash_attention(Q, K, V, causal=False, scale=None, sram_bytes=64 * 1024,
                    return_device=False):
    dev = Device(sram_bytes=sram_bytes)
    Oh, Lh = flash_forward(dev, dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V"),
                           causal=causal, scale=scale)
    out = (Oh.to_numpy(), Lh.to_numpy())
    return (out, dev) if return_device else out


def flash_attention_backward(dO, Q, K, V, O, L, causal=False, scale=None,
                             sram_bytes=64 * 1024, return_device=False):
    dev = Device(sram_bytes=sram_bytes)
    grads = flash_backward(dev, dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V"),
                           dev.hbm(O, "O"), dev.hbm(L, "L"), dev.hbm(dO, "dO"),
                           causal=causal, scale=scale)
    out = tuple(g.to_numpy() for g in grads)
    return (out, dev) if return_device else out


# ---------------------------------------------------------------------------

def _demo():
    from baseline import attention, attention_backward
    from common import human_bytes, max_abs_error, random_qkv

    n, d = 256, 32
    Q, K, V = random_qkv(n, d, seed=0)
    reference, extras = attention(Q, K, V, return_extras=True)

    std_dev = Device()
    Ostd = standard_attention_io(std_dev, std_dev.hbm(Q, "Q"), std_dev.hbm(K, "K"),
                                 std_dev.hbm(V, "V"))
    (Of, Lf), flash_dev = flash_attention(Q, K, V, return_device=True)

    print(f"N={n}, d={d}, SRAM budget {human_bytes(std_dev.sram_bytes)}\n")
    print("standard attention, in the IO model")
    print(std_dev.report())
    print(f"  max error vs numpy: {max_abs_error(Ostd.to_numpy(), reference):.2e}\n")
    print("FlashAttention-1")
    print(flash_dev.report())
    print(f"  max error vs numpy: {max_abs_error(Of, reference):.2e}")
    print(f"  L vs logsumexp:     {max_abs_error(Lf, extras['lse']):.2e}\n")
    print(f"  HBM traffic ratio: {std_dev.total_bytes / flash_dev.total_bytes:.2f}x less")
    print(f"  largest HBM tensor: {std_dev.largest_allocation} elements "
          f"(standard, = N^2) vs {flash_dev.largest_allocation} (flash, = N*d)")

    dO = np.random.default_rng(3).standard_normal(V.shape)
    (dQ, dK, dV), back_dev = flash_attention_backward(
        dO, Q, K, V, Of, Lf, return_device=True)
    rQ, rK, rV = attention_backward(dO, Q, K, V, extras["P"])
    print(f"\nbackward, no P stored: dQ {max_abs_error(dQ, rQ):.2e}"
          f"  dK {max_abs_error(dK, rK):.2e}  dV {max_abs_error(dV, rV):.2e}")
    print(f"  largest HBM tensor in backward: {back_dev.largest_allocation} elements")

    causal_ref = attention(Q, K, V, causal=True)
    (Oc, _), causal_dev = flash_attention(Q, K, V, causal=True, return_device=True)
    print(f"\ncausal: max error {max_abs_error(Oc, causal_ref):.2e}, "
          f"blocks computed {causal_dev.counters['blocks_computed']}, "
          f"skipped {causal_dev.counters['blocks_skipped']}")


if __name__ == "__main__":
    _demo()
