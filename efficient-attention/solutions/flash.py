"""
Steps 11-15 — FlashAttention. Complete solution.

Dao, Fu, Ermon, Rudra, Re, "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness", NeurIPS 2022.

Nothing in this file changes what attention computes. Every function here
returns, up to floating-point reassociation, exactly softmax(QK^T/sqrt(d))V.
What changes is how many bytes cross the HBM boundary, and that is measured
rather than asserted: every implementation runs on the `Device` from
io_model.py, which charges for each transfer and refuses to hold more than its
SRAM budget.
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

    matmul -> softmax -> matmul, with the score matrix written to HBM by one
    kernel and read back by the next, because separate kernel launches cannot
    keep anything on chip between them.

    The passes, and what each one costs in HBM traffic:

      1. S = Q K^T / sqrt(d)          write N*M      (+ re-read K per row block)
      2. row maxima of S              read  N*M
      3. row sums of exp(S - m)       read  N*M
      4. P = exp(S - m) / l           read  N*M, write N*M
      5. O = P V                      read  N*M      (+ re-read V per row block)

    Six passes over an N x M array. Passes 2 and 3 are separate because a
    single pass would need the rescaling trick of step 12 — which is exactly
    the observation FlashAttention is built on.

    The quadratic term is not just traffic: `S` is *allocated* in HBM, so the
    memory footprint is quadratic too, and in training it stays allocated until
    the backward pass has consumed it.
    """
    n, d = Qh.shape
    m = Kh.shape[0]
    dv = Vh.shape[1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    itemsize = np.dtype(Qh.dtype).itemsize
    # A row block small enough that one row block of Q plus one column block of
    # K plus the score tile fit on chip.
    block = block or max(1, int(dev.sram_bytes // itemsize) // (4 * max(d, dv)))
    block = max(1, min(block, n, m))

    S = dev.hbm_zeros((n, m), dtype=Qh.dtype, name="S")     # <- the quadratic tensor
    Oh = dev.hbm_zeros((n, dv), dtype=Qh.dtype, name="O")
    row_max = dev.hbm_zeros((n,), dtype=Qh.dtype, name="m")
    row_sum = dev.hbm_zeros((n,), dtype=Qh.dtype, name="l")

    row_blocks = [slice(i, min(i + block, n)) for i in range(0, n, block)]
    col_blocks = [slice(j, min(j + block, m)) for j in range(0, m, block)]

    # pass 1 -- scores
    for ri in row_blocks:
        with dev.scope() as sram:
            q = sram.load(Qh, rows=ri)
            for cj in col_blocks:
                with dev.scope() as tile:      # freed each inner iteration
                    k = tile.load(Kh, rows=cj)
                    tile.store(S, tile.keep(q @ k.T * scale), rows=ri, cols=cj)

    # pass 2 -- row maxima
    for ri in row_blocks:
        with dev.scope() as sram:
            best = sram.alloc((ri.stop - ri.start,), dtype=Qh.dtype, fill=-np.inf)
            for cj in col_blocks:
                with dev.scope() as tile:
                    best = np.maximum(best, tile.load(S, rows=ri, cols=cj).max(axis=-1))
            sram.store(row_max, best, rows=ri)

    # pass 3 -- row sums of exp(S - m)
    for ri in row_blocks:
        with dev.scope() as sram:
            mrow = sram.load(row_max, rows=ri)
            total = sram.alloc((ri.stop - ri.start,), dtype=Qh.dtype)
            for cj in col_blocks:
                with dev.scope() as tile:
                    s = tile.load(S, rows=ri, cols=cj)
                    total = total + np.exp(s - mrow[:, None]).sum(axis=-1)
            sram.store(row_sum, total, rows=ri)

    # pass 4 -- normalise in place: S becomes P
    for ri in row_blocks:
        with dev.scope() as sram:
            mrow = sram.load(row_max, rows=ri)
            lrow = sram.load(row_sum, rows=ri)
            for cj in col_blocks:
                with dev.scope() as tile:
                    s = tile.load(S, rows=ri, cols=cj)
                    tile.store(S, np.exp(s - mrow[:, None]) / lrow[:, None],
                               rows=ri, cols=cj)

    # pass 5 -- output
    for ri in row_blocks:
        with dev.scope() as sram:
            acc = sram.alloc((ri.stop - ri.start, dv), dtype=Qh.dtype)
            for cj in col_blocks:
                with dev.scope() as tile:
                    p = tile.load(S, rows=ri, cols=cj)
                    v = tile.load(Vh, rows=cj)
                    acc = acc + p @ v
            sram.store(Oh, acc, rows=ri)

    return Oh


# ---------------------------------------------------------------------------
# Step 12 — online softmax
# ---------------------------------------------------------------------------

def online_softmax_update(m_prev, l_prev, acc_prev, s_block, v_block):
    """Fold one block of scores into a running softmax-weighted sum.

    State: the running maximum m, the running sum of exponentials l, and the
    running UNNORMALISED output acc = sum_j exp(s_j - m) v_j.

    The identity that makes it work: an exponential sum written relative to one
    maximum can be rewritten relative to another for the price of a single
    multiply,

        sum exp(s - m_old) = exp(m_new - m_old)^{-1} * sum exp(s - m_new)

    so when a new block raises the maximum, everything accumulated so far is
    corrected by one scalar per row. Nothing is recomputed, and nothing is left
    in a form that could overflow: every exponent is <= 0 at all times.

    Returns (m_new, l_new, acc_new). Divide acc by l at the very end — once —
    to get the answer.
    """
    s_block = np.asarray(s_block)
    m_block = np.max(s_block, axis=-1)
    m_block = np.where(np.isfinite(m_block), m_block, -np.inf)
    m_new = np.maximum(m_prev, m_block)
    # A row still entirely -inf (fully masked so far) would give exp(-inf - -inf)
    # = nan; force its correction factor to zero instead.
    safe = np.where(np.isfinite(m_new), m_new, 0.0)

    correction = np.exp(np.where(np.isfinite(m_prev), m_prev, -np.inf) - safe)
    p = np.exp(s_block - safe[:, None])
    l_new = correction * l_prev + p.sum(axis=-1)
    acc_new = correction[:, None] * acc_prev + p @ np.asarray(v_block)
    return m_new, l_new, acc_new


def online_softmax(scores, values, block: int = 8):
    """Reference use of the update rule: stream the row in blocks, one pass."""
    scores, values = np.asarray(scores), np.asarray(values)
    n, m = scores.shape
    m_run = np.full(n, -np.inf, dtype=scores.dtype)
    l_run = np.zeros(n, dtype=scores.dtype)
    acc = np.zeros((n, values.shape[1]), dtype=np.result_type(scores, values))
    for j in range(0, m, block):
        sl = slice(j, min(j + block, m))
        m_run, l_run, acc = online_softmax_update(
            m_run, l_run, acc, scores[:, sl], values[sl])
    return acc / l_run[:, None], m_run + np.log(l_run)


# ---------------------------------------------------------------------------
# Step 13 — the forward pass
# ---------------------------------------------------------------------------

def flash_block_sizes(sram_bytes: int, d: int, itemsize: int = 8,
                      stage: str = "forward") -> Tuple[int, int]:
    """Block sizes derived from the SRAM budget — not chosen, derived.

    Forward (FlashAttention Algorithm 1): K_j and V_j (2 B_c d), Q_i and O_i
    (2 B_r d) and the score tile (B_r B_c) must be resident at once, giving

        B_c = ceil(M / 4d),   B_r = min(B_c, d)

    with M the SRAM capacity in elements. B_r is capped at d so the score tile
    B_r x B_c stays within M/4.

    "fa2": FlashAttention-2 keeps the output accumulator on chip for the whole
    inner loop, so it wants square tiles — the largest B with 4Bd + B^2 <= M.

    "backward": twice as many tiles are live (dQ, dK, dV, dO, and a second
    score-shaped tile for dS), so the divisor grows. Real kernels do the same
    thing, which is why backward tiles are smaller than forward ones.

    A one-eighth headroom is kept for the per-row accumulators m and l, and for
    the temporaries any real implementation spills. The paper's formula is
    asymptotic; the accountant in io_model.py is not.
    """
    usable = int(sram_bytes * 7 // 8) // itemsize
    if stage == "forward":
        bc = max(1, usable // (4 * d))
        return max(1, min(bc, d)), bc
    if stage == "fa2":
        b = max(1, int(math.floor(-2 * d + math.sqrt(4 * d * d + usable))))
        return b, b
    if stage == "backward":
        bc = max(1, usable // (10 * d))
        return max(1, min(bc, d)), bc
    raise ValueError(f"unknown stage {stage!r}")


def flash_forward(dev: Device, Qh: HBMTensor, Kh: HBMTensor, Vh: HBMTensor,
                  causal: bool = False, scale: Optional[float] = None,
                  block_q: Optional[int] = None, block_k: Optional[int] = None
                  ) -> Tuple[HBMTensor, HBMTensor]:
    """FlashAttention-1 forward. Returns (O, L) in HBM; L is the row logsumexp.

    Loop order is the paper's: OUTER over K/V blocks, INNER over Q blocks. That
    means K_j and V_j are loaded once each, and the running state for every
    query block — O_i, l_i, m_i — is read from HBM and written back on every
    outer iteration. Counting those round trips is the point of step 16, where
    FlashAttention-2 swaps the loops to get rid of them.

    HBM traffic is Theta(N^2 d^2 / M) against standard attention's Theta(N^2):
    the same asymptotic in N, smaller by the factor d^2/M. The other win is not
    asymptotic at all and matters more in practice — the N x N matrix is never
    allocated, so memory goes from quadratic to linear.

    L = m + log(l) is stored instead of P. It is one number per query, and it
    is everything the backward pass needs to reconstruct P exactly.
    """
    n, d = Qh.shape
    m_len = Kh.shape[0]
    dv = Vh.shape[1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    itemsize = np.dtype(Qh.dtype).itemsize
    br, bc = flash_block_sizes(dev.sram_bytes, max(d, dv), itemsize, "forward")
    br = max(1, min(block_q or br, n))
    bc = max(1, min(block_k or bc, m_len))

    Oh = dev.hbm_zeros((n, dv), dtype=Qh.dtype, name="O")
    Lh = dev.hbm_zeros((n,), dtype=Qh.dtype, name="L")
    mh = dev.hbm_zeros((n,), dtype=Qh.dtype, name="m")
    lh = dev.hbm_zeros((n,), dtype=Qh.dtype, name="l")
    with dev.scope() as sram:
        sram.store(mh, np.full(n, -np.inf, dtype=Qh.dtype))

    for j0 in range(0, m_len, bc):
        cj = slice(j0, min(j0 + bc, m_len))
        with dev.scope() as sram:
            k = sram.load(Kh, rows=cj)
            v = sram.load(Vh, rows=cj)

            for i0 in range(0, n, br):
                ri = slice(i0, min(i0 + br, n))

                if causal and cj.start > ri.stop - 1:
                    # Every key in this block is in the future of every query in
                    # that one: the whole tile is masked. Skipping it is what
                    # makes causal attention roughly half the work rather than
                    # the same work with half the results thrown away.
                    dev.count("blocks_skipped")
                    continue
                dev.count("blocks_computed")

                with dev.scope() as tile:
                    q = tile.load(Qh, rows=ri)
                    o = tile.load(Oh, rows=ri)      # <- re-read every outer step
                    l = tile.load(lh, rows=ri)
                    mm = tile.load(mh, rows=ri)

                    s = tile.keep(q @ k.T * scale)
                    if causal:
                        rows = np.arange(ri.start, ri.stop)[:, None]
                        cols = np.arange(cj.start, cj.stop)[None, :]
                        s = np.where(cols <= rows, s, -np.inf)

                    # FlashAttention-1 keeps O_i NORMALISED in HBM, so each
                    # update un-normalises, rescales, and re-normalises: three
                    # elementwise passes over the (br, dv) accumulator. Step 16
                    # removes two of them.
                    m_new, l_new, acc = online_softmax_update(
                        mm, l, o * l[:, None], s, v)
                    dev.count("o_elementwise_ops", 3 * (ri.stop - ri.start) * dv)
                    o_new = acc / np.where(l_new > 0, l_new, 1.0)[:, None]

                    tile.store(Oh, o_new, rows=ri)
                    tile.store(lh, l_new, rows=ri)
                    tile.store(mh, m_new, rows=ri)

    with dev.scope() as sram:
        mm = sram.load(mh)
        l = sram.load(lh)
        sram.store(Lh, np.where(l > 0, mm + np.log(np.where(l > 0, l, 1.0)), -np.inf))
    return Oh, Lh


# ---------------------------------------------------------------------------
# Step 14 — the backward pass, by recomputation
# ---------------------------------------------------------------------------

def flash_backward(dev: Device, Qh, Kh, Vh, Oh, Lh, dOh, causal: bool = False,
                   scale: Optional[float] = None, block_q: Optional[int] = None,
                   block_k: Optional[int] = None):
    """Gradients without ever storing P. Returns (dQ, dK, dV) in HBM.

    Standard backward reads the saved N x N probability matrix. Flash backward
    recomputes each tile from Q, K and the stored L:

        P_ij = exp(Q_i K_j^T / sqrt(d) - L_i)

    which is exact, because L_i is precisely the log normaliser of row i. The
    trade is arithmetic for memory — a second QK^T over the whole matrix — and
    it is a good trade on a memory-bound kernel: the recomputed tile never
    leaves SRAM, while the stored one would have cost two N^2 HBM round trips.

    The other half of the trick is D_i = sum_k dO_ik O_ik, which equals
    sum_k P_ik dP_ik (see baseline.attention_backward). One vector of length N,
    computed in a cheap first pass, replaces a second N x N intermediate.

    dK_j and dV_j accumulate entirely in SRAM across the inner loop; dQ_i does
    not — it is read and written per (i, j) pair, because the outer loop is over
    j. On a GPU that is an atomic add or a split-and-reduce, and it is the
    reason FlashAttention-2 reorganises this loop as well.
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

    # pass 0: D = rowsum(dO * O), O(N dv)
    for i0 in range(0, n, br):
        ri = slice(i0, min(i0 + br, n))
        with dev.scope() as sram:
            do = sram.load(dOh, rows=ri)
            o = sram.load(Oh, rows=ri)
            sram.store(Dh, np.sum(do * o, axis=-1), rows=ri)

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
                    dev.count("blocks_skipped")
                    continue
                dev.count("blocks_computed")

                with dev.scope() as tile:
                    q = tile.load(Qh, rows=ri)
                    do = tile.load(dOh, rows=ri)
                    lse = tile.load(Lh, rows=ri)
                    D = tile.load(Dh, rows=ri)
                    dq = tile.load(dQh, rows=ri)    # read-modify-write per (i, j)

                    s = tile.keep(q @ k.T * scale)
                    if causal:
                        rows = np.arange(ri.start, ri.stop)[:, None]
                        cols = np.arange(cj.start, cj.stop)[None, :]
                        s = np.where(cols <= rows, s, -np.inf)
                    p = np.exp(s - lse[:, None])     # recomputed, exact

                    dv_acc += p.T @ do
                    dp = tile.keep(do @ v.T)
                    ds = p * (dp - D[:, None])
                    dk += ds.T @ q * scale
                    tile.store(dQh, dq + ds @ k * scale, rows=ri)
                    dev.count("dq_stores")

            sram.store(dKh, dk, rows=cj)
            sram.store(dVh, dv_acc, rows=cj)

    return dQh, dKh, dVh


# ---------------------------------------------------------------------------
# Convenience wrappers (plain numpy in, plain numpy out)
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
