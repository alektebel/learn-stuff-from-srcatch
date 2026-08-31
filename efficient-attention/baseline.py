"""
Steps 1-2 — Standard scaled dot-product attention
==================================================
The thing all three papers are trying to beat. Everything later in this
directory is measured against what you write here, so get it exactly right.

    S = Q K^T / sqrt(d)          (N, M)   <- the quadratic object
    P = softmax(S, axis=-1)      (N, M)
    O = P V                      (N, dv)

Effort: small. Half an hour if you have written attention before, an hour if
not. The parts that are easy to get subtly wrong are the numerical stability of
the softmax and the row-coupling term in the backward pass — and both come back
in FlashAttention, where they are no longer optional details.

What you build:
  softmax               -> stable, and correct on fully-masked rows
  logsumexp             -> per row; FlashAttention stores exactly this
  attention_scores      -> S with masking applied
  attention             -> the forward pass
  attention_backward    -> dQ, dK, dV given the saved P
  rowsum_dO_O           -> the identity that makes flash-backward possible
  attention_cost        -> what one forward pass costs, analytically
"""

import numpy as np

from common import causal_mask, max_abs_error, numerical_gradient, random_qkv


# ---------------------------------------------------------------------------
# Step 1 — the forward pass
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Background:
      exp(90) already overflows float32, and attention logits reach that
      routinely — a large d, an untrained model, or one outlier key is enough.
      Subtracting the row maximum before exponentiating is EXACT (softmax is
      invariant to a constant shift per row), so it costs one pass and buys
      total safety.

      The case that catches people: a row where every entry is -inf, which
      happens with some masks. exp gives all zeros, the sum is zero, and the
      division produces nan that spreads through the whole batch. Decide what a
      fully masked row should be and handle it explicitly.

    TODO:
    1. row_max = max over `axis`, keepdims. Replace any non-finite row max with
       0.0 so that a fully-masked row does not produce (-inf) - (-inf) = nan.
    2. exps = exp(x - row_max); total = sum of exps over `axis`, keepdims.
    3. Return exps / total, but return 0 for rows whose total is 0.

    Test: softmax([[1000., 1000., 999.]]) must be finite and symmetric in its
    first two entries.
    """
    raise NotImplementedError


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log sum exp along `axis`, stably, with that axis removed.

    This is the log normaliser of a softmax row. FlashAttention stores one of
    these per query instead of storing P, and that single vector is everything
    its backward pass needs — so it is worth having as its own function.

    TODO: row_max + log(sum(exp(x - row_max))), same -inf care as above, then
    squeeze `axis` out.
    """
    raise NotImplementedError


def attention_scores(Q, K, mask=None, causal=False, scale=None):
    """S = Q K^T / sqrt(d), with disallowed entries set to -inf.

    Background:
      Use -inf, not -1e9. exp(-inf) is exactly 0, so a masked key contributes
      exactly nothing. With -1e9 the contribution is merely very small, and
      "very small" times thousands of masked positions is not nothing.

      Why the 1/sqrt(d)? Q_i . K_j is a sum of d products of unit-variance
      terms, so its standard deviation grows like sqrt(d). Without the scaling,
      softmax saturates as d grows and the gradients vanish.

    TODO:
    1. scale defaults to 1/sqrt(d) where d = Q.shape[-1].
    2. S = (Q @ K.T) * scale.
    3. If causal, AND the given mask (if any) with causal_mask(N, M) from
       common.py — True means ALLOWED.
    4. Where the mask is False, put -inf.
    """
    raise NotImplementedError


def attention(Q, K, V, mask=None, causal=False, scale=None, return_extras=False):
    """Standard attention: O(N*M) time and O(N*M) memory.

    The memory is the part that hurts. At N = 8192 the score matrix alone is
    268 MB in fp32 — per head, per layer — and during training it must stay
    alive until the backward pass consumes it.

    TODO:
    1. S = attention_scores(...); P = softmax(S); O = P @ V.
    2. If return_extras, also return {"S": S, "P": P, "lse": logsumexp(S)}.
       Later steps need P (for the backward pass) and lse (to check flash).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2 — the backward pass, and what the forward pass costs
# ---------------------------------------------------------------------------

def attention_backward(dO, Q, K, V, P, scale=None):
    """Gradients of standard attention, given the probabilities P from forward.

    Background — the only line that is not mechanical:
      Softmax rows are coupled. Raising one logit lowers every other
      probability in the same row, so the Jacobian is not diagonal:

          dS_ij = P_ij * (dP_ij - sum_k P_ik dP_ik)

      The subtracted term is one scalar per row, and it is exactly what makes
      the gradient of a probability *distribution* sum to zero along the row.
      Forget it and your gradients will look plausible, pass a smoke test, and
      train to a worse loss.

    TODO:
    1. dV = P.T @ dO
    2. dP = dO @ V.T
    3. D = rowsum(dP * P), keepdims
    4. dS = P * (dP - D)
    5. dQ = (dS @ K) * scale ; dK = (dS.T @ Q) * scale

    Test with common.numerical_gradient on a tiny input. Nothing else in this
    directory is worth writing until this matches to ~1e-8.
    """
    raise NotImplementedError


def rowsum_dO_O(dO, O):
    """The row-coupling term D, computed as rowsum(dO * O).

    Background:
      sum_k P_ik dP_ik  ==  sum_k dO_ik O_ik

      Substitute dP = dO V^T and O = P V and the two sides are the same sum.
      The left needs P and dP, both N x M. The right needs only dO and O, both
      N x dv. That is why FlashAttention's backward pass gets away with O(N)
      extra state instead of O(N^2) — check the identity here, use it in
      step 14.

    TODO: one line. Return shape (N, 1) so it broadcasts against an (N, M) tile.
    """
    raise NotImplementedError


def attention_cost(n, m=None, d=64, dv=None, itemsize=8):
    """Analytic cost of one standard-attention forward pass.

    Background:
      `score_elements` is separated out because it is the only quadratic term,
      the only one that must stay alive for the backward pass, and the one all
      three papers attack: clustered attention computes fewer ROWS of it,
      linear attention never FORMS it, FlashAttention forms it only in SRAM,
      one tile at a time.

    TODO: return a dict with
      score_elements  n * m
      score_bytes     score_elements * itemsize
      flops           2*n*m*d  (Q K^T)  +  5*n*m  (softmax)  +  2*n*m*dv  (P V)
      io_elements     n*d + m*d + m*dv + n*dv     (the inputs and the output)
      peak_bytes      (score_elements + io_elements) * itemsize
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------

def _demo():
    from common import human_bytes

    Q, K, V = random_qkv(6, 4, seed=0)
    O = attention(Q, K, V)
    print("output shape", O.shape)

    P = attention(Q, K, V, return_extras=True)[1]["P"]
    assert np.allclose(P.sum(axis=-1), 1.0)
    print("rows of P sum to 1 ✓")

    Oc = attention(Q, K, V, causal=True)
    O0 = attention(Q[:1], K[:1], V[:1])
    print("causal row 0 == attention over one key:", np.allclose(Oc[0], O0[0]))

    dO = np.random.default_rng(1).standard_normal(O.shape)
    _, extras = attention(Q, K, V, return_extras=True)
    dQ, dK, dV = attention_backward(dO, Q, K, V, extras["P"])
    num = numerical_gradient(lambda q: float(np.sum(attention(q, K, V) * dO)), Q)
    print("dQ max error vs numerical:", max_abs_error(dQ, num))

    print()
    for n in (1024, 4096, 16384):
        cost = attention_cost(n, d=64, itemsize=4)
        print(f"N={n:>6}  scores {human_bytes(cost['score_bytes']):>10}"
              f"  flops {cost['flops']/1e9:8.2f} G")


if __name__ == "__main__":
    _demo()
