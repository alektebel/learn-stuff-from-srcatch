"""
Step 1-2 — Standard scaled dot-product attention. Complete solution.

The thing all three papers are trying to beat:

    S = Q K^T / sqrt(d)          (N, M)   <- the quadratic object
    P = softmax(S, axis=-1)      (N, M)
    O = P V                      (N, dv)
"""

import numpy as np

from common import causal_mask, max_abs_error, numerical_gradient, random_qkv


# ---------------------------------------------------------------------------
# Step 1 — the forward pass
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the row max is not a nicety: exp(90) already overflows float32,
    and attention logits routinely reach that when d is large or the model has
    not been trained yet. The subtraction is exact — softmax is invariant to a
    constant shift per row — so it costs nothing but a pass over the data.

    A row that is entirely -inf (every position masked out) has no valid
    distribution; we return zeros for it rather than nan, so that a fully
    masked row contributes nothing instead of poisoning the whole output.
    """
    x = np.asarray(x)
    row_max = np.max(x, axis=axis, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exps = np.exp(x - row_max)
    total = np.sum(exps, axis=axis, keepdims=True)
    return np.where(total > 0, exps / np.where(total > 0, total, 1.0), 0.0)


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log sum exp, stably. FlashAttention stores exactly this per row, and it
    is the one extra tensor its backward pass needs."""
    x = np.asarray(x)
    row_max = np.max(x, axis=axis, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    out = row_max + np.log(np.sum(np.exp(x - row_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def attention_scores(Q, K, mask=None, causal=False, scale=None):
    """S = Q K^T / sqrt(d), with masked entries set to -inf.

    -inf, not a large negative number: exp(-inf) is exactly 0, so a masked key
    contributes exactly nothing. With -1e9 the contribution is merely tiny, and
    "tiny" accumulates over thousands of masked positions.
    """
    Q, K = np.asarray(Q), np.asarray(K)
    scale = 1.0 / np.sqrt(Q.shape[-1]) if scale is None else scale
    S = (Q @ K.T) * scale
    if causal:
        cm = causal_mask(Q.shape[0], K.shape[0])
        mask = cm if mask is None else (np.asarray(mask) & cm)
    if mask is not None:
        S = np.where(np.asarray(mask), S, -np.inf)
    return S


def attention(Q, K, V, mask=None, causal=False, scale=None, return_extras=False):
    """Standard attention. O(N*M) time and O(N*M) memory, and the memory is the
    part that hurts: at N = 8192 the score matrix alone is 268 MB in fp32, per
    head, per layer, and it must be kept for the backward pass.
    """
    S = attention_scores(Q, K, mask=mask, causal=causal, scale=scale)
    P = softmax(S, axis=-1)
    O = P @ np.asarray(V)
    if return_extras:
        return O, {"S": S, "P": P, "lse": logsumexp(S, axis=-1)}
    return O


# ---------------------------------------------------------------------------
# Step 2 — the backward pass, and what the forward pass costs
# ---------------------------------------------------------------------------

def attention_backward(dO, Q, K, V, P, scale=None):
    """Gradients of standard attention given the saved probabilities P.

    The only non-obvious line is dS. Softmax rows are coupled — raising one
    logit lowers every other probability in the row — so the Jacobian is not
    diagonal:

        dS_ij = P_ij * (dP_ij - sum_k P_ik dP_ik)

    The subtracted term is a per-row scalar, and it is exactly what makes the
    gradient of a probability *distribution* sum to zero along the row.

    That row-sum has a second identity worth noticing, because FlashAttention's
    backward pass depends on it:

        sum_k P_ik dP_ik = sum_k dO_ik O_ik

    i.e. it can be computed from dO and O alone, without ever forming dP. That
    is why flash-backward needs only O(N) extra state, not O(N^2).
    """
    Q, K, V, P, dO = map(np.asarray, (Q, K, V, P, dO))
    scale = 1.0 / np.sqrt(Q.shape[-1]) if scale is None else scale

    dV = P.T @ dO                          # (M, dv)
    dP = dO @ V.T                          # (N, M)
    D = np.sum(dP * P, axis=-1, keepdims=True)
    dS = P * (dP - D)                      # (N, M)
    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale
    return dQ, dK, dV


def rowsum_dO_O(dO, O):
    """The D vector of the backward pass, computed the cheap way: rowsum(dO*O).

    O(N*dv) instead of O(N*M), and it needs neither P nor dP.
    """
    return np.sum(np.asarray(dO) * np.asarray(O), axis=-1, keepdims=True)


def attention_cost(n, m=None, d=64, dv=None, itemsize=8):
    """Analytic cost of one standard-attention forward pass.

    Separating `score_elements` from everything else is the whole point of the
    table: it is the only term that is quadratic, it is the term that must be
    kept alive for the backward pass, and it is the term all three papers
    attack (clustered: compute fewer rows of it; linear: never form it;
    flash: form it only in SRAM, one tile at a time).
    """
    m = n if m is None else m
    dv = d if dv is None else dv
    score_elements = n * m
    flops = (2 * n * m * d          # Q K^T
             + 5 * n * m            # softmax: max, subtract, exp, sum, divide
             + 2 * n * m * dv)      # P V
    io_elements = n * d + m * d + m * dv + n * dv
    return {
        "flops": flops,
        "score_elements": score_elements,
        "score_bytes": score_elements * itemsize,
        "io_elements": io_elements,
        "peak_bytes": (score_elements + io_elements) * itemsize,
    }


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

    # gradient check
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
