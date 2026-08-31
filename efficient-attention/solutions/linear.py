"""
Steps 7-10 — Linear attention. Complete solution.

Katharopoulos, Vyas, Pappas, Fleuret, "Transformers are RNNs: Fast
Autoregressive Transformers with Linear Attention", ICML 2020.

One rewriting, three consequences:

    softmax(Q K^T) V  ~=  phi(Q) (phi(K)^T V)   /   phi(Q) (phi(K)^T 1)
                              ^^^^^^^^^^^^^^
                              (m, dv), never (N, N)

  * non-causal: O(N m dv) time, O(m dv) extra memory
  * causal: a prefix sum, still linear
  * causal at inference: a recurrent state, O(1) time and memory per token
"""

import numpy as np

from baseline import attention


# ---------------------------------------------------------------------------
# Step 7 — the feature map and the associativity trick
# ---------------------------------------------------------------------------

def elu_feature_map(x: np.ndarray) -> np.ndarray:
    """phi(x) = elu(x) + 1, applied elementwise. Strictly positive.

    Positivity is the whole specification, and it is not aesthetic:

      * the denominator phi(Q_i) . sum_j phi(K_j) is a sum of positive terms,
        so it cannot vanish or change sign. With a feature map that takes
        negative values, a near-zero denominator makes the output explode, and
        it happens rarely enough during training to look like a mystery.
      * the implied weights are non-negative and sum to one after
        normalisation, so the output stays inside the convex hull of V exactly
        as softmax attention does.

    elu+1 in particular is cheap, smooth, and has non-vanishing gradient for
    x < 0 (unlike relu+eps, which stops learning for negative pre-activations).
    It is *not* an approximation of exp: this method does not approximate
    softmax attention, it replaces it with a different attention that happens
    to be linear. Performer's positive random features are the choice you make
    when you do want to approximate softmax; the accuracy/complexity trade is
    different and so is the cost.
    """
    x = np.asarray(x)
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))


def elu_feature_map_grad(x: np.ndarray) -> np.ndarray:
    """d/dx [elu(x) + 1] = 1 for x > 0, exp(x) otherwise."""
    x = np.asarray(x)
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))


def quadratic_linear_attention(Q, K, V, feature_map=elu_feature_map, eps=1e-6,
                               causal=False):
    """The same function written the slow way: form the N x M weight matrix.

    Only ever used to check the associativity rearrangement — if this and
    `linear_attention` disagree, the rearrangement is wrong, and no amount of
    staring at the fast version will show you where.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    fq, fk = feature_map(Q), feature_map(K)
    A = fq @ fk.T                                   # (N, M) >= 0
    if causal:
        A = np.tril(A)
    return (A @ V) / (A.sum(axis=-1, keepdims=True) + eps)


def linear_attention(Q, K, V, feature_map=elu_feature_map, eps=1e-6):
    """Non-causal linear attention. O(N m dv), never forms an N x M matrix.

        KV = phi(K)^T V          (m, dv)     <- summarises the whole sequence
        Z  = phi(K)^T 1          (m,)
        O_i = phi(Q_i) KV / (phi(Q_i) . Z)

    Matrix products are associative; softmax is not, which is precisely why
    standard attention cannot do this. The N x M matrix exists only because
    softmax has to normalise across it. Remove the softmax and the sum over
    keys can be done *first*, once, for all queries.

    The cost, stated honestly: rank. phi(K)^T V has rank at most m, so the
    entire past is compressed into an m x dv summary no matter how long the
    sequence is. Softmax attention can, in principle, retrieve one specific
    token out of a million; this cannot, and that is the measurable quality
    gap on recall-heavy tasks.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    fq, fk = feature_map(Q), feature_map(K)
    KV = fk.T @ V                                   # (m, dv)
    Z = fk.sum(axis=0)                              # (m,)
    num = fq @ KV                                   # (N, dv)
    den = fq @ Z                                    # (N,)
    return num / (den[:, None] + eps)


# ---------------------------------------------------------------------------
# Step 8 — causal masking as a prefix sum
# ---------------------------------------------------------------------------

def causal_linear_attention(Q, K, V, feature_map=elu_feature_map, eps=1e-6,
                            method: str = "loop", return_extras: bool = False):
    """Causal linear attention.

        S_i = sum_{j<=i} phi(K_j) V_j^T      (m, dv)
        z_i = sum_{j<=i} phi(K_j)            (m,)
        O_i = phi(Q_i)^T S_i / (phi(Q_i) . z_i)

    Causal masking, which in standard attention means zeroing half of an N x N
    matrix, here means "use the running sum instead of the total". No mask is
    ever built.

    Two implementations of the same recurrence, and the difference matters:

      "loop"   sequential scan, O(m dv) memory. What a real kernel does.
      "cumsum" one vectorised np.cumsum over all N outer products, which
               materialises an (N, m, dv) tensor. Fast in numpy and a disaster
               in memory: at N=4096, m=dv=64 that is 134 MB per head, worse
               than the N^2 score matrix it replaced. The paper's CUDA kernel
               exists precisely to get the vectorised speed with the loop's
               memory.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    fq, fk = feature_map(Q), feature_map(K)
    n, m = fq.shape
    dv = V.shape[1]

    if method == "cumsum":
        outer = fk[:, :, None] * V[:, None, :]           # (N, m, dv)
        S = np.cumsum(outer, axis=0)                     # (N, m, dv)
        z = np.cumsum(fk, axis=0)                        # (N, m)
        num = np.einsum("nm,nmv->nv", fq, S)             # (N, dv)
        den = np.einsum("nm,nm->n", fq, z)               # (N,)
    elif method == "loop":
        num = np.zeros((n, dv), dtype=np.result_type(Q, V))
        den = np.zeros(n, dtype=np.result_type(Q, V))
        S = np.zeros((m, dv), dtype=np.result_type(Q, V))
        z = np.zeros(m, dtype=np.result_type(Q, V))
        for i in range(n):
            S += np.outer(fk[i], V[i])
            z += fk[i]
            num[i] = fq[i] @ S
            den[i] = fq[i] @ z
    else:
        raise ValueError(f"unknown method {method!r}")

    O = num / (den[:, None] + eps)
    if return_extras:
        return O, {"num": num, "den": den, "fq": fq, "fk": fk}
    return O


# ---------------------------------------------------------------------------
# Step 9 — the same thing as an RNN
# ---------------------------------------------------------------------------

class RecurrentLinearAttention:
    """Autoregressive decoding with O(1) time and memory per token.

        s <- s + phi(k) v^T          (m, dv)
        z <- z + phi(k)              (m,)
        o  = phi(q)^T s / (phi(q) . z)

    This is the title of the paper. A softmax transformer decoding token t must
    keep every previous key and value — the KV cache — and does O(t) work for
    token t, so generating L tokens costs O(L^2) and the cache grows without
    bound. A linear transformer keeps a *fixed-size* state instead: m x dv
    numbers, the same after one token as after a million, and O(1) work per
    step. Generation becomes O(L).

    What was traded away is visible in the state: s is a sum of outer products.
    Adding a new (k, v) pair never removes an old one, so the state saturates —
    old information is not forgotten, it is superimposed. Every later linear
    RNN (gated variants, decay terms, delta rules) is an answer to that one
    sentence.
    """

    def __init__(self, d_k: int, d_v: int, feature_map=elu_feature_map,
                 eps: float = 1e-6, dtype=np.float64):
        self.feature_map = feature_map
        self.eps = eps
        self.dtype = dtype
        self.d_k, self.d_v = d_k, d_v
        self.reset()

    def reset(self) -> None:
        self.s = np.zeros((self.d_k, self.d_v), dtype=self.dtype)
        self.z = np.zeros(self.d_k, dtype=self.dtype)
        self.steps = 0

    @property
    def state_elements(self) -> int:
        """Constant in sequence length. Compare with a KV cache's 2 * t * d."""
        return self.s.size + self.z.size

    def step(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        fq = self.feature_map(np.asarray(q, dtype=self.dtype))
        fk = self.feature_map(np.asarray(k, dtype=self.dtype))
        self.s += np.outer(fk, v)
        self.z += fk
        self.steps += 1
        return (fq @ self.s) / (fq @ self.z + self.eps)

    def run(self, Q, K, V) -> np.ndarray:
        """Decode a whole sequence one token at a time, from a clean state."""
        self.reset()
        return np.stack([self.step(q, k, v) for q, k, v in zip(Q, K, V)])


# ---------------------------------------------------------------------------
# Step 10 — the backward pass, in linear memory
# ---------------------------------------------------------------------------

def causal_linear_attention_backward(dO, Q, K, V, feature_map=elu_feature_map,
                                     feature_map_grad=elu_feature_map_grad,
                                     eps: float = 1e-6):
    """Gradients of causal linear attention without storing anything of size N^2
    — or of size (N, m, dv), which is the trap.

    The trick that makes the algebra symmetric: append a column of ones to V.
    Then the denominator is just one more output channel of the same numerator
    computation, and there is a single rule to differentiate rather than two.

        V' = [V | 1]                 (N, dv+1)
        S'_i = sum_{j<=i} phi(K_j) V'^T_j
        N'_i = phi(Q_i)^T S'_i       = [num_i , den_i]
        O_i  = num_i / den_i

    Differentiating the division gives the seed:

        dnum_i = dO_i / den_i        ddenom_i = -(dO_i . O_i) / den_i

    and then, writing G'_j = sum_{i>=j} phi(Q_i) dN'^T_i for the REVERSE
    cumulative sum:

        dphi(Q_i) = S'_i dN'_i            (forward scan)
        dphi(K_j) = G'_j V'_j             (reverse scan)
        dV_j      = (G'_j)^T phi(K_j)     (reverse scan, drop the ones column)

    Two scans, each carrying an (m, dv+1) state. Nothing quadratic, nothing of
    size N times the state. This is what "linear memory training" means, and it
    is why the method needs a custom kernel rather than falling out of autograd:
    autograd would happily store all N intermediate states.
    """
    Q, K, V, dO = map(np.asarray, (Q, K, V, dO))
    n, d = Q.shape
    dv = V.shape[1]
    dtype = np.result_type(Q, K, V, dO)

    fq, fk = feature_map(Q), feature_map(K)
    m = fq.shape[1]
    V_aug = np.concatenate([V, np.ones((n, 1), dtype=V.dtype)], axis=1)  # (N, dv+1)

    # ---- forward scan: recompute num/den, and dphi(Q) on the way through ----
    dfq = np.zeros((n, m), dtype=dtype)
    dN = np.zeros((n, dv + 1), dtype=dtype)
    S = np.zeros((m, dv + 1), dtype=dtype)
    for i in range(n):
        S += np.outer(fk[i], V_aug[i])
        Ni = fq[i] @ S                                   # [num_i, den_i]
        num_i, den_i = Ni[:dv], Ni[dv] + eps
        o_i = num_i / den_i
        dN[i, :dv] = dO[i] / den_i
        dN[i, dv] = -float(dO[i] @ o_i) / den_i
        dfq[i] = S @ dN[i]

    # ---- reverse scan: dphi(K) and dV --------------------------------------
    dfk = np.zeros((n, m), dtype=dtype)
    dV = np.zeros((n, dv), dtype=dtype)
    G = np.zeros((m, dv + 1), dtype=dtype)
    for j in range(n - 1, -1, -1):
        G += np.outer(fq[j], dN[j])
        dfk[j] = G @ V_aug[j]
        dV[j] = (G.T @ fk[j])[:dv]

    dQ = dfq * feature_map_grad(Q)
    dK = dfk * feature_map_grad(K)
    return dQ, dK, dV


# ---------------------------------------------------------------------------

def _demo():
    from common import human_bytes, max_abs_error, numerical_gradient, random_qkv, timeit

    Q, K, V = random_qkv(64, 16, seed=0)

    fast, slow = linear_attention(Q, K, V), quadratic_linear_attention(Q, K, V)
    print("associativity holds  (fast vs quadratic form):",
          f"{max_abs_error(fast, slow):.2e}")
    print("causal loop vs cumsum:",
          f"{max_abs_error(causal_linear_attention(Q, K, V, method='loop'), causal_linear_attention(Q, K, V, method='cumsum')):.2e}")
    print("causal vs masked quadratic:",
          f"{max_abs_error(causal_linear_attention(Q, K, V), quadratic_linear_attention(Q, K, V, causal=True)):.2e}")

    rnn = RecurrentLinearAttention(16, 16)
    print("RNN decoding vs parallel causal form:",
          f"{max_abs_error(rnn.run(Q, K, V), causal_linear_attention(Q, K, V)):.2e}")
    print(f"RNN state: {rnn.state_elements} numbers after {rnn.steps} tokens "
          f"(a KV cache would hold {2 * rnn.steps * 16})")

    dO = np.random.default_rng(2).standard_normal(V.shape)
    dQ, dK, dV = causal_linear_attention_backward(dO, Q, K, V)
    for name, analytic, wrt in (("dQ", dQ, Q), ("dK", dK, K), ("dV", dV, V)):
        num = numerical_gradient(
            lambda x, wrt=wrt: float(np.sum(causal_linear_attention(
                x if wrt is Q else Q, x if wrt is K else K, x if wrt is V else V) * dO)),
            wrt.copy())
        print(f"{name} max error vs numerical: {max_abs_error(analytic, num):.2e}")

    print()
    print(f"{'N':>6} {'softmax':>10} {'linear':>10} {'speedup':>8} "
          f"{'softmax scores':>15}")
    for n in (128, 512, 2048, 8192):
        q, k, v = random_qkv(n, 64, seed=1)
        t_soft = timeit(lambda: attention(q, k, v))
        t_lin = timeit(lambda: linear_attention(q, k, v))
        print(f"{n:>6} {t_soft * 1e3:>9.2f}ms {t_lin * 1e3:>9.2f}ms "
              f"{t_soft / t_lin:>7.1f}x {human_bytes(n * n * 8):>15}")


if __name__ == "__main__":
    _demo()
