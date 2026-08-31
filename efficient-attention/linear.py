"""
Steps 7-10 — Linear attention
==============================
Paper: Katharopoulos, Vyas, Pappas, Fleuret, "Transformers are RNNs: Fast
Autoregressive Transformers with Linear Attention", ICML 2020
(arXiv:2006.16236).

One rewriting, three consequences:

    softmax(Q K^T) V  ~=  phi(Q) (phi(K)^T V)   /   phi(Q) (phi(K)^T 1)
                              ^^^^^^^^^^^^^^
                              (m, dv), never (N, N)

  * non-causal: O(N m dv) time, O(m dv) extra memory
  * causal: a prefix sum, still linear
  * causal at inference: a recurrent state, O(1) time and memory per token

Effort: steps 7-9 are short and mostly pleasant. Step 10 is the hardest piece
of algebra in the directory; do it with pen and paper before you write code.

What you build:
  elu_feature_map / _grad          -> phi(x) = elu(x) + 1
  quadratic_linear_attention       -> the slow form, to check against
  linear_attention                 -> the associativity rearrangement
  causal_linear_attention          -> prefix sums, two implementations
  RecurrentLinearAttention         -> the RNN the title is about
  causal_linear_attention_backward -> gradients in linear memory
"""

import numpy as np

from baseline import attention


# ---------------------------------------------------------------------------
# Step 7 — the feature map and the associativity trick
# ---------------------------------------------------------------------------

def elu_feature_map(x: np.ndarray) -> np.ndarray:
    """phi(x) = elu(x) + 1, elementwise. Strictly positive.

    Background — positivity is the whole specification:
      * The denominator phi(Q_i) . sum_j phi(K_j) is a sum of positive terms,
        so it cannot vanish or change sign. With a feature map that takes
        negative values, a near-zero denominator makes the output explode — and
        it happens rarely enough during training to look like a mystery rather
        than a bug.
      * The implied weights are non-negative and sum to one after
        normalisation, so the output stays inside the convex hull of V, exactly
        as softmax attention does.

      elu+1 is cheap, smooth, and keeps a non-vanishing gradient for x < 0
      (unlike relu, which stops learning there). It is NOT an approximation of
      exp: this method does not approximate softmax attention, it replaces it
      with a different attention that happens to be linear. If you do want to
      approximate softmax, that is Performer's positive random features, and it
      is a different accuracy/cost trade.

    TODO: np.where(x > 0, x + 1, exp(x)). Clip the argument of exp at 0 so the
    untaken branch cannot overflow — numpy evaluates both.
    """
    raise NotImplementedError


def elu_feature_map_grad(x: np.ndarray) -> np.ndarray:
    """d/dx [elu(x) + 1] = 1 for x > 0, exp(x) otherwise. Step 10 needs it."""
    raise NotImplementedError


def quadratic_linear_attention(Q, K, V, feature_map=elu_feature_map, eps=1e-6,
                               causal=False):
    """The same function written the slow way: form the N x M weight matrix.

    Its only purpose is to check the rearrangement. If this and
    `linear_attention` disagree, the rearrangement is wrong, and no amount of
    staring at the fast version will show you where.

    TODO: A = phi(Q) @ phi(K).T; if causal, np.tril(A); return
    (A @ V) / (A.sum(-1, keepdims) + eps).
    """
    raise NotImplementedError


def linear_attention(Q, K, V, feature_map=elu_feature_map, eps=1e-6):
    """Non-causal linear attention. O(N m dv), never forms an N x M matrix.

        KV = phi(K)^T V          (m, dv)     <- summarises the whole sequence
        Z  = phi(K)^T 1          (m,)
        O_i = phi(Q_i) KV / (phi(Q_i) . Z)

    Background:
      Matrix products are associative; softmax is not. That is the whole
      argument. The N x M matrix exists in standard attention only because
      softmax has to normalise across it. Remove the softmax and the sum over
      keys can be done FIRST, once, for every query at the same time.

      The cost, stated honestly: rank. phi(K)^T V has rank at most m, so the
      entire past is compressed into an m x dv summary however long the
      sequence is. Softmax attention can in principle retrieve one specific
      token out of a million; this cannot. That is the measurable quality gap
      on recall-heavy tasks, and no amount of engineering removes it.

    TODO: four lines, exactly as written above. Add eps to the denominator.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 8 — causal masking as a prefix sum
# ---------------------------------------------------------------------------

def causal_linear_attention(Q, K, V, feature_map=elu_feature_map, eps=1e-6,
                            method: str = "loop", return_extras: bool = False):
    """Causal linear attention.

        S_i = sum_{j<=i} phi(K_j) V_j^T      (m, dv)
        z_i = sum_{j<=i} phi(K_j)            (m,)
        O_i = phi(Q_i)^T S_i / (phi(Q_i) . z_i)

    Background:
      Causal masking, which in standard attention means zeroing half of an
      N x N matrix, here means "use the running sum instead of the total". No
      mask is ever built. Nothing is thrown away, so nothing is computed and
      discarded.

    TODO — implement both, because the difference is the lesson:

      method="loop"    Sequential scan carrying S (m, dv) and z (m,). Memory
                       O(m dv), independent of N. This is what a real kernel
                       does.

      method="cumsum"  np.cumsum over all N outer products phi(K_i) V_i^T at
                       once: einsum("nm,nmv->nv", phi(Q), S) after the scan.
                       Fast in numpy and a disaster in memory — it materialises
                       an (N, m, dv) tensor, which at N=4096, m=dv=64 is 134 MB
                       per head, worse than the N^2 matrix it replaced. The
                       paper's CUDA kernel exists precisely to get this
                       version's speed with the loop's memory.

    Return O = num / (den + eps), and the extras dict when asked (num, den, fq,
    fk) — nothing later needs it, but it is what you will want when a gradient
    disagrees in step 10.

    Test: must equal quadratic_linear_attention(..., causal=True), and changing
    K or V after position t must not change any output before t.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 9 — the same thing as an RNN
# ---------------------------------------------------------------------------

class RecurrentLinearAttention:
    """Autoregressive decoding with O(1) time and memory per token.

        s <- s + phi(k) v^T          (m, dv)
        z <- z + phi(k)              (m,)
        o  = phi(q)^T s / (phi(q) . z)

    Background — this is the title of the paper:
      A softmax transformer decoding token t must keep every previous key and
      value (the KV cache) and does O(t) work for token t, so generating L
      tokens costs O(L^2) time and O(L) memory that grows without bound. A
      linear transformer keeps a FIXED-SIZE state: m x dv numbers, the same
      after one token as after a million, and O(1) work per step. Generation
      becomes O(L).

      What was traded away is visible in the state. s is a sum of outer
      products; adding a new (k, v) pair never removes an old one, so the state
      saturates — old information is not forgotten, it is superimposed. Every
      later linear RNN (gated variants, decay terms, delta rules, and the
      state-space models) is an answer to that one sentence.

    TODO:
      reset()          zero s (d_k, d_v) and z (d_k,), steps = 0
      state_elements   s.size + z.size — must NOT change as tokens arrive
      step(q, k, v)    the three lines above; return the output vector
      run(Q, K, V)     reset, then step over the sequence, stack the outputs

    Test: run() must equal causal_linear_attention() to floating-point noise.
    """

    def __init__(self, d_k: int, d_v: int, feature_map=elu_feature_map,
                 eps: float = 1e-6, dtype=np.float64):
        self.feature_map = feature_map
        self.eps = eps
        self.dtype = dtype
        self.d_k, self.d_v = d_k, d_v
        self.reset()

    def reset(self) -> None:
        raise NotImplementedError

    @property
    def state_elements(self) -> int:
        """Constant in sequence length. Compare with a KV cache's 2 * t * d."""
        raise NotImplementedError

    def step(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self, Q, K, V) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 10 — the backward pass, in linear memory
# ---------------------------------------------------------------------------

def causal_linear_attention_backward(dO, Q, K, V, feature_map=elu_feature_map,
                                     feature_map_grad=elu_feature_map_grad,
                                     eps: float = 1e-6):
    """Gradients of causal linear attention, storing nothing of size N^2 —
    or of size (N, m, dv), which is the trap autograd falls into.

    Background — the trick that makes the algebra symmetric:
      Append a column of ones to V. Then the denominator is just one more
      output channel of the same numerator computation, and there is a single
      rule to differentiate instead of two.

          V' = [V | 1]                  (N, dv+1)
          S'_i = sum_{j<=i} phi(K_j) V'^T_j
          N'_i = phi(Q_i)^T S'_i        = [num_i , den_i]
          O_i  = num_i / den_i

      Differentiating the division gives the seed:

          dnum_i = dO_i / den_i         dden_i = -(dO_i . O_i) / den_i

      and then, writing G'_j = sum_{i>=j} phi(Q_i) dN'^T_i for the REVERSE
      cumulative sum:

          dphi(Q_i) = S'_i dN'_i             (forward scan)
          dphi(K_j) = G'_j V'_j              (reverse scan)
          dV_j      = (G'_j)^T phi(K_j)      (reverse scan, drop the ones column)

      Derive those three yourself before coding: each is one line of index
      manipulation from N'_i = sum_{j<=i} (phi(Q_i) . phi(K_j)) V'_j, and the
      only thing to be careful about is which index the sum runs over. Query i
      sees keys up to i; key j is seen by queries from j onwards. That
      asymmetry is why one scan runs forwards and the other backwards.

    TODO:
    1. fq = phi(Q), fk = phi(K), V_aug = [V | ones].
    2. Forward scan over i: maintain S (m, dv+1); recompute num_i, den_i;
       fill dN[i] = [dO_i/den_i, -(dO_i . o_i)/den_i]; set dfq[i] = S @ dN[i].
    3. Reverse scan over j: maintain G (m, dv+1); set dfk[j] = G @ V_aug[j] and
       dV[j] = (G.T @ fk[j])[:dv].
    4. Chain through the feature map: dQ = dfq * phi'(Q), dK = dfk * phi'(K).

    Two scans, each carrying an (m, dv+1) state. Nothing quadratic, nothing of
    size N times the state — that is what "linear memory training" means, and
    it is why the method needs a custom kernel rather than falling out of
    autograd, which would happily store all N intermediate states.

    Test: common.numerical_gradient on a 10-token sequence.
    """
    raise NotImplementedError


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
