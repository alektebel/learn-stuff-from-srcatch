"""
Steps 3-6 — Clustered attention
================================
Paper: Vyas, Katharopoulos, Fleuret, "Fast Transformers with Clustered
Attention", NeurIPS 2020 (arXiv:2007.04825).

The idea in one sentence: queries that are close together produce nearly the
same attention distribution, so group the queries, compute attention once per
group, and give every member of the group the group's answer.

Effort: this is the longest file, and step 6 is the one that repays the most
thought. Budget an afternoon.

What you build:
  lsh_bits                       -> random-hyperplane hashing
  hamming_distances              -> (N, C) distances between bit vectors
  hamming_kmeans                 -> Lloyd's algorithm in Hamming space
  cluster_queries                -> assignments + centroids back in R^d
  clustered_attention            -> C rows of attention instead of N
  attention_error_bound          -> the guarantee, computed without the answer
  improved_clustered_attention   -> exact on each cluster's top-k keys
  improved_error_bound           -> the much better guarantee that buys
"""

from typing import Optional, Tuple

import numpy as np

from baseline import attention, softmax


# ---------------------------------------------------------------------------
# Step 3 — clustering the queries
# ---------------------------------------------------------------------------

def lsh_bits(Q: np.ndarray, n_bits: int = 16, seed: int = 0) -> np.ndarray:
    """Random-hyperplane LSH: bit b of query i is [ Q_i . r_b > 0 ].

    Background — why hash at all, when K-means works in R^d directly?
      Cost. A Euclidean K-means iteration is O(N C d) multiply-adds. In Hamming
      space each distance is a popcount over B bits: a couple of integer
      instructions, no floating point. The clustering is a means to an end and
      has to be much cheaper than the attention it replaces, or the exercise is
      pointless.

      What the hash preserves: for unit-norm vectors, P[bit differs] = theta/pi
      where theta is the angle between them, so Hamming distance is a monotone
      estimator of angular distance.

      What it loses: magnitude. Q and 3Q hash identically but produce very
      differently peaked attention distributions. That is a real limitation of
      the method — the checker asserts the behaviour so that you meet it
      deliberately rather than discovering it as a mystery later.

    TODO:
    1. rng = np.random.default_rng(seed); draw (d, n_bits) gaussian planes.
       Use the seed so the hash is reproducible — clustering that changes
       between calls makes every later measurement noise.
    2. Return (Q @ planes > 0) as uint8, shape (N, n_bits).
    """
    raise NotImplementedError


def hamming_distances(bits: np.ndarray, centroid_bits: np.ndarray) -> np.ndarray:
    """(N, B) and (C, B) -> (N, C) Hamming distances.

    TODO: for 0/1 values, |a - b| is XOR. Broadcast to (N, C, B) and sum over
    the last axis. Cast to a signed type first — uint8 subtraction wraps
    around, and 0 - 1 = 255 will give you a silently wrong clustering.
    """
    raise NotImplementedError


def hamming_kmeans(bits: np.ndarray, n_clusters: int, iters: int = 10,
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm in Hamming space. Returns (assignments, centroid_bits).

    Background:
      The only change from Euclidean K-means is the centroid update. The point
      minimising total Hamming distance to a set of bit vectors is the
      coordinate-wise MAJORITY bit, not the mean.

      Empty clusters must be handled. Left alone they collapse the method
      quietly: you ask for 32 clusters, get 9 non-empty ones, and conclude that
      clustered attention is less accurate than it really is.

    TODO:
    1. Initialise centroid_bits from n_clusters distinct random rows of `bits`.
    2. Repeat `iters` times:
       a. distances = hamming_distances(bits, centroid_bits);
          assignments = argmin over clusters.
       b. For each cluster: if it has members, set its centroid to the
          majority bit (members.mean(axis=0) >= 0.5). If it is EMPTY, re-seed
          it to the point currently furthest from its own centroid, and
          reassign that point.
    3. Return (assignments, centroid_bits).

    Test: 3 well-separated bit prototypes with 5% noise must come back as 3
    clusters, each containing exactly one prototype's points.
    """
    raise NotImplementedError


def cluster_queries(Q: np.ndarray, n_clusters: int, n_bits: int = 16,
                    iters: int = 10, seed: int = 0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Group queries; return (assignments, centroids) with centroids in R^d.

    Background — two different spaces, and the distinction matters:
      Clustering happens in Hamming space because it is cheap. The centroid
      that goes into the attention computation has to live in the queries' own
      space, so it is the plain MEAN of the cluster's members in R^d.

    TODO:
    1. bits = lsh_bits(Q, n_bits, seed); assignments = hamming_kmeans(...)[0].
    2. Distinct queries can hash to identical bit strings, so a cluster can
       still come back empty. Drop empty clusters and renumber the assignments
       (np.unique(assignments, return_inverse=True) does both at once) — the
       mean of an empty cluster is nan, and nan spreads through every
       subsequent matmul without a single error message.
    3. centroids[c] = mean of the queries assigned to c. Return both.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 4 — clustered attention
# ---------------------------------------------------------------------------

def clustered_attention(Q, K, V, n_clusters: int = 8, n_bits: int = 16,
                        iters: int = 10, seed: int = 0, scale: Optional[float] = None,
                        return_extras: bool = False):
    """Attention computed once per cluster of queries, then broadcast.

        A^c = softmax(Q^c K^T / sqrt(d))     (C, M)
        V^c = A^c V                          (C, dv)
        O_i = V^c_{cluster(i)}               (N, dv)

    Cost: O(C M d) for the attention plus O(N C B) for the clustering, against
    O(N M d) for the real thing. With C fixed, that is LINEAR in sequence
    length.

    Background — the limitation to have in mind while you write it:
      Causal masking does not work here. Members of a cluster sit at different
      positions, so no single mask is correct for the group, and recomputing
      the centroid distribution per member is the cost you were avoiding.
      Clustered attention is an encoder method. That is the honest reason the
      paper evaluates on speech recognition rather than language modelling, and
      the reason linear attention (next file) is the one used for
      autoregressive decoding.

    TODO:
    1. assignments, centroids = cluster_queries(...).
    2. Ac = softmax(centroids @ K.T * scale) -- (C, M), each row a distribution.
    3. Vc = Ac @ V -- (C, dv).
    4. O = Vc[assignments] -- (N, dv). One gather, no per-query work at all.
    5. If return_extras, also return assignments / centroids / Ac / Vc /
       n_clusters; steps 5 and 6 need them.

    Test: with n_clusters = N (one cluster per query) the result must be EXACT,
    because each centroid is then the query itself.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 5 — what the approximation guarantees
# ---------------------------------------------------------------------------

def attention_error_bound(Q, K, V, assignments, centroids, scale=None,
                          tight: bool = False) -> np.ndarray:
    """Per-query upper bound on || V_i - V_hat_i ||_2, without computing V_i.

    Background — the derivation, three elementary inequalities:

      1. Let a = softmax(u) be the true attention row and b = softmax(v) the
         centroid's, with delta = ||u - v||_inf. For every n,

             |log a_n - log b_n| <= |u_n - v_n| + |log Z_u - log Z_v| <= 2 delta

         so a_n <= b_n e^{2 delta}, and summing gives ||a - b||_1 <= e^{2 delta} - 1.

      2. || sum_n (a_n - b_n) V_n ||_2 <= ||a - b||_1 * max_n ||V_n||_2.

      3. delta = max_n |(Q_i - c) . K_n| / sqrt(d), which Cauchy-Schwarz bounds
         by ||Q_i - c||_2 * max_n ||K_n||_2 / sqrt(d).

      Together:

          ||V_i - V_hat_i||_2 <= ( exp(2 ||Q_i - c||_2 max||K||_2 / sqrt(d)) - 1 )
                                 * max_n ||V_n||_2

      Read it for its SHAPE, not its size. The exponential makes it numerically
      vacuous unless clusters are tight — often worse than the trivial bound
      2*max||V||, which holds for any two distributions. What it says is the
      useful part, and it is the paper's argument: the error is controlled by
      the distance from each query to its centroid and vanishes with it. That
      is what licenses spending compute on better clustering, and why the
      method degrades gracefully rather than unpredictably.

    TODO:
    1. residual = Q - centroids[assignments].
    2. delta: if tight, max over keys of |residual @ K.T| * scale (exact, but
       costs a full N x M pass — a diagnostic, not something a real forward
       pass would compute); otherwise ||residual|| * max||K|| * scale.
    3. l1 = expm1(2 * delta) — use np.expm1 and suppress the overflow warning.
    4. Return min(l1, 2.0) * max_n||V_n||: the trivial bound is always valid
       too, and taking the smaller of the two is free.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 6 — improved clustered attention
# ---------------------------------------------------------------------------

def improved_clustered_attention(Q, K, V, n_clusters: int = 8, top_k: int = 8,
                                 n_bits: int = 16, iters: int = 10, seed: int = 0,
                                 scale: Optional[float] = None,
                                 return_extras: bool = False):
    """Exact attention on each cluster's top-k keys, centroid approximation elsewhere.

    Background:
      Attention distributions are peaked: a handful of keys carry most of the
      mass. So use the centroid pass only to FIND those keys, then pay for
      exact dot products against them for every query in the cluster.

      For query i in cluster j, with T = the cluster's top-k keys by A^c:

          w       = softmax over T of (Q_i . K_n / sqrt(d))     exact, k terms
          m_j     = sum_{n in T} A^c_{jn}                       centroid's mass on T
          V_hat_i = m_j * sum_{n in T} w_n V_n
                    + ( V^c_j - sum_{n in T} A^c_{jn} V_n )

      The two pieces are "the mass the centroid says belongs to the top-k,
      spent according to this query's own preferences" and "everything else,
      spent as the centroid would have". The second is a subtraction from V^c_j,
      so it costs nothing beyond the pass already done.

      Why the softmax over a SUBSET is legitimate: restricting a softmax to a
      subset and renormalising gives exactly the conditional distribution of
      the full softmax on that subset. So the first term is the true
      distribution restricted to T, rescaled to carry mass m_j.

    Cost: O(C M d) + O(N k d).

    TODO:
    1. Cluster, then Ac and Vc as in step 4.
    2. For each cluster c with members:
       a. topk = indices of the k largest entries of Ac[c] (np.argpartition).
       b. mass = Ac[c, topk].sum().
       c. w = softmax(Q[members] @ K[topk].T * scale) -- (n_c, k).
       d. O[members] = mass * (w @ V[topk]) + (Vc[c] - Ac[c, topk] @ V[topk]).
    3. Record kept_mass[c] = mass; the bound below needs it.

    Two properties worth checking rather than believing, and the checker does:
      * top_k = M is EXACT for every query, whatever the clustering.
      * error falls monotonically as top_k grows.
    """
    raise NotImplementedError


def improved_error_bound(Q, K, V, extras, top_k: int, scale=None) -> np.ndarray:
    """Per-query bound 2[(1 - alpha_i) + (1 - m_j)] * max||V||_2.

    Background:
      alpha_i is the query's OWN true mass on the cluster's top-k keys, m_j the
      centroid's. Write the error out and the only surviving terms are the mass
      each distribution puts outside T — missing mass is the only way to be
      wrong. That is a far more informative guarantee than step 5's
      exponential, and it degrades to 0 as k grows, for every query, regardless
      of how bad the clustering is.

      alpha_i needs the true attention row, so this is a DIAGNOSTIC, not
      something a real forward pass computes. It is here because watching the
      bound track the measured error is what makes a guarantee mean anything.

    TODO:
    1. A = softmax(Q @ K.T * scale) -- the true attention, for diagnosis only.
    2. For each cluster, recompute its top-k (same rule as above) and set
       alpha[members] = A[members][:, topk].sum(axis=-1).
    3. Return min(2*((1-alpha) + (1-kept_mass[assignments])), 2) * max||V||.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------

def _demo():
    from common import clustered_qkv, random_qkv, rel_error

    n, d = 256, 32
    Q, K, V = clustered_qkv(n, d, groups=8, seed=0, spread=0.1, scale=2.0)
    exact = attention(Q, K, V)

    print("relative error vs exact attention   (256 queries in 8 real groups)\n")
    print(f"{'clusters':>9} {'cost/exact':>11} {'clustered':>11} {'+top-8':>9}"
          f" {'+top-32':>9}")
    for c in (2, 4, 8, 16, 32, 64):
        approx = clustered_attention(Q, K, V, n_clusters=c)
        imp8 = improved_clustered_attention(Q, K, V, n_clusters=c, top_k=8)
        imp32 = improved_clustered_attention(Q, K, V, n_clusters=c, top_k=32)
        print(f"{c:>9} {c / n:>11.3f} {rel_error(approx, exact):>11.4f}"
              f" {rel_error(imp8, exact):>9.4f} {rel_error(imp32, exact):>9.4f}")

    print("\nthe bound of step 5 only says something once clusters are tight")
    print(f"  (trivial bound, always true: {2 * np.max(np.linalg.norm(V, axis=-1)):.2f})\n")
    print(f"{'spread':>8} {'mean error':>11} {'mean bound':>11} {'bound/trivial':>14}")
    for spread in (0.3, 0.1, 0.03, 0.01):
        Qs, Ks, Vs = clustered_qkv(n, d, groups=8, seed=0, spread=spread, scale=2.0)
        approx, extras = clustered_attention(Qs, Ks, Vs, n_clusters=8,
                                             return_extras=True)
        err = np.linalg.norm(approx - attention(Qs, Ks, Vs), axis=-1)
        bound = attention_error_bound(Qs, Ks, Vs, extras["assignments"],
                                      extras["centroids"], tight=True)
        trivial = 2 * np.max(np.linalg.norm(Vs, axis=-1))
        assert np.all(err <= bound + 1e-9), "the bound must never be violated"
        print(f"{spread:>8} {err.mean():>11.4f} {bound.mean():>11.4f}"
              f" {bound.mean() / trivial:>14.2f}")

    print("\ntop_k = M is exact regardless of clustering:",
          np.allclose(improved_clustered_attention(Q, K, V, n_clusters=4, top_k=n),
                      exact))

    Qr, Kr, Vr = random_qkv(n, d, seed=0, scale=2.0)
    print("relative error on UNclustered (iid gaussian) queries, 8 clusters:",
          f"{rel_error(clustered_attention(Qr, Kr, Vr, n_clusters=8), attention(Qr, Kr, Vr)):.4f}")


if __name__ == "__main__":
    _demo()
