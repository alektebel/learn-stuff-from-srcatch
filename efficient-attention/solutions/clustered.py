"""
Steps 3-6 — Clustered attention. Complete solution.

Vyas, Katharopoulos, Fleuret, "Fast Transformers with Clustered Attention",
NeurIPS 2020.

The idea in one sentence: queries that are close together produce nearly the
same attention distribution, so group the queries, compute attention once per
group, and hand every member of the group the group's answer.
"""

from typing import Optional, Tuple

import numpy as np

from baseline import attention, softmax


# ---------------------------------------------------------------------------
# Step 3 — clustering the queries
# ---------------------------------------------------------------------------

def lsh_bits(Q: np.ndarray, n_bits: int = 16, seed: int = 0) -> np.ndarray:
    """Random-hyperplane LSH: bit b of query i is [ Q_i . r_b > 0 ].

    Why hash at all, when K-means works in R^d directly? Cost. A Euclidean
    K-means iteration is O(N C d) multiply-adds; in Hamming space each distance
    is a popcount over B bits, which on real hardware is a handful of integer
    instructions and no floating point at all. The clustering is a means to an
    end and has to be much cheaper than the attention it replaces, or the whole
    exercise is pointless.

    What is preserved: for unit-norm vectors, P[bit differs] = theta / pi where
    theta is the angle between them. Hamming distance is therefore a monotone
    estimator of angular distance. What is lost: magnitude. Two queries that
    point the same way but differ in length hash identically, yet produce very
    differently peaked attention distributions. That is a real failure mode of
    the method, not a detail of this implementation.
    """
    rng = np.random.default_rng(seed)
    planes = rng.standard_normal((np.asarray(Q).shape[1], n_bits))
    return (np.asarray(Q) @ planes > 0).astype(np.uint8)


def hamming_distances(bits: np.ndarray, centroid_bits: np.ndarray) -> np.ndarray:
    """(N, B) x (C, B) -> (N, C) Hamming distances."""
    bits = np.asarray(bits).astype(np.int16)
    centroid_bits = np.asarray(centroid_bits).astype(np.int16)
    # XOR is |a - b| for 0/1 values; summing over bits gives the distance.
    return np.abs(bits[:, None, :] - centroid_bits[None, :, :]).sum(axis=-1)


def hamming_kmeans(bits: np.ndarray, n_clusters: int, iters: int = 10,
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm in Hamming space. Returns (assignments, centroid_bits).

    The only change from Euclidean K-means is the centroid update: the point
    minimising total Hamming distance to a set of bit vectors is the
    coordinate-wise MAJORITY bit, not the mean. Taking a mean and rounding is
    the same thing here, but "majority" is what it is.

    Empty clusters are re-seeded to the point currently worst served by its own
    centroid. Left alone they collapse the method silently: you ask for 32
    clusters, get 9 non-empty ones, and conclude clustered attention is less
    accurate than it is.
    """
    bits = np.asarray(bits).astype(np.uint8)
    n, n_bits = bits.shape
    n_clusters = min(n_clusters, n)
    rng = np.random.default_rng(seed)

    centroid_bits = bits[rng.choice(n, size=n_clusters, replace=False)].copy()
    assignments = np.zeros(n, dtype=np.int64)

    for _ in range(iters):
        distances = hamming_distances(bits, centroid_bits)
        assignments = np.argmin(distances, axis=1)

        for c in range(n_clusters):
            members = bits[assignments == c]
            if len(members) == 0:
                worst = int(np.argmax(distances[np.arange(n), assignments]))
                centroid_bits[c] = bits[worst]
                assignments[worst] = c
                continue
            centroid_bits[c] = (members.mean(axis=0) >= 0.5).astype(np.uint8)

    return assignments, centroid_bits


def cluster_queries(Q: np.ndarray, n_clusters: int, n_bits: int = 16,
                    iters: int = 10, seed: int = 0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Group queries and return (assignments, centroids) with centroids in R^d.

    Note the two spaces. Clustering happens in Hamming space because it is
    cheap; the centroid that goes into the attention computation must live in
    the queries' own space, so it is the plain mean of the cluster's members.
    """
    Q = np.asarray(Q)
    bits = lsh_bits(Q, n_bits=n_bits, seed=seed)
    assignments, _ = hamming_kmeans(bits, n_clusters, iters=iters, seed=seed)

    # Distinct queries can hash to identical bit strings, so clusters can still
    # come back empty. Drop them and renumber, or the centroid of an empty
    # cluster is a nan that silently contaminates every later matmul.
    labels, assignments = np.unique(assignments, return_inverse=True)
    centroids = np.zeros((len(labels), Q.shape[1]), dtype=Q.dtype)
    for c in range(len(labels)):
        centroids[c] = Q[assignments == c].mean(axis=0)
    return assignments, centroids


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
    O(N M d) for the real thing. With C fixed this is LINEAR in sequence length.

    What it cannot do: causal masking. Members of a cluster sit at different
    positions, so there is no single mask that is correct for the group, and
    the centroid distribution would have to be recomputed per member — which is
    the cost you were avoiding. Clustered attention is an encoder method. This
    is the honest reason the paper evaluates it on speech recognition rather
    than language modelling, and the reason linear attention (next file) is the
    one that gets used for autoregressive decoding.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    scale = 1.0 / np.sqrt(Q.shape[1]) if scale is None else scale

    assignments, centroids = cluster_queries(Q, n_clusters, n_bits=n_bits,
                                             iters=iters, seed=seed)
    Ac = softmax(centroids @ K.T * scale, axis=-1)       # (C, M)
    Vc = Ac @ V                                          # (C, dv)
    O = Vc[assignments]                                  # (N, dv)

    if return_extras:
        return O, {"assignments": assignments, "centroids": centroids,
                   "Ac": Ac, "Vc": Vc, "n_clusters": len(centroids)}
    return O


# ---------------------------------------------------------------------------
# Step 5 — what the approximation guarantees
# ---------------------------------------------------------------------------

def attention_error_bound(Q, K, V, assignments, centroids, scale=None,
                          tight: bool = False) -> np.ndarray:
    """Per-query upper bound on || V_i - V_hat_i ||_2. Never computes attention.

    Derivation (three inequalities, each elementary):

      1. Write a = softmax(u), b = softmax(v) for the true and centroid logits,
         and d = ||u - v||_inf. Then for every n

             |log a_n - log b_n| <= |u_n - v_n| + |log Z_u - log Z_v| <= 2d

         so a_n <= b_n e^{2d}, hence ||a - b||_1 <= e^{2d} - 1.

      2. || sum_n (a_n - b_n) V_n ||_2 <= ||a - b||_1 * max_n ||V_n||_2.

      3. d = max_n |(Q_i - c) . K_n| / sqrt(d) <= ||Q_i - c||_2 * max_n ||K_n||_2 / sqrt(d)
         by Cauchy-Schwarz (`tight=True` skips this last step and uses the
         exact max over keys instead).

    Putting them together:

        ||V_i - V_hat_i||_2 <= ( exp(2 ||Q_i - c||_2 max_n||K_n||_2 / sqrt(d)) - 1 )
                               * max_n ||V_n||_2

    Read it for its shape, not its size. The exponential makes it numerically
    vacuous for anything but very tight clusters — at unit scale it can exceed
    the trivial bound 2 max||V||. What it says is the useful part, and it is
    the paper's argument: *the error is controlled by the distance from each
    query to its centroid, and vanishes as that distance does.* That is what
    licenses spending compute on better clustering, and it is why the method
    degrades gracefully rather than unpredictably.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    scale = 1.0 / np.sqrt(Q.shape[1]) if scale is None else scale

    residual = Q - centroids[assignments]                       # (N, d)
    if tight:
        delta = np.max(np.abs(residual @ K.T), axis=-1) * scale  # (N,)
    else:
        max_k = np.max(np.linalg.norm(K, axis=-1))
        delta = np.linalg.norm(residual, axis=-1) * max_k * scale

    max_v = np.max(np.linalg.norm(V, axis=-1))
    with np.errstate(over="ignore"):
        l1 = np.expm1(2.0 * delta)
    return np.minimum(l1, 2.0) * max_v      # 2*max||V|| is always true as well


# ---------------------------------------------------------------------------
# Step 6 — improved clustered attention
# ---------------------------------------------------------------------------

def improved_clustered_attention(Q, K, V, n_clusters: int = 8, top_k: int = 8,
                                 n_bits: int = 16, iters: int = 10, seed: int = 0,
                                 scale: Optional[float] = None,
                                 return_extras: bool = False):
    """Exact attention on each cluster's top-k keys, centroid approximation elsewhere.

    The observation: attention distributions are peaked, so a handful of keys
    carry most of the mass. Use the centroid pass only to *find* those keys,
    then pay for exact dot products against them for every query in the cluster.

    For query i in cluster j with top-k set T:

        w      = softmax over T of (Q_i . K_n / sqrt(d))    exact, k terms
        m_j    = sum_{n in T} A^c_{jn}                      centroid mass on T
        V_hat_i = m_j * sum_{n in T} w_n V_n  +  (V^c_j - sum_{n in T} A^c_{jn} V_n)

    The two pieces are "the mass the centroid says belongs to the top-k, spent
    according to the query's own preferences" and "everything else, spent as the
    centroid would have". The second is a subtraction from V^c_j, so it costs
    nothing beyond the pass already done.

    Cost: O(C M d) + O(N k d). Two exactness properties fall out, and both are
    worth checking rather than believing:

      * k = M makes it EXACT for every query, whatever the clustering. Then
        m_j = 1 and the fallback term is zero, so the formula collapses to
        softmax over all keys.
      * The error is bounded by 2[(1 - alpha_i) + (1 - m_j)] max||V||, where
        alpha_i is the query's own true mass on T. Missing mass is the only
        way to be wrong, which is a far more informative guarantee than the
        exponential of step 5.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d) if scale is None else scale
    top_k = min(top_k, K.shape[0])

    assignments, centroids = cluster_queries(Q, n_clusters, n_bits=n_bits,
                                             iters=iters, seed=seed)
    Ac = softmax(centroids @ K.T * scale, axis=-1)               # (C, M)
    Vc = Ac @ V                                                  # (C, dv)

    O = np.zeros((n, V.shape[1]), dtype=np.result_type(Q, V))
    kept_mass = np.zeros(len(centroids))

    for c in range(len(centroids)):
        members = np.flatnonzero(assignments == c)
        if len(members) == 0:
            continue
        topk = np.argpartition(Ac[c], -top_k)[-top_k:]            # (k,)
        mass = float(Ac[c, topk].sum())
        kept_mass[c] = mass

        w = softmax(Q[members] @ K[topk].T * scale, axis=-1)      # (n_c, k)
        exact_part = mass * (w @ V[topk])                         # (n_c, dv)
        fallback = Vc[c] - Ac[c, topk] @ V[topk]                  # (dv,)
        O[members] = exact_part + fallback

    if return_extras:
        return O, {"assignments": assignments, "centroids": centroids,
                   "Ac": Ac, "Vc": Vc, "kept_mass": kept_mass,
                   "n_clusters": len(centroids)}
    return O


def improved_error_bound(Q, K, V, extras, top_k: int, scale=None) -> np.ndarray:
    """Per-query bound 2[(1 - alpha_i) + (1 - m_j)] * max||V||_2.

    alpha_i, the query's true mass on the cluster's top-k, needs the true
    attention row and so is a *diagnostic*, not something you would compute in
    a real forward pass. It is here because seeing the bound track the measured
    error is what makes the guarantee mean anything.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    scale = 1.0 / np.sqrt(Q.shape[1]) if scale is None else scale
    Ac, assignments = extras["Ac"], extras["assignments"]
    top_k = min(top_k, K.shape[0])

    A = softmax(Q @ K.T * scale, axis=-1)
    alpha = np.zeros(len(Q))
    for c in range(len(extras["centroids"])):
        members = np.flatnonzero(assignments == c)
        if len(members) == 0:
            continue
        topk = np.argpartition(Ac[c], -top_k)[-top_k:]
        alpha[members] = A[np.ix_(members, topk)].sum(axis=-1)

    max_v = np.max(np.linalg.norm(V, axis=-1))
    return np.minimum(
        2.0 * ((1.0 - alpha) + (1.0 - extras["kept_mass"][assignments])), 2.0) * max_v


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
