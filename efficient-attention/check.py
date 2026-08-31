"""
Progress checker for the efficient-attention templates.

    python3 check.py           # run in order, stop at the first unimplemented step
    python3 check.py 7         # run only step 7
    python3 check.py 7 9       # run steps 7 through 9
    python3 check.py --all     # run everything, do not stop at the first gap

A check that raises NotImplementedError is reported as TODO, not a failure —
that is simply the next thing to write.

Nothing here imports solutions/. It tests YOUR code, against a small reference
attention written from scratch inside this file, so that a mistake in your
baseline cannot quietly excuse a mistake anywhere else.
"""

import math
import pathlib
import shutil
import sys
import traceback
from typing import Callable, List, Tuple

# Always read the learner's source fresh: python validates cached bytecode on
# (mtime, size), so an edit that keeps a file the same size within the same
# second can be masked by a stale __pycache__.
sys.dont_write_bytecode = True
shutil.rmtree(pathlib.Path(__file__).parent / "__pycache__", ignore_errors=True)

import numpy as np

PASS, FAIL, TODO, ERROR = "PASS", "FAIL", "TODO", "ERROR"
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")


# ---------------------------------------------------------------------------
# An independent reference, so the checks do not grade your code against itself
# ---------------------------------------------------------------------------

def ref_attention(Q, K, V, causal=False):
    """Returns (O, P, lse). Deliberately naive and quadratic."""
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    S = Q @ K.T / math.sqrt(Q.shape[1])
    if causal:
        S = np.where(np.tril(np.ones(S.shape, dtype=bool)), S, -np.inf)
    mx = S.max(axis=-1, keepdims=True)
    E = np.exp(S - mx)
    Z = E.sum(axis=-1, keepdims=True)
    return (E / Z) @ V, E / Z, (mx + np.log(Z)).ravel()


def ref_data(n=64, d=16, seed=0, scale=1.0, groups=None, spread=0.1):
    rng = np.random.default_rng(seed)
    if groups is None:
        Q = rng.standard_normal((n, d)) * scale
    else:
        centres = rng.standard_normal((groups, d)) * scale
        Q = centres[rng.integers(0, groups, n)] + rng.standard_normal((n, d)) * spread
    return Q, rng.standard_normal((n, d)) * scale, rng.standard_normal((n, d))


def numgrad(f, x, eps=1e-6):
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + eps
        a = f(x)
        x[idx] = old - eps
        b = f(x)
        x[idx] = old
        g[idx] = (a - b) / (2 * eps)
    return g


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def relerr(a, b):
    b = np.asarray(b)
    return float(np.linalg.norm(np.asarray(a) - b) / (np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------------------
# Steps 1-2 — baseline.py
# ---------------------------------------------------------------------------

def check_softmax_and_attention():
    from baseline import attention, logsumexp, softmax

    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(x)
    assert np.allclose(p.sum(axis=-1), 1.0), f"rows must sum to 1, got {p.sum(-1)}"
    assert np.allclose(p[1], 1 / 3), "equal logits must give a uniform row"

    big = softmax(np.array([[1000.0, 1000.0, 999.0]]))
    assert np.all(np.isfinite(big)), (
        "softmax overflowed on large logits. Subtract the row max first — it "
        "is exact, since softmax is invariant to a per-row shift.")
    assert abs(big[0, 0] - big[0, 1]) < 1e-12

    Q, K, V = ref_data(48, 16, seed=1)
    O, P, lse = ref_attention(Q, K, V)
    assert maxerr(attention(Q, K, V), O) < 1e-12, (
        f"attention disagrees with the reference by "
        f"{maxerr(attention(Q, K, V), O):.2e}. Check the 1/sqrt(d) scaling.")
    assert maxerr(logsumexp(Q @ K.T / math.sqrt(16)), lse) < 1e-12, \
        "logsumexp must return one number per row, stably"

    Oc = attention(Q, K, V, causal=True)
    assert maxerr(Oc, ref_attention(Q, K, V, causal=True)[0]) < 1e-12, \
        "causal attention disagrees with the reference"

    V2 = V.copy()
    V2[24:] += 100.0
    assert maxerr(attention(Q, K, V2, causal=True)[:24], Oc[:24]) < 1e-9, (
        "changing V at positions >= 24 changed the output at positions < 24: "
        "the causal mask is leaking the future. Masked entries must be -inf "
        "before the softmax, not merely small.")

    mask = np.zeros((48, 48), dtype=bool)
    mask[:, :4] = True
    masked = attention(Q, K, V, mask=mask)
    assert maxerr(masked, ref_attention(Q, K[:4], V[:4])[0]) < 1e-12, \
        "an explicit boolean mask must restrict attention to the True entries"


def check_backward_and_cost():
    from baseline import attention_backward, attention_cost, rowsum_dO_O

    Q, K, V = ref_data(12, 8, seed=2)
    dO = np.random.default_rng(9).standard_normal(V.shape)
    O, P, _ = ref_attention(Q, K, V)
    dQ, dK, dV = attention_backward(dO, Q, K, V, P)

    for name, g, wrt in (("dQ", dQ, "Q"), ("dK", dK, "K"), ("dV", dV, "V")):
        def loss(x, wrt=wrt):
            args = {"Q": Q, "K": K, "V": V}
            args[wrt] = x
            return float(np.sum(ref_attention(**args)[0] * dO))
        num = numgrad(loss, {"Q": Q, "K": K, "V": V}[wrt])
        err = maxerr(g, num)
        assert err < 1e-6, (
            f"{name} disagrees with the numerical gradient by {err:.2e}.\n"
            "      The usual cause is dS: softmax rows are coupled, so\n"
            "      dS = P * (dP - rowsum(P * dP)), not just P * dP.")

    dP = dO @ V.T
    assert maxerr(rowsum_dO_O(dO, O), np.sum(dP * P, axis=-1, keepdims=True)) < 1e-10, (
        "rowsum(dO * O) must equal rowsum(P * dP). This identity is what lets "
        "FlashAttention's backward pass avoid a second N x N intermediate.")

    cost = attention_cost(1024, d=64, itemsize=4)
    assert cost["score_elements"] == 1024 * 1024, \
        f"score_elements should be N*M = 1048576, got {cost['score_elements']}"
    assert cost["score_bytes"] == 1024 * 1024 * 4
    ratio = attention_cost(2048, d=64)["flops"] / attention_cost(1024, d=64)["flops"]
    assert 3.5 < ratio < 4.5, \
        f"doubling N should roughly quadruple the flops, got {ratio:.2f}x"


# ---------------------------------------------------------------------------
# Steps 3-6 — clustered.py
# ---------------------------------------------------------------------------

def check_lsh_and_kmeans():
    from clustered import hamming_distances, hamming_kmeans, lsh_bits

    Q, _, _ = ref_data(40, 16, seed=3)
    bits = lsh_bits(Q, n_bits=24, seed=0)
    assert bits.shape == (40, 24), f"expected (40, 24) bits, got {bits.shape}"
    assert set(np.unique(bits)) <= {0, 1}, "bits must be 0/1"
    assert maxerr(lsh_bits(Q, 24, seed=0), bits) == 0, \
        "the same seed must give the same hyperplanes, or clustering is not reproducible"
    assert np.array_equal(lsh_bits(np.vstack([Q[0], Q[0] * 3]), 24, seed=0)[0],
                          lsh_bits(np.vstack([Q[0], Q[0] * 3]), 24, seed=0)[1]), (
        "a query and a scaled copy of it must hash identically — random "
        "hyperplane LSH sees direction only. (That blindness to magnitude is a "
        "real limitation of the method, not a bug in your code.)")

    a = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
    b = np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    D = hamming_distances(a, b)
    assert D.shape == (2, 2) and D[0, 0] == 0 and D[0, 1] == 1 and D[1, 0] == 2, \
        f"hamming_distances is wrong: {D}"

    rng = np.random.default_rng(7)
    prototypes = np.array([[0] * 12, [1] * 12, [0] * 6 + [1] * 6], dtype=np.uint8)
    noisy = np.repeat(prototypes, 20, axis=0)
    flips = rng.random(noisy.shape) < 0.05
    noisy = (noisy ^ flips).astype(np.uint8)

    assignments, centroids = hamming_kmeans(noisy, 3, iters=20, seed=0)
    assert assignments.shape == (60,) and centroids.shape == (3, 12)
    counts = np.bincount(assignments, minlength=3)
    assert (counts > 0).all(), (
        f"an empty cluster survived: {counts}. Re-seed empty clusters — left "
        "alone they silently reduce the number of clusters you actually get.")
    for group in range(3):
        labels = assignments[group * 20:(group + 1) * 20]
        assert len(set(labels.tolist())) == 1, (
            f"the 20 members of well-separated group {group} landed in "
            f"{len(set(labels.tolist()))} different clusters: {labels}")


def check_clustered_attention():
    from clustered import clustered_attention

    Q, K, V = ref_data(96, 16, seed=4, groups=8, spread=0.08, scale=2.0)
    exact = ref_attention(Q, K, V)[0]

    O, extras = clustered_attention(Q, K, V, n_clusters=32, return_extras=True)
    assert O.shape == V.shape, f"expected {V.shape}, got {O.shape}"
    assert extras["Ac"].shape[1] == K.shape[0], \
        "A^c has one column per KEY and one row per CLUSTER"
    assert extras["Ac"].shape[0] <= 32
    assert np.allclose(extras["Ac"].sum(axis=-1), 1.0), \
        "each centroid's attention row must still be a distribution"

    err = relerr(O, exact)
    assert err < 0.2, (
        f"relative error {err:.3f} on queries that genuinely form 8 groups, "
        "with 32 clusters available. Check that the centroid is the MEAN of "
        "its cluster's queries in R^d, and that A^c is softmax over keys.")

    one = clustered_attention(Q, K, V, n_clusters=1)
    assert maxerr(one, one[0]) < 1e-12, \
        "with a single cluster every query must get the same output row"

    Qd, Kd, Vd = ref_data(24, 16, seed=11, scale=2.0)
    every = clustered_attention(Qd, Kd, Vd, n_clusters=24, n_bits=32)
    assert maxerr(every, ref_attention(Qd, Kd, Vd)[0]) < 1e-10, (
        f"with one cluster per query the approximation must be EXACT "
        f"(each centroid is then the query itself), got error "
        f"{maxerr(every, ref_attention(Qd, Kd, Vd)[0]):.2e}")


def check_error_bound():
    from clustered import attention_error_bound, clustered_attention

    max_bound = None
    for seed in range(4):
        for spread in (0.4, 0.1, 0.02):
            Q, K, V = ref_data(64, 16, seed=seed, groups=4, spread=spread, scale=1.5)
            O, extras = clustered_attention(Q, K, V, n_clusters=8, return_extras=True)
            actual = np.linalg.norm(O - ref_attention(Q, K, V)[0], axis=-1)

            loose = attention_error_bound(Q, K, V, extras["assignments"],
                                          extras["centroids"])
            tight = attention_error_bound(Q, K, V, extras["assignments"],
                                          extras["centroids"], tight=True)
            assert loose.shape == (64,) and tight.shape == (64,), \
                "the bound is per query: one number for each row of Q"
            bad = int(np.sum(actual > loose + 1e-9))
            assert bad == 0, (
                f"{bad} queries exceeded the bound at spread={spread}, seed={seed} "
                f"(worst: actual {actual.max():.4f} vs bound "
                f"{loose[np.argmax(actual - loose)]:.4f}). A bound that can be "
                "violated is not a bound.")
            assert np.all(tight <= loose + 1e-9), (
                "the tight form (exact max over keys) must never exceed the "
                "loose form (Cauchy-Schwarz on the key norms)")
            if spread == 0.02:
                max_bound = tight.mean() if max_bound is None else max(max_bound,
                                                                      tight.mean())

    trivial = None
    Q, K, V = ref_data(64, 16, seed=0, groups=4, spread=0.02, scale=1.5)
    trivial = 2 * np.max(np.linalg.norm(V, axis=-1))
    assert max_bound < 0.5 * trivial, (
        f"at spread=0.02 the bound averages {max_bound:.3f}, no better than the "
        f"trivial 2*max||V|| = {trivial:.3f}. It should become informative as "
        "clusters tighten — check the exponent: 2 * ||Q_i - c|| * max||K|| / sqrt(d).")


def check_improved_clustered():
    from clustered import clustered_attention, improved_clustered_attention

    Q, K, V = ref_data(96, 16, seed=4, groups=8, spread=0.08, scale=2.0)
    exact = ref_attention(Q, K, V)[0]

    full = improved_clustered_attention(Q, K, V, n_clusters=3, top_k=96)
    assert maxerr(full, exact) < 1e-10, (
        f"with top_k = number of keys the result must be EXACT for every query, "
        f"whatever the clustering (3 clusters here), got {maxerr(full, exact):.2e}.\n"
        "      Then the kept mass is 1 and the centroid fallback term is zero,\n"
        "      so the formula collapses to a plain softmax over all keys.")

    plain = relerr(clustered_attention(Q, K, V, n_clusters=16), exact)
    errors = [relerr(improved_clustered_attention(Q, K, V, n_clusters=16, top_k=k),
                     exact) for k in (4, 16, 48)]
    assert errors[0] < plain, (
        f"improved with top_k=4 ({errors[0]:.4f}) should already beat plain "
        f"clustered attention ({plain:.4f}) at the same cluster count")
    assert errors[2] < errors[1] < errors[0], (
        f"error must fall as top_k grows, got {errors}. If it does not, the "
        "kept-mass rescaling is probably wrong: the exact part must be scaled "
        "by the centroid's mass on the top-k keys, and the remainder taken "
        "from V^c minus that mass's contribution.")
    assert errors[2] < 0.05, f"top_k=48 of 96 keys still has error {errors[2]:.4f}"


# ---------------------------------------------------------------------------
# Steps 7-10 — linear.py
# ---------------------------------------------------------------------------

def check_feature_map_and_associativity():
    from linear import (elu_feature_map, elu_feature_map_grad, linear_attention,
                        quadratic_linear_attention)

    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    phi = elu_feature_map(x)
    assert np.all(phi > 0), (
        f"phi must be strictly positive everywhere, got {phi}. Positivity is "
        "what keeps the denominator from vanishing or changing sign.")
    assert np.allclose(phi, np.where(x > 0, x + 1, np.exp(x))), \
        f"elu(x)+1 is x+1 for x>0 and exp(x) for x<=0, got {phi}"
    assert np.allclose(elu_feature_map_grad(x),
                       numgrad(lambda t: float(elu_feature_map(t).sum()), x), atol=1e-6)

    Q, K, V = ref_data(64, 16, seed=6)
    fast, slow = linear_attention(Q, K, V), quadratic_linear_attention(Q, K, V)
    assert maxerr(fast, slow) < 1e-9, (
        f"the associativity rearrangement changed the answer by {maxerr(fast, slow):.2e}. "
        "phi(Q) (phi(K)^T V) must equal (phi(Q) phi(K)^T) V exactly — if not, "
        "check which factor the normaliser is applied to.")

    from common import timeit
    Q, K, V = ref_data(768, 32, seed=7)
    t_lin = timeit(lambda: linear_attention(Q, K, V))
    t_quad = timeit(lambda: quadratic_linear_attention(Q, K, V))
    assert t_lin < t_quad, (
        f"the linear form ({t_lin * 1e3:.2f}ms) was not faster than the "
        f"quadratic one ({t_quad * 1e3:.2f}ms) at N=768 — it is probably still "
        "forming the N x N matrix somewhere.")


def check_causal_linear():
    from linear import causal_linear_attention, quadratic_linear_attention

    Q, K, V = ref_data(48, 12, seed=8)
    loop = causal_linear_attention(Q, K, V, method="loop")
    cumsum = causal_linear_attention(Q, K, V, method="cumsum")
    masked = quadratic_linear_attention(Q, K, V, causal=True)

    assert maxerr(loop, masked) < 1e-9, (
        f"the causal scan disagrees with the masked quadratic form by "
        f"{maxerr(loop, masked):.2e}. Position i must use the prefix sums "
        "INCLUDING j = i.")
    assert maxerr(loop, cumsum) < 1e-9, \
        "the loop and cumsum implementations must agree exactly"

    K2, V2 = K.copy(), V.copy()
    K2[24:] += 50.0
    V2[24:] += 50.0
    assert maxerr(causal_linear_attention(Q, K2, V2)[:24], loop[:24]) < 1e-9, \
        "changing K/V after position 24 changed outputs before it"

    first = causal_linear_attention(Q[:1], K[:1], V[:1])
    assert maxerr(first[0], V[0]) < 1e-4, (
        "with one token attending only to itself, the output must be V[0] (up "
        f"to the eps in the denominator); got a difference of "
        f"{maxerr(first[0], V[0]):.2e}")


def check_recurrent_form():
    from linear import RecurrentLinearAttention, causal_linear_attention

    Q, K, V = ref_data(40, 12, seed=9)
    rnn = RecurrentLinearAttention(12, 12)
    streamed = rnn.run(Q, K, V)
    parallel = causal_linear_attention(Q, K, V)
    assert maxerr(streamed, parallel) < 1e-9, (
        f"decoding one token at a time gave a different answer from the "
        f"parallel causal form ({maxerr(streamed, parallel):.2e}). They are the "
        "same recurrence.")

    small = RecurrentLinearAttention(12, 12)
    small.step(Q[0], K[0], V[0])
    after_one = small.state_elements
    for i in range(1, 40):
        small.step(Q[i], K[i], V[i])
    assert small.state_elements == after_one, (
        f"the state grew from {after_one} to {small.state_elements} numbers "
        "over 40 tokens. The whole claim of the paper is that it does not: a "
        "KV cache grows with t, this state does not.")
    assert after_one == 12 * 12 + 12, \
        f"state should be d_k*d_v + d_k = 156 numbers, got {after_one}"

    resumed = RecurrentLinearAttention(12, 12)
    for i in range(30):
        resumed.step(Q[i], K[i], V[i])
    out = resumed.step(Q[30], K[30], V[30])
    assert maxerr(out, parallel[30]) < 1e-9, \
        "the 31st decoded token must match row 30 of the parallel computation"


def check_causal_linear_backward():
    from linear import causal_linear_attention, causal_linear_attention_backward

    Q, K, V = ref_data(10, 6, seed=10)
    dO = np.random.default_rng(4).standard_normal(V.shape)
    dQ, dK, dV = causal_linear_attention_backward(dO, Q, K, V)

    for name, g, wrt in (("dQ", dQ, "Q"), ("dK", dK, "K"), ("dV", dV, "V")):
        def loss(x, wrt=wrt):
            args = {"Q": Q, "K": K, "V": V}
            args[wrt] = x
            return float(np.sum(causal_linear_attention(**args) * dO))
        num = numgrad(loss, {"Q": Q, "K": K, "V": V}[wrt])
        err = maxerr(g, num)
        hint = ""
        if name == "dK":
            hint = ("\n      dphi(K_j) needs the REVERSE cumulative sum "
                    "G_j = sum_{i>=j} phi(Q_i) dN_i^T:\n"
                    "      key j influences every query at or after it.")
        if name == "dQ":
            hint = ("\n      dphi(Q_i) = S_i dN_i uses the FORWARD state at i, "
                    "and dN_i\n      carries both the numerator and the "
                    "denominator gradient.")
        assert err < 1e-6, f"{name} is off by {err:.2e} from the numerical gradient.{hint}"


# ---------------------------------------------------------------------------
# Steps 11-15 — flash.py
# ---------------------------------------------------------------------------

def check_standard_attention_io():
    from flash import standard_attention_io
    from io_model import Device

    n, d = 96, 16
    Q, K, V = ref_data(n, d, seed=12)
    dev = Device(sram_bytes=16 * 1024)
    O = standard_attention_io(dev, dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V"))
    assert maxerr(O.to_numpy(), ref_attention(Q, K, V)[0]) < 1e-12, \
        "the IO-model version must compute the same thing as plain attention"

    assert dev.largest_allocation >= n * n, (
        f"the largest HBM tensor is {dev.largest_allocation} elements; standard "
        f"attention must materialise the N x N score matrix ({n * n}) — that is "
        "the thing FlashAttention removes, and the comparison is the point.")
    assert dev.peak_sram <= dev.sram_bytes
    moved = dev.total_bytes / (n * n * 8)
    assert moved > 3.0, (
        f"only {moved:.1f} N^2-sized transfers were charged. Standard attention "
        "writes S, reads it to find the row maxima, reads it again for the row "
        "sums, reads and writes it to normalise, and reads P for the second "
        "matmul. Load and store through sram.load/sram.store so they count.")


def check_online_softmax():
    from flash import online_softmax, online_softmax_update

    rng = np.random.default_rng(13)
    S = rng.standard_normal((8, 40)) * 3
    V = rng.standard_normal((40, 6))
    E = np.exp(S - S.max(axis=-1, keepdims=True))
    ref_O = (E / E.sum(axis=-1, keepdims=True)) @ V
    ref_L = S.max(axis=-1) + np.log(np.exp(S - S.max(axis=-1, keepdims=True)).sum(-1))

    for block in (1, 5, 7, 40):
        O, L = online_softmax(S, V, block=block)
        assert maxerr(O, ref_O) < 1e-12, (
            f"streaming in blocks of {block} gave a different answer "
            f"({maxerr(O, ref_O):.2e}). When the running maximum rises, the "
            "accumulator AND the running sum must both be rescaled by "
            "exp(m_old - m_new).")
        assert maxerr(L, ref_L) < 1e-12, "the returned logsumexp is wrong"

    m = np.full(8, -np.inf)
    l = np.zeros(8)
    acc = np.zeros((8, 6))
    m1, l1, acc1 = online_softmax_update(m, l, acc, S[:, :10], V[:10])
    assert np.allclose(m1, S[:, :10].max(axis=-1)), \
        "after the first block the running max is that block's max"
    assert np.allclose(acc1, np.exp(S[:, :10] - m1[:, None]) @ V[:10]), \
        "the accumulator must be UNNORMALISED: sum_j exp(s_j - m) v_j"

    huge = S + 800.0
    O_huge, _ = online_softmax(huge, V, block=7)
    assert np.all(np.isfinite(O_huge)) and maxerr(O_huge, ref_O) < 1e-9, (
        "adding 800 to every logit changed the answer or overflowed. Every "
        "exponent must be <= 0 by construction.")

    order = np.array([3, 0, 4, 1, 2])
    perm = np.concatenate([np.arange(i * 8, i * 8 + 8) for i in order])
    O_perm, _ = online_softmax(S[:, perm], V[perm], block=8)
    assert maxerr(O_perm, ref_O) < 1e-12, \
        "the result must not depend on the order blocks are streamed in"


def check_flash_forward():
    from flash import flash_attention, flash_block_sizes, flash_forward
    from io_model import Device

    br, bc = flash_block_sizes(64 * 1024, 32, itemsize=8)
    assert br >= 1 and bc >= 1
    assert br <= 32, f"B_r must be capped at d (=32) so the score tile fits, got {br}"
    assert (2 * bc * 32 + 2 * br * 32 + br * bc) * 8 <= 64 * 1024, (
        f"B_r={br}, B_c={bc} do not fit: K, V, Q, O blocks and the B_r x B_c "
        "score tile must be resident at once")
    assert flash_block_sizes(16 * 1024, 32)[1] < bc, \
        "a smaller SRAM budget must give smaller blocks — derive them from it"

    n, d = 128, 32
    Q, K, V = ref_data(n, d, seed=14, scale=2.0)
    exact, _, lse = ref_attention(Q, K, V)

    (O, L), dev = flash_attention(Q, K, V, return_device=True)
    assert maxerr(O, exact) < 1e-12, (
        f"FlashAttention is EXACT, not approximate — error {maxerr(O, exact):.2e} "
        "is too large to be float64 reassociation.")
    assert maxerr(L, lse) < 1e-12, \
        "L must be the row logsumexp m + log(l); the backward pass needs it"
    assert dev.largest_allocation <= 4 * n * d, (
        f"the largest HBM tensor was {dev.largest_allocation} elements. Nothing "
        f"of size N^2 = {n * n} may be allocated: the score tile lives in SRAM.")

    dev2 = Device(sram_bytes=8 * 1024)
    Oh, _ = flash_forward(dev2, dev2.hbm(Q, "Q"), dev2.hbm(K, "K"), dev2.hbm(V, "V"))
    assert maxerr(Oh.to_numpy(), exact) < 1e-12, \
        "the answer must not depend on the SRAM budget, only the block sizes do"

    from flash import standard_attention_io
    std = Device()
    standard_attention_io(std, std.hbm(Q, "Q"), std.hbm(K, "K"), std.hbm(V, "V"))
    assert dev.total_bytes < std.total_bytes, (
        f"flash moved {dev.total_bytes} bytes, standard attention "
        f"{std.total_bytes}. Flash should move fewer: it never sends an N x N "
        "matrix to HBM.")


def check_flash_backward():
    from flash import flash_attention, flash_attention_backward

    n, d = 96, 16
    Q, K, V = ref_data(n, d, seed=15)
    O_ref, P, _ = ref_attention(Q, K, V)
    dO = np.random.default_rng(6).standard_normal(V.shape)

    dV_ref = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))
    dQ_ref, dK_ref = dS @ K / math.sqrt(d), dS.T @ Q / math.sqrt(d)

    (O, L) = flash_attention(Q, K, V)
    (dQ, dK, dV), dev = flash_attention_backward(dO, Q, K, V, O, L,
                                                 return_device=True)
    for name, got, want in (("dQ", dQ, dQ_ref), ("dK", dK, dK_ref), ("dV", dV, dV_ref)):
        assert maxerr(got, want) < 1e-10, (
            f"{name} is off by {maxerr(got, want):.2e}. P is recomputed as "
            "exp(S - L), which is exact because L is the row log-normaliser; "
            "and D = rowsum(dO * O) replaces rowsum(P * dP).")
    assert dev.largest_allocation <= 4 * n * d, (
        f"backward allocated a tensor of {dev.largest_allocation} elements. It "
        "must not store or rebuild P in HBM — that is the point of keeping L.")


def check_causal_flash():
    from flash import flash_attention

    n, d = 128, 16
    Q, K, V = ref_data(n, d, seed=16, scale=2.0)
    exact = ref_attention(Q, K, V, causal=True)[0]
    (O, L), dev = flash_attention(Q, K, V, causal=True, return_device=True)
    assert maxerr(O, exact) < 1e-12, \
        f"causal flash disagrees with the reference by {maxerr(O, exact):.2e}"

    computed = dev.counters.get("blocks_computed", 0)
    skipped = dev.counters.get("blocks_skipped", 0)
    assert skipped > 0, (
        "no blocks were skipped. A block of keys entirely in the future of a "
        "block of queries is fully masked, and computing it is pure waste — "
        "skipping is what makes causal attention ~2x cheaper rather than the "
        "same cost with half the output discarded.")
    fraction = computed / (computed + skipped)
    assert 0.45 < fraction < 0.75, (
        f"{computed} blocks computed and {skipped} skipped ({fraction:.0%}). "
        "Roughly half the blocks lie strictly above the diagonal; diagonal "
        "blocks must still be computed (and masked elementwise).")


# ---------------------------------------------------------------------------
# Steps 16-18 — flash2.py
# ---------------------------------------------------------------------------

def check_flash2_forward():
    from flash import flash_attention
    from flash2 import flash2_attention, flash2_forward
    from io_model import Device

    n, d = 128, 32
    Q, K, V = ref_data(n, d, seed=17, scale=2.0)
    exact, _, lse = ref_attention(Q, K, V)

    (O1, L1), dev1 = flash_attention(Q, K, V, return_device=True)
    (O2, L2), dev2 = flash2_attention(Q, K, V, return_device=True)
    assert maxerr(O2, exact) < 1e-12, \
        f"FlashAttention-2 must be exact too, got {maxerr(O2, exact):.2e}"
    assert maxerr(L2, lse) < 1e-12, "L is still the row logsumexp"

    ops1 = dev1.counters.get("o_elementwise_ops", 0)
    ops2 = dev2.counters.get("o_elementwise_ops", 0)
    assert ops1 > 0 and ops2 > 0, \
        "count o_elementwise_ops in both files, or there is nothing to compare"
    assert ops2 < 0.75 * ops1, (
        f"FA-2 did {ops2} elementwise ops on the accumulator against FA-1's "
        f"{ops1}. Keep the accumulator UNNORMALISED inside the inner loop and "
        "divide by l once at the end; FA-1 un-normalises and re-normalises "
        "every step.")
    assert dev2.bytes_written < dev1.bytes_written, (
        f"FA-2 wrote {dev2.bytes_written} bytes, FA-1 {dev1.bytes_written}. With "
        "the loops swapped, O_i is written exactly once instead of once per "
        "K/V block.")

    from flash import flash_block_sizes
    dev = Device()
    Qh, Kh, Vh = dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V")
    full = flash2_forward(Device(), Qh, Kh, Vh)[0].to_numpy()
    br = flash_block_sizes(dev.sram_bytes, d, 8, "fa2")[0]
    n_blocks = math.ceil(n / br)
    pieces = [flash2_forward(Device(), Qh, Kh, Vh, only_rows=[i])[0].to_numpy()
              for i in reversed(range(n_blocks))]
    stitched = sum(pieces)
    assert maxerr(stitched, full) < 1e-12, (
        "computing row blocks one at a time, in isolation, gave a different "
        "answer. Under FA-2's loop order every outer iteration is independent — "
        "that independence is what lets one long sequence fill a whole GPU.")


def check_split_kv():
    from flash2 import combine_partials, flash2_partial
    from io_model import Device

    n, d = 64, 16
    Q, K, V = ref_data(n, d, seed=18, scale=2.0)
    dev = Device()
    Qh, Kh, Vh = dev.hbm(Q, "Q"), dev.hbm(K, "K"), dev.hbm(V, "V")

    exact, _, lse = ref_attention(Q, K, V)
    whole = [flash2_partial(Device(), Qh, Kh, Vh, slice(0, n))]
    O_one, L_one = combine_partials(whole)
    assert maxerr(O_one, exact) < 1e-12 and maxerr(L_one, lse) < 1e-12, \
        "combining a single partial result must be the identity"

    for splits in ([slice(0, 32), slice(32, 64)],
                   [slice(0, 5), slice(5, 33), slice(33, 64)],
                   [slice(0, 1), slice(1, 2), slice(2, 64)]):
        partials = [flash2_partial(Device(), Qh, Kh, Vh, s) for s in splits]
        O, L = combine_partials(partials)
        assert maxerr(O, exact) < 1e-12, (
            f"merging {len(splits)} key splits gave error {maxerr(O, exact):.2e}. "
            "Each piece is normalised by its own denominator, so weight piece p "
            "by exp(m_p - m) * l_p before summing.")
        assert maxerr(L, lse) < 1e-12, "the merged logsumexp is wrong"

    cexact, _, clse = ref_attention(Q, K, V, causal=True)
    splits = [slice(0, 7), slice(7, 40), slice(40, 64)]
    partials = [flash2_partial(Device(), Qh, Kh, Vh, s, causal=True) for s in splits]
    O, L = combine_partials(partials)
    assert maxerr(O, cexact) < 1e-12, (
        f"causal split-K gave error {maxerr(O, cexact):.2e}. Some splits are "
        "entirely in the future for some queries: those pieces have l = 0 and "
        "m = -inf and must contribute nothing, without producing nan.")


def check_flash2_backward():
    from flash import flash_attention, flash_attention_backward
    from flash2 import flash2_attention_backward

    n, d = 96, 16
    Q, K, V = ref_data(n, d, seed=19)
    dO = np.random.default_rng(8).standard_normal(V.shape)
    (O, L) = flash_attention(Q, K, V)

    (g1, dev1) = flash_attention_backward(dO, Q, K, V, O, L, return_device=True)
    (g2, dev2) = flash2_attention_backward(dO, Q, K, V, O, L, return_device=True)
    for name, a, b in zip(("dQ", "dK", "dV"), g2, g1):
        assert maxerr(a, b) < 1e-10, \
            f"{name} disagrees with the FlashAttention-1 backward by {maxerr(a, b):.2e}"

    stores1 = dev1.counters.get("dq_stores", 0)
    stores2 = dev2.counters.get("dq_stores", 0)
    assert stores1 > 0 and stores2 > 0, "count dq_stores in both backward passes"
    assert stores2 < stores1 / 2, (
        f"dQ was written {stores2} times against FA-1's {stores1}. Running one "
        "pass with the outer loop over Q blocks lets dQ_i accumulate in SRAM "
        "and be stored once — no atomic add, no read-modify-write through HBM.")

    for name, a, b in zip(("dQ", "dK", "dV"), g2,
                          flash2_attention_backward(dO, Q, K, V, O, L)):
        assert maxerr(a, b) == 0.0, f"{name} is not deterministic across runs"

    (Oc, Lc) = flash_attention(Q, K, V, causal=True)
    gc1 = flash_attention_backward(dO, Q, K, V, Oc, Lc, causal=True)
    gc2 = flash2_attention_backward(dO, Q, K, V, Oc, Lc, causal=True)
    for name, a, b in zip(("dQ", "dK", "dV"), gc2, gc1):
        assert maxerr(a, b) < 1e-10, f"causal {name} disagrees by {maxerr(a, b):.2e}"


# ---------------------------------------------------------------------------
# Step 19 — benchmark.py
# ---------------------------------------------------------------------------

def check_benchmark():
    from benchmark import (count_flops, count_hbm_bytes, fit_slope,
                           measure_time, peak_extra_elements)

    assert abs(fit_slope([1, 2, 4, 8], [1, 4, 16, 64]) - 2.0) < 1e-9, \
        "fit_slope must return the exponent: log y against log x"

    soft = [count_flops("softmax", n, 64) for n in (256, 512, 1024)]
    lin = [count_flops("linear", n, 64) for n in (256, 512, 1024)]
    assert abs(fit_slope([256, 512, 1024], soft) - 2.0) < 0.05, \
        f"softmax attention must be quadratic in N, measured exponent {fit_slope([256, 512, 1024], soft):.2f}"
    assert abs(fit_slope([256, 512, 1024], lin) - 1.0) < 0.05, \
        f"linear attention must be linear in N, measured exponent {fit_slope([256, 512, 1024], lin):.2f}"
    clu = count_flops("clustered", 1024, 64, clusters=32)
    assert clu < 0.2 * count_flops("softmax", 1024, 64), (
        f"clustered attention with 32 clusters at N=1024 should cost far less "
        f"than full attention; you have {clu / count_flops('softmax', 1024, 64):.2f}x")
    assert count_flops("improved", 1024, 64, clusters=32, top_k=32) > clu, \
        "improved clustered attention does strictly more work than plain"

    assert peak_extra_elements("softmax", 512, 64) == 512 * 512, \
        "standard attention's intermediate is the N x N score matrix"
    assert (peak_extra_elements("linear", 512, 64)
            == peak_extra_elements("linear", 8192, 64)), (
        "linear attention's state does not depend on the sequence length — "
        "that is the entire point")
    assert peak_extra_elements("flash", 4096, 64) <= 64 * 1024 // 8, \
        "flash's live intermediate is one tile, and it fits in SRAM by construction"

    bytes_std = count_hbm_bytes("softmax", 256, 32)
    bytes_flash = count_hbm_bytes("flash", 256, 32)
    assert bytes_flash < bytes_std / 1.5, (
        f"measured HBM traffic: standard {bytes_std}, flash {bytes_flash}. "
        "Flash should move materially less.")

    t_soft = measure_time("softmax", 1024, 32)
    t_lin = measure_time("linear", 1024, 32)
    assert t_lin < t_soft, (
        f"at N=1024 linear attention ({t_lin * 1e3:.2f}ms) should beat softmax "
        f"attention ({t_soft * 1e3:.2f}ms) in wall clock, not just in flops")


CHECKS: List[Tuple[str, str, Callable[[], None]]] = [
    ("baseline.py", "stable softmax and attention", check_softmax_and_attention),
    ("baseline.py", "backward pass and cost model", check_backward_and_cost),
    ("clustered.py", "LSH bits and Hamming K-means", check_lsh_and_kmeans),
    ("clustered.py", "attention once per cluster", check_clustered_attention),
    ("clustered.py", "the approximation bound", check_error_bound),
    ("clustered.py", "improved: exact on the top-k", check_improved_clustered),
    ("linear.py", "feature map and associativity", check_feature_map_and_associativity),
    ("linear.py", "causal masking as a prefix sum", check_causal_linear),
    ("linear.py", "the same thing as an RNN", check_recurrent_form),
    ("linear.py", "backward in linear memory", check_causal_linear_backward),
    ("flash.py", "standard attention, IO counted", check_standard_attention_io),
    ("flash.py", "online softmax", check_online_softmax),
    ("flash.py", "tiled forward, exact", check_flash_forward),
    ("flash.py", "backward by recomputation", check_flash_backward),
    ("flash.py", "causal block skipping", check_causal_flash),
    ("flash2.py", "loop swap, one division", check_flash2_forward),
    ("flash2.py", "split-K and merging partials", check_split_kv),
    ("flash2.py", "backward without atomics", check_flash2_backward),
    ("benchmark.py", "the asymptotics, measured", check_benchmark),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(check: Callable[[], None]) -> Tuple[str, str]:
    try:
        check()
        return PASS, ""
    except NotImplementedError as exc:
        where = ""
        for frame in reversed(traceback.extract_tb(sys.exc_info()[2])):
            if frame.filename.endswith(".py") and "check.py" not in frame.filename:
                where = (f"{frame.filename.split('/')[-1]}:{frame.lineno} "
                         f"in {frame.name}()")
                break
        return TODO, (str(exc) or where)
    except AssertionError as exc:
        return FAIL, str(exc) or "assertion failed"
    except Exception as exc:                       # noqa: BLE001
        where = ""
        for frame in reversed(traceback.extract_tb(sys.exc_info()[2])):
            if "check.py" not in frame.filename:
                where = (f"\n      at {frame.filename.split('/')[-1]}:"
                         f"{frame.lineno} in {frame.name}()")
                break
        return ERROR, f"{type(exc).__name__}: {exc}{where}"


def main(argv: List[str]) -> int:
    keep_going = "--all" in argv
    wanted = [int(a) for a in argv if a.isdigit()]
    if len(wanted) > 1:
        wanted = list(range(min(wanted), max(wanted) + 1))

    print(f"\n{BOLD}Efficient attention from scratch — progress check{RESET}")
    print(f"{GREY}implement the templates, re-run this after each step{RESET}\n")

    passed = failed = todo = 0
    first_gap = None

    for index, (filename, title, check) in enumerate(CHECKS, start=1):
        if wanted and index not in wanted:
            continue

        status, detail = run_one(check)
        if status == PASS:
            passed += 1
            print(f"  {GREEN}✓{RESET} {index:>2}. {filename:<14} {title}")
        elif status == TODO:
            todo += 1
            first_gap = first_gap or index
            print(f"  {GREY}·{RESET} {index:>2}. {filename:<14} {title}")
            print(f"      {GREY}not implemented yet"
                  f"{(' — ' + detail) if detail else ''}{RESET}")
            if not keep_going and not wanted:
                remaining = len(CHECKS) - index
                if remaining:
                    print(f"\n  {GREY}({remaining} later checks not run; "
                          f"use --all to run them anyway){RESET}")
                break
        else:
            failed += 1
            first_gap = first_gap or index
            colour = RED if status == FAIL else YELLOW
            print(f"  {colour}✗{RESET} {index:>2}. {filename:<14} {title}")
            for line in detail.splitlines():
                print(f"      {colour}{line}{RESET}")

    total = len(wanted) if wanted else len(CHECKS)
    print(f"\n  {passed}/{total} passing", end="")
    if failed:
        print(f", {RED}{failed} failing{RESET}", end="")
    if todo:
        print(f", {GREY}{todo} to write{RESET}", end="")
    print()

    if passed == len(CHECKS):
        print(f"\n  {GREEN}{BOLD}All checks pass — you have implemented clustered "
              f"attention,{RESET}")
        print(f"  {GREEN}{BOLD}linear attention and FlashAttention 1 and 2.{RESET}")
        print(f"  {GREY}Run each file's own demo for the measurements, then read "
              f"THEORY.md{RESET}")
        print(f"  {GREY}with your implementation open beside it.{RESET}\n")
    elif first_gap:
        filename, title, _ = CHECKS[first_gap - 1]
        print(f"\n  {BOLD}Next:{RESET} step {first_gap} — {title} ({filename})")
        print(f"  {GREY}The TODO comments in that file walk through it. "
              f"Stuck? solutions/{filename}{RESET}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
