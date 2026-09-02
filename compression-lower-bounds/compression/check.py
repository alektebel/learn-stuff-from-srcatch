"""
Progress checker for the compression templates.

    python3 check.py           # every check, stop at the first unimplemented step
    python3 check.py 4         # step 4 only
    python3 check.py 2 4       # steps 2 through 4
    python3 check.py --all     # run everything, do not stop at the first gap

A check that raises NotImplementedError reports TODO, not FAIL: that is simply the next
thing to write. There is no solutions/ directory. This file tests YOUR code and, where
it can, tests an INVARIANT rather than a value, so that a passing check means the thing
is right rather than that it happens to match a number.
"""

import math
import pathlib
import shutil
import sys
import traceback

# Read the learner's source fresh: Python validates cached bytecode on (mtime, size), so
# an edit that keeps a file the same size within one second can be masked by a stale
# __pycache__ -- and a checker you cannot trust is worse than no checker.
sys.dont_write_bytecode = True
shutil.rmtree(pathlib.Path(__file__).parent / "__pycache__", ignore_errors=True)
sys.path.insert(0, str(pathlib.Path(__file__).parent))

try:
    import numpy as np
except ImportError:
    print("This track needs numpy:  pip install numpy")
    raise SystemExit(1)

PASS, FAIL, TODO, ERROR = "PASS", "FAIL", "TODO", "ERROR"
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m")

SCALAR_D_AT_R2 = 0.117482          # scalar Lloyd-Max, unit Gaussian, 4 levels
PANTER_DITE = math.sqrt(3) * math.pi / 2
LLOYD_TABLE = {2: 0.36338023, 4: 0.11748185, 8: 0.03454776,
               16: 0.00950101, 32: 0.00250469}


# ---------------------------------------------------------------------------
# Step 1 -- objective.py
# ---------------------------------------------------------------------------

def check_objective():
    from objective import encode, distortion, centroids, centroid_gap

    rng = np.random.default_rng(0)

    for d in (1, 3):
        x = rng.standard_normal((200, d)) if d > 1 else rng.standard_normal(200)
        c = rng.standard_normal((7, d)) if d > 1 else rng.standard_normal(7)
        got = np.asarray(encode(x, c))
        xx = x.reshape(200, -1); cc = np.asarray(c).reshape(7, -1)
        want = np.argmin(((xx[:, None, :] - cc[None, :, :]) ** 2).sum(-1), axis=1)
        bad = int((got != want).sum())
        assert bad == 0, (
            f"encode disagrees with brute-force nearest codeword on {bad}/200 samples "
            f"in d={d}. The optimal partition given a codebook is not a modelling "
            f"choice; it is forced by the objective.")

    x = rng.standard_normal(4000)

    # centroids() must be the exact per-cell conditional mean, for an ARBITRARY
    # partition -- not only for one produced by encode().
    a = rng.integers(0, 5, size=4000)
    got = np.asarray(centroids(x, a, 5), dtype=float).reshape(5, -1)
    want = np.array([[x[a == k].mean()] for k in range(5)])
    e = float(np.abs(got - want).max())
    assert e < 1e-10, (
        f"centroids() differs from the per-cell mean by {e:.2e}. Under squared error "
        f"the optimal codeword for a fixed cell is its conditional mean; there is no "
        f"choice to make here.")

    # An exact fixed point, constructed rather than iterated to: a symmetric sample
    # with the symmetric 2-point codebook satisfies BOTH conditions exactly.
    u = np.abs(rng.standard_normal(3000)) + 0.05
    xs = np.concatenate([u, -u])
    m = float(u.mean())
    g_fix = float(centroid_gap(xs, np.array([-m, m])))
    assert g_fix < 1e-10, (
        f"centroid_gap = {g_fix:.2e} on a codebook that satisfies both conditions "
        f"exactly by construction. It must be 0 there -- that is what makes it a "
        f"usable certificate in step 2.")

    c0 = rng.standard_normal(6) * 3
    g0 = float(centroid_gap(x, c0))
    assert g0 > 1e-3, f"centroid_gap = {g0:.2e} on a random codebook; it should be large"

    d0 = float(distortion(x, c0))
    c1 = np.asarray(centroids(x, np.asarray(encode(x, c0)), 6), dtype=float)
    d1 = float(distortion(x, c1))
    assert d1 <= d0 + 1e-12, (
        f"distortion rose under one round: {d0:.6f} -> {d1:.6f}. Each condition is a "
        f"minimisation holding the other fixed, so neither can increase the objective. "
        f"One of them is implemented wrongly.")

    # distortion is PER DIMENSION: a d-dimensional codebook of one codeword at the mean
    # must give (approximately) the per-dimension variance, not d times it.
    y = rng.standard_normal((5000, 4))
    dv = float(distortion(y, y.mean(0, keepdims=True)))
    assert 0.8 < dv < 1.25, (
        f"distortion of a 4-d unit-Gaussian sample against its own mean is {dv:.3f}. "
        f"Expected ~1.0 (per dimension). {dv:.3f} ~ 4.0 means you are summing over "
        f"dimensions without dividing -- that passes every 1-d test and makes E8 look "
        f"8x worse than Z in step 3.")

    # an empty cell must not produce nan
    c_far = np.array([-100.0, 0.0, 100.0])
    assert np.isfinite(centroid_gap(x, c_far)), (
        "centroid_gap returned nan/inf for a codebook with empty cells. Decide the "
        "empty-cell policy explicitly; it will not decide itself.")
    a = np.asarray(encode(x, c_far))
    cnew = np.asarray(centroids(x, a, 3), dtype=float)
    assert np.all(np.isfinite(cnew)), "centroids returned nan for an empty cell"


# ---------------------------------------------------------------------------
# Step 2 -- lloyd_max.py
# ---------------------------------------------------------------------------

def check_lloyd_max():
    from lloyd_max import lloyd_max, panter_dite_constant
    from objective import distortion, centroid_gap

    c = float(panter_dite_constant())
    assert abs(c - PANTER_DITE) < 1e-6, (
        f"panter_dite_constant() = {c:.8f}, expected {PANTER_DITE:.8f} = sqrt(3)*pi/2. "
        f"It is (1/12) * (integral of p^(1/3))^3 for the unit Gaussian.")

    rng = np.random.default_rng(20240301)
    train = rng.standard_normal(200_000)
    test = rng.standard_normal(200_000)

    prev = None
    for N, ref in sorted(LLOYD_TABLE.items()):
        cb, _ = lloyd_max(train, N)
        cb = np.asarray(cb, dtype=float)
        assert cb.shape == (N,), f"N={N}: codebook shape {cb.shape}, expected ({N},)"
        assert np.all(np.diff(cb) > 0), (
            f"N={N}: codebook is not strictly increasing: {cb}. Repeated or unsorted "
            f"codewords mean a collapsed or empty cell.")
        gap = float(centroid_gap(train, cb))
        assert gap < 1e-6, (
            f"N={N}: centroid_gap = {gap:.2e} at the returned codebook, so it is not a "
            f"fixed point. This is the iteration budget, not the algorithm -- reaching "
            f"a fixed point at N=256 takes ~28,000 alternations. Assert on the "
            f"certificate, never on the iteration count.")
        d = float(distortion(test, cb))
        rel = abs(d / ref - 1.0)
        assert rel < 0.06, (
            f"N={N}: held-out D = {d:.6f}, reference {ref:.6f} ({rel*100:.1f}% off). "
            f"N=2 has the closed form 1 - 2/pi = 0.36338023; if that one is wrong, "
            f"nothing else in the table will be right.")
        dn2 = d * N * N
        assert dn2 < PANTER_DITE, (
            f"N={N}: D*N^2 = {dn2:.4f} exceeds the asymptote {PANTER_DITE:.4f}. "
            f"Measured behaviour is monotone approach from below; above it means "
            f"unconverged.")
        if prev is not None:
            assert dn2 > prev, (
                f"N={N}: D*N^2 = {dn2:.4f} did not exceed the previous {prev:.4f}. "
                f"D*N^2 rises monotonically toward {PANTER_DITE:.4f}.")
        prev = dn2


# ---------------------------------------------------------------------------
# Step 3 -- lattices.py
# ---------------------------------------------------------------------------

def _g_of(decoder, x, d):
    q = np.asarray(decoder(x), dtype=float)
    return float(((x - q) ** 2).sum(-1).mean() / d)


def check_lattices():
    from lattices import z_nearest, d8_nearest, e8_nearest, normalised_second_moment

    rng = np.random.default_rng(1)
    x = rng.uniform(-4, 4, size=(400_000, 8))

    gz = _g_of(z_nearest, x, 8)
    assert abs(gz - 1 / 12) < 0.01 / 12, f"G(Z^8) = {gz:.6f}, expected 1/12 = 0.083333"

    q = np.asarray(e8_nearest(x), dtype=float)
    two = 2 * q
    is_int = np.all(np.abs(two - np.round(two)) < 1e-9, axis=1)
    assert np.all(is_int), "e8_nearest returned coordinates that are not in (1/2)Z"
    parity = np.round(two).astype(np.int64) % 2
    same = np.all(parity == parity[:, :1], axis=1)
    assert np.all(same), (
        f"{int((~same).sum())} outputs mix integer and half-integer coordinates. An E8 "
        f"point is either all-integer or all-half-odd-integer; never a mixture.")
    sums = q.sum(1)
    integral = np.abs(sums - np.round(sums)) < 1e-9
    assert np.all(integral), "coordinate sum is not an integer -- not an E8 point"
    allint = parity[:, 0] == 0
    even = np.abs(np.round(q[allint].sum(1)) % 2) < 1e-9
    assert np.all(even), (
        f"{int((~even).sum())} all-integer outputs have odd coordinate sum, so they are "
        f"in Z^8 but not in D8. This is the parity fix: when the rounded vector has odd "
        f"sum you must move ONE coordinate to its next-nearest integer.")

    ge8 = _g_of(e8_nearest, x, 8)
    gain = 10 * math.log10((1 / 12) / ge8)
    assert abs(ge8 - 0.0716821) < 0.02 * 0.0716821, (
        f"G(E8) = {ge8:.6f}, expected 0.071682 (gain {gain:+.3f} dB, expected +0.654). "
        + ("Near 0.0902 means you built D8 and never decoded the +1/2 coset: E8 is the "
           "UNION of the two, and you kept only the first. "
           if abs(ge8 - 0.0902) < 0.004 else
           "Above 1/12 means the parity fix moves the WRONG coordinate: it must be the "
           "one whose rounding error was largest in absolute value. "
           if ge8 > 1 / 12 else ""))
    assert abs(gain - 0.654) < 0.05, f"space-filling gain {gain:+.4f} dB, expected +0.654"

    gd8 = _g_of(d8_nearest, x, 8)
    assert gd8 > ge8, (
        f"G(D8)={gd8:.6f} is not worse than G(E8)={ge8:.6f}; the two decoders are "
        f"probably the same function.")

    g = float(normalised_second_moment(z_nearest, 8, 200_000, np.random.default_rng(2)))
    assert abs(g - 1 / 12) < 0.02 / 12, (
        f"normalised_second_moment(z_nearest, ...) = {g:.6f}, expected 1/12. Your "
        f"sampling region is biased or the det normalisation is wrong.")


# ---------------------------------------------------------------------------
# Step 4 -- incoherence.py
# ---------------------------------------------------------------------------

def check_incoherence():
    from incoherence import hadamard, fast_hadamard, incoherence, random_hadamard_transform

    for n in (64, 256, 1024):
        H = np.asarray(hadamard(n), dtype=float)
        assert H.shape == (n, n), f"hadamard({n}) has shape {H.shape}"
        assert set(np.unique(H)) <= {-1.0, 1.0}, "hadamard() must return +-1 entries"
        err = float(np.abs(H @ H.T / n - np.eye(n)).max())
        assert err < 1e-12, (
            f"n={n}: ||H H^T / n - I||_max = {err:.2e}. The normalisation is 1/sqrt(n) "
            f"per side. mu is scale-invariant so it will NOT catch this -- step 5 will, "
            f"silently, as a loss ratio that drifts to 1 for no visible reason.")
        v = np.random.default_rng(n).standard_normal(n)
        assert float(np.abs(np.asarray(fast_hadamard(v)) - H @ v).max()) < 1e-10, \
            f"n={n}: fast_hadamard disagrees with the dense product"

    rng = np.random.default_rng(1)
    ones = np.ones((16, 16)); ones[::2, ::3] = -1
    m = float(incoherence(ones))
    assert abs(m - 1.0) < 1e-9, (
        f"mu of an all-+-1 matrix is {m:.6f}, must be exactly 1.0 -- it is the minimum. "
        f"Check the sqrt(m*n) factor.")

    # The transform itself must be orthogonal, not merely built from an H that can be
    # normalised after the fact. Testing hadamard() alone cannot see this.
    n = 256
    Wr = np.random.default_rng(3).standard_normal((n, n))
    ratio = float(np.linalg.norm(np.asarray(random_hadamard_transform(Wr, np.random.default_rng(5))))
                  / np.linalg.norm(Wr))
    assert abs(ratio - 1.0) < 1e-9, (
        f"random_hadamard_transform changed ||W||_F by a factor {ratio:.6g}; it must be "
        f"exactly 1. Dropping the 1/sqrt(n) gives {n:.0f}. mu is scale-invariant and "
        f"will never tell you -- step 5 will, as a loss ratio that drifts to 1.")

    # Independence of the two sign vectors. W = I is a fixed point of the SHARED-sign
    # transform: (H S) I (H S)^T = H S S H^T = I, since S^2 = I. So mu stays at
    # sqrt(n) = 16 exactly. With independent signs it drops to ~3.
    eye = np.eye(256)
    mu_eye = float(incoherence(np.asarray(random_hadamard_transform(eye, np.random.default_rng(11)))))
    assert mu_eye < 6.0, (
        f"mu(RHT(I)) = {mu_eye:.3f}, expected ~3. Exactly sqrt(256) = 16.0 means you "
        f"are using the SAME sign vector on both sides: S^2 = I, so the identity is a "
        f"fixed point of your transform and no randomisation happens at all.")

    mus = []
    for n in (64, 256, 1024):
        W = rng.standard_normal((n, n)); W[3, :] *= 60.0
        before = float(incoherence(W))
        after = float(incoherence(np.asarray(random_hadamard_transform(W, np.random.default_rng(7)))))
        assert after >= 1.0 - 1e-9, f"mu = {after} < 1 is impossible"
        assert after < 8.0, f"n={n}: mu after RHT = {after:.2f}, expected < 8 (measured ~3-5)"
        assert after < before / 5, f"n={n}: mu only fell {before:.1f} -> {after:.2f}"
        mus.append(after)
    assert mus[-1] < 2.0 * mus[0], (
        f"mu after RHT grew {mus[0]:.2f} -> {mus[-1]:.2f} over a 16x increase in n. It "
        f"should grow like sqrt(log(mn)), i.e. barely.")

    # The test that separates randomness from decoration.
    n = 256
    Hu = np.asarray(hadamard(n), dtype=float)
    W = np.outer(Hu[7], Hu[11]) / n
    mu0 = float(incoherence(W))
    assert abs(mu0 - 1.0) < 1e-9, f"mu of the adversarial matrix is {mu0}, expected 1.0"
    Hn = Hu / math.sqrt(n)
    mu_det = float(incoherence(Hn @ W @ Hn.T))
    assert mu_det > 0.9 * n, (
        f"deterministic Hadamard gave mu = {mu_det:.1f}, expected ~{n}. On this input "
        f"H concentrates all the Frobenius mass into ONE entry.")
    mu_rand = float(incoherence(np.asarray(random_hadamard_transform(W, np.random.default_rng(9)))))
    assert mu_rand < 15.0, (
        f"RHT gave mu = {mu_rand:.1f} on the adversarial input, expected < 15 "
        f"(measured 7.2). If this is also ~{n}, your sign flips are not being applied "
        f"-- or the same sign vector is used on both sides.")


# ---------------------------------------------------------------------------
# Step 5 -- gptq.py
# ---------------------------------------------------------------------------

def _synthetic_layer(rng, d_out=64, d_in=128, n=512, rho=0.9):
    A = rng.standard_normal((d_in, d_in))
    C = A @ A.T / d_in + rho * np.ones((d_in, d_in))
    L = np.linalg.cholesky(C + 1e-6 * np.eye(d_in))
    return rng.standard_normal((d_out, d_in)), L @ rng.standard_normal((d_in, n))


def check_gptq():
    from gptq import hessian, quantize_rtn, optimal_update, gptq, proxy_loss

    rng = np.random.default_rng(7)

    # 1. the exact-update identity: no tolerance to hide in
    for d, d_out, q in ((8, 3, 0), (12, 5, 4), (12, 2, 10)):
        A = rng.standard_normal((d, d))
        H = A @ A.T + d * np.eye(d)
        err = rng.standard_normal(d_out)
        got = np.asarray(optimal_update(H, q, err), dtype=float)
        want = -np.outer(err, np.linalg.solve(H[q + 1:, q + 1:], H[q + 1:, q]))
        assert got.shape == want.shape, (
            f"optimal_update returned shape {got.shape}, expected {want.shape} "
            f"(d_out, d_in - q - 1)")
        e = float(np.abs(got - want).max())
        if np.abs(got + want).max() < 1e-9:
            raise AssertionError(
                "optimal_update has the right magnitude and the WRONG SIGN. err is "
                "defined as W_hat[:,q] - W[:,q] and the return value is ADDED to "
                "W[:, q+1:]. This is the single most common bug in this file and it "
                "makes GPTQ worse than round-to-nearest.")
        assert e < 1e-9, (
            f"optimal_update differs from the brute-force block least-squares "
            f"minimiser by {e:.2e} (q={q}). A correct implementation agrees to 0.0. "
            f"Nothing below this line means anything until it does.")

    # 2. GPTQ vs round-to-nearest
    W, X = _synthetic_layer(rng)
    H = np.asarray(hessian(X, 0.01), dtype=float)
    assert H.shape == (X.shape[0], X.shape[0]), f"hessian shape {H.shape}"
    assert float(np.abs(H - H.T).max()) < 1e-8, "hessian is not symmetric"
    ev = np.linalg.eigvalsh(H)
    assert ev.min() > 0, (
        f"hessian has a non-positive eigenvalue ({ev.min():.3e}); damping is not being "
        f"applied. With n_samples < d_in it is singular by construction.")

    for bits in (2, 3, 4):
        Qr = np.asarray(quantize_rtn(W, bits), dtype=float)
        Qg = np.asarray(gptq(W, H, bits), dtype=float)
        assert Qr.shape == W.shape and Qg.shape == W.shape
        lr = float(proxy_loss(W, Qr, X))
        lg = float(proxy_loss(W, Qg, X))
        assert lr > 0, "round-to-nearest proxy loss is zero -- the grid is not coarse"
        ratio = lg / lr
        assert ratio < 0.6, (
            f"{bits} bits: proxy_loss(GPTQ)/proxy_loss(RTN) = {ratio:.3f}, expected "
            f"~0.25-0.34 and required < 0.6. "
            + ("Above 1.0 means the update is applied with the wrong sign. "
               if ratio > 1.0 else
               "Near 1.0 means the update is not applied at all, or the damping is so "
               "large that GPTQ has degenerated into RTN -- sweep the damping over "
               "three orders of magnitude and look at where the curve starts rising."))
        nvals = len(np.unique(np.round(Qg / np.abs(Qg).max() * 1e6)))
        assert nvals <= 2 ** bits * W.shape[0] + 2, (
            f"{bits} bits: the GPTQ output takes {nvals} distinct values; it must lie "
            f"on the grid.")


# ---------------------------------------------------------------------------
# Step 6 -- trellis.py
# ---------------------------------------------------------------------------

def check_trellis():
    from trellis import state_codebook, viterbi, brute_force

    V = np.asarray(state_codebook(6), dtype=float)
    assert V.shape == (64,), f"state_codebook(6) has shape {V.shape}, expected (64,)"
    assert abs(float(V.mean())) < 0.5 and 0.5 < float(V.std()) < 2.0, (
        f"state_codebook(6): mean {V.mean():.3f}, std {V.std():.3f}. The codeword "
        f"values should look like a standard Gaussian sample.")
    assert np.allclose(V, np.asarray(state_codebook(6), dtype=float)), \
        "state_codebook is not deterministic"

    rng = np.random.default_rng(5)
    for trial in range(20):
        x = rng.standard_normal(8)
        dv = float(viterbi(x, 3, 1)[1])
        db = float(brute_force(x, 3, 1))
        assert abs(dv - db) < 1e-12, (
            f"trial {trial}: viterbi D = {dv:.10f}, brute force D = {db:.10f}. The "
            f"recursion is exact, so these must agree to machine precision. A small "
            f"gap is the traceback: you stored the best PREDECESSOR where you needed "
            f"the best INPUT BITS, or the other way round.")

    x = np.random.default_rng(3).standard_normal(3000)
    prev, first, last = None, None, None
    for L in (2, 4, 6, 8, 10):
        d = float(viterbi(x, L, 2)[1])
        assert d > 0.0625, (
            f"L={L}: D = {d:.5f} is below the Gaussian rate-distortion function at "
            f"R=2, D(R) = 2^-4 = 0.0625. Nothing may beat it. This is the one "
            f"assertion here that is a theorem rather than a measurement, so a "
            f"violation is a bug with certainty.")
        if prev is not None:
            assert d < prev, f"L={L}: D = {d:.5f} did not improve on {prev:.5f}"
        prev = d
        if L == 2: first = d
        last = d
    assert first > SCALAR_D_AT_R2, (
        f"L=2 gave D = {first:.5f}, which already beats scalar Lloyd-Max "
        f"({SCALAR_D_AT_R2:.5f}). With a 4-state register and a random codebook it "
        f"should be clearly WORSE. That is expected -- do not fix it.")
    assert last < 0.80 * SCALAR_D_AT_R2, (
        f"L=10 gave D = {last:.5f}, needed < {0.80*SCALAR_D_AT_R2:.5f}. Measured "
        f"across three hash families: 0.0735-0.0832.")


# ---------------------------------------------------------------------------

CHECKS = [
    ("objective.py", "the objective and its two optimality conditions", check_objective),
    ("lloyd_max.py", "scalar Lloyd-Max, and the asymptote it never reaches", check_lloyd_max),
    ("lattices.py", "E8 nearest-point decoding, space-filling gain", check_lattices),
    ("incoherence.py", "random Hadamard transform, incoherence mu", check_incoherence),
    ("gptq.py", "GPTQ: Hessian, exact update, error feedback", check_gptq),
    ("trellis.py", "trellis-coded quantization, Viterbi encoding", check_trellis),
]


def run(fn):
    try:
        fn()
        return PASS, ""
    except NotImplementedError:
        return TODO, ""
    except AssertionError as e:
        return FAIL, str(e)
    except ImportError as e:
        return TODO, str(e)
    except Exception:
        return ERROR, traceback.format_exc().strip().splitlines()[-1]


def main(argv):
    run_all = "--all" in argv
    nums = [int(a) for a in argv if a.isdigit()]
    lo = nums[0] if nums else 1
    hi = nums[1] if len(nums) > 1 else (nums[0] if nums else len(CHECKS))

    print(f"\n{BOLD}compression{RESET}  --  {len(CHECKS)} graded checks\n")
    counts = {PASS: 0, FAIL: 0, TODO: 0, ERROR: 0}
    for i, (name, desc, fn) in enumerate(CHECKS, 1):
        if not (lo <= i <= hi):
            continue
        status, msg = run(fn)
        counts[status] += 1
        mark = {PASS: f"{GREEN}OK{RESET}", FAIL: f"{RED}XX{RESET}",
                TODO: f"{GREY}--{RESET}", ERROR: f"{YELLOW}!!{RESET}"}[status]
        print(f"  {mark} {i}. {name:<18} {GREY}{desc}{RESET}")
        if msg:
            for line in msg.splitlines():
                print(f"        {line}")
        if status in (TODO, FAIL, ERROR) and not run_all:
            if status == TODO:
                print(f"\n  {BOLD}Next:{RESET} {name} -- {desc}\n")
            else:
                print(f"\n  Fix {name}, then re-run. (--all to keep going.)\n")
            return 1
    print()
    print(f"  {counts[PASS]} passed, {counts[FAIL]} failed, {counts[TODO]} to write, "
          f"{counts[ERROR]} errored\n")
    return 0 if counts[FAIL] == 0 and counts[ERROR] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
