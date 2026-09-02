"""
Progress checker for the lower-bounds templates.

    python3 check.py           # every check, stop at the first unimplemented step
    python3 check.py 3         # step 3 only
    python3 check.py --all     # run everything, do not stop at the first gap

A check that raises NotImplementedError reports TODO, not FAIL. There is no solutions/
directory. Steps 1, 3 and 4 use the standard library; step 2 needs numpy.
"""

import math
import pathlib
import shutil
import sys
import traceback

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).parent
shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
sys.path.insert(0, str(HERE))

PASS, FAIL, TODO, ERROR = "PASS", "FAIL", "TODO", "ERROR"
GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[36m", "\033[90m", "\033[1m", "\033[0m")

INFO_BOUND = [0, 1, 3, 5, 7, 10, 13, 16, 19, 22, 26, 29]     # ceil(log2 n!), n = 1..12
MIN_COMP = [0, 1, 3, 5, 7]                                    # n = 1..5


# ---------------------------------------------------------------------------
# Step 1 -- decision_tree.py
# ---------------------------------------------------------------------------

def check_decision_tree():
    import itertools
    from decision_tree import (counting_sort_key, worst_case_comparisons,
                               information_bound, min_comparisons)

    for n, want in enumerate(INFO_BOUND, 1):
        got = information_bound(n)
        assert got == want, f"information_bound({n}) = {got}, expected {want}"

    for n in range(1, 8):
        for perm in itertools.permutations(range(n)):
            counter = {"n": 0}
            out = list(counting_sort_key(list(perm), counter))
            assert out == sorted(perm), (
                f"the sort is wrong on {perm}: returned {out}. Fix correctness before "
                f"counting anything.")

    for n in range(1, 8):
        w = worst_case_comparisons(counting_sort_key, n)
        b = information_bound(n)
        assert w >= b, (
            f"n={n}: your sort's worst case is {w} comparisons, below the information "
            f"bound {b}. That is impossible for a comparison sort, so the counter is "
            f"not counting every comparison -- most likely the ones inside a merge "
            f"step's tail copy.")

    for n, want in enumerate(MIN_COMP, 1):
        got = min_comparisons(n)
        assert got == want, (
            f"min_comparisons({n}) = {got}, expected {want}. "
            + (f"A value above {want} usually means the search is non-adaptive: a "
               f"decision tree may test different pairs in its two subtrees. "
               if got > want else
               f"A value below {want} contradicts the information bound, so the search "
               f"is not requiring every permutation to reach its own leaf. "))


# ---------------------------------------------------------------------------
# Step 2 -- tensor_rank.py
# ---------------------------------------------------------------------------

def check_tensor_rank():
    import numpy as np
    from tensor_rank import matmul_tensor, strassen_factors, multiply_from_factors, als_fit

    T = np.asarray(matmul_tensor(2), dtype=float)
    assert T.shape == (4, 4, 4), f"matmul_tensor(2) has shape {T.shape}, expected (4,4,4)"
    assert abs(T.sum() - 8.0) < 1e-12, (
        f"the tensor has {T.sum():.0f} non-zero entries, expected 8 (= n^3 for n=2). "
        f"Each of the n^3 scalar multiplications in the definition contributes one.")

    U, V, W = (np.asarray(a, dtype=float) for a in strassen_factors())
    assert U.shape == V.shape == W.shape == (7, 4), (
        f"factors have shapes {U.shape}, {V.shape}, {W.shape}; expected (7,4) each")
    R = np.einsum('ra,rb,rc->abc', U, V, W)
    e = float(np.abs(R - T).max())
    assert e == 0.0, (
        f"reconstruction error {e:.3e}, must be exactly 0.0 -- the entries are small "
        f"integers. If it is ~2.0, the third factor is indexed (i,k) where the tensor "
        f"wants (k,i): your product comes out transposed.")

    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.standard_normal((2, 2)); B = rng.standard_normal((2, 2))
        C, nmul = multiply_from_factors(A, B, (U, V, W))
        C = np.asarray(C, dtype=float)
        err = float(np.abs(C - A @ B).max())
        errT = float(np.abs(C - (A @ B).T).max())
        assert err < 1e-12, (
            f"A @ B is wrong by {err:.2e}. "
            + ("The result is the exact TRANSPOSE of A @ B -- this is the third-index "
               "convention, and symmetric test matrices would have hidden it. "
               if errT < 1e-12 else ""))
        assert nmul == 7, f"reported {nmul} multiplications, expected exactly 7"

    norm = float(np.linalg.norm(T))
    assert abs(norm - 2 * math.sqrt(2)) < 1e-12, f"||T|| = {norm}, expected 2*sqrt(2)"
    r7 = float(als_fit(T, 7, np.random.default_rng(5)))
    assert r7 < 1e-6, (
        f"ALS could not reach rank 7 (best residual {r7:.4f}). With rank > 4 on a 4x4x4 "
        f"tensor the least-squares subproblems are rank-deficient: use lstsq, not solve.")
    r6 = float(als_fit(T, 6, np.random.default_rng(5)))
    assert r6 > 0.5, (
        f"ALS reached residual {r6:.4f} at rank 6. Measured plateau: 1.000. A residual "
        f"near zero would contradict rank(<2,2,2>) = 7 -- check that you are fitting "
        f"with 6 components and not silently 7.")


# ---------------------------------------------------------------------------
# Step 3 -- baur_strassen.py
# ---------------------------------------------------------------------------

def check_baur_strassen():
    import random
    from baur_strassen import evaluate, gradient, random_circuit

    rng = random.Random(0)
    ratios = []
    for n in (4, 16, 64, 256, 1024):
        nodes = random_circuit(n, 4 * n, random.Random(n))
        x = [rng.gauss(0, 1) for _ in range(n)]
        vals, fwd = evaluate(nodes, x)
        grad, tot = gradient(nodes, x)
        assert len(grad) == n, f"n={n}: gradient has {len(grad)} entries, expected {n}"
        assert fwd > 0, "evaluate reported zero operations"
        ratios.append(tot / fwd)
        assert tot / fwd < 8.0, (
            f"n={n}: op ratio {tot/fwd:.2f}. It must be a constant; anything that grows "
            f"means the backward pass is being run once per input variable, which is "
            f"forward mode wearing a disguise.")

    assert ratios[-1] < 1.25 * ratios[0], (
        f"op ratio grew from {ratios[0]:.2f} at n=4 to {ratios[-1]:.2f} at n=1024, over "
        f"a 256x increase in the number of inputs. The theorem says it does not grow at "
        f"all. Measured with one convention: 4.00, 3.84, 4.05, 3.98, 4.00.")

    # correctness on EVERY coordinate, not a spot check: overwriting adjoints instead of
    # accumulating them is wrong only where a node feeds more than one consumer.
    n = 12
    nodes = random_circuit(n, 40, random.Random(99))
    x = [rng.gauss(0, 1) for _ in range(n)]
    grad, _ = gradient(nodes, x)
    h = 1e-6
    for i in range(n):
        xp = list(x); xp[i] += h
        xm = list(x); xm[i] -= h
        fd = (evaluate(nodes, xp)[0][-1] - evaluate(nodes, xm)[0][-1]) / (2 * h)
        scale = max(1.0, abs(fd))
        assert abs(fd - grad[i]) / scale < 1e-4, (
            f"coordinate {i}: gradient {grad[i]:.8g}, finite difference {fd:.8g}. If "
            f"most coordinates are right and a few are not, you are ASSIGNING adjoints "
            f"where you must ACCUMULATE them -- correct on a tree, wrong on a DAG.")

    # a deliberately shared node: one variable feeding many gates
    shared = [('var', 0), ('var', 1)] + [('*', 0, 1)] + [('+', 2, 1) for _ in range(30)]
    shared = shared[:3] + [('+', 2, i % 3) for i in range(3, 33)]
    xs = [1.3, -0.7]
    g, tot = gradient(shared, xs)
    _, fwd = evaluate(shared, xs)
    assert tot / fwd < 8.0, (
        f"op ratio {tot/fwd:.2f} on a circuit where one node feeds 30 consumers. The "
        f"backward cost is proportional to EDGES; if this ratio blows up, your "
        f"representation is not sharing subexpressions.")


# ---------------------------------------------------------------------------
# Step 4 -- pebble_game.py
# ---------------------------------------------------------------------------

def check_pebble_game():
    from pebble_game import Cache, naive_traffic, tiled_traffic, best_tile

    # Two traces, chosen so that FIFO errs in a DIFFERENT DIRECTION on each: on the
    # first it reports too few misses, on the second too many. One trace alone would
    # let a wrong policy look merely miscalibrated.
    for trace, want, fifo in ((("a", "b", "a", "c", "b"), 4, 3),
                              (("a", "b", "a", "c", "a", "b"), 4, 5)):
        c = Cache(2)
        for k in trace:
            c.touch(k)
        assert c.miss == want, (
            f"Cache(2) reported {c.miss} misses on {','.join(trace)}; LRU gives {want}. "
            + (f"{fifo} is what FIFO gives: a HIT must move the key to the "
               f"most-recently-used end, not leave it where it is. "
               if c.miss == fifo else ""))

    for M in (48, 108, 192, 300):
        b = best_tile(M)
        assert isinstance(b, int) and b >= 1, f"best_tile({M}) = {b!r}"
        assert 3 * b * b <= M, (
            f"best_tile({M}) = {b}: three {b}x{b} tiles need {3*b*b} words, more than "
            f"M = {M}. They must be simultaneously resident.")
        assert 3 * (b + 1) ** 2 > M, f"best_tile({M}) = {b} is smaller than it needs to be"

    qs_t, qs_n = [], []
    for n in (32, 48, 64):
        for M in (48, 108, 192, 300):
            b = best_tile(M)
            t = tiled_traffic(n, M, b)
            nv = naive_traffic(n, M)
            qs_t.append(t * math.sqrt(M) / n ** 3)
            qs_n.append(nv * math.sqrt(M) / n ** 3)

    lo, hi = min(qs_t), max(qs_t)
    assert 4.0 <= lo and hi <= 8.0, (
        f"tiled traffic * sqrt(M) / n^3 ranged [{lo:.2f}, {hi:.2f}], expected inside "
        f"[4, 8] (measured 5.09 to 5.42).")
    assert hi / lo < 1.5, (
        f"tiled q ranged [{lo:.2f}, {hi:.2f}], a factor {hi/lo:.2f}. It must be nearly "
        f"constant -- that constancy IS the n^3/sqrt(M) scaling, and it is the entire "
        f"measurement.")

    lo_n, hi_n = min(qs_n), max(qs_n)
    assert hi_n / lo_n > 1.5, (
        f"naive q ranged [{lo_n:.2f}, {hi_n:.2f}], a factor {hi_n/lo_n:.2f}. It should "
        f"NOT be constant: naive traffic is ~n^3 regardless of M, so q must drift like "
        f"sqrt(M) (measured 10.8 to 20.9). If yours is flat, the tiled and naive "
        f"schedules are the same function.")


# ---------------------------------------------------------------------------
# Step 5 -- the_gap.md
# ---------------------------------------------------------------------------

def check_the_gap():
    p = HERE / "the_gap.md"
    if not p.exists():
        raise NotImplementedError
    text = p.read_text()
    if "<!-- UNWRITTEN -->" in text:
        raise NotImplementedError
    sections = [s for s in text.split("\n## ")[1:]]
    assert len(sections) >= 6, f"found {len(sections)} sections, expected 6"
    for s in sections:
        head = s.splitlines()[0].strip()
        assert "_your answer_" not in s, (
            f"section '{head}' still has the placeholder. This is the only step here "
            f"with no code, and it is the one the other four exist to set up.")
        # count only YOUR text: the blockquote lines, not the prompt above them
        answer = " ".join(ln.lstrip(">").strip()
                          for ln in s.splitlines() if ln.lstrip().startswith(">"))
        words = len(answer.split())
        assert words > 60, (
            f"section '{head}': your answer is {words} words. Sixty is not a quality "
            f"bar, it is a floor -- a real answer to any of these does not fit in less. "
            f"(Only the blockquote lines are counted, not the prompt.)")


# ---------------------------------------------------------------------------

CHECKS = [
    ("decision_tree.py", "the comparison bound, and the slack it hides", check_decision_tree),
    ("tensor_rank.py", "Strassen from tensor rank; the rank-6 plateau", check_tensor_rank),
    ("baur_strassen.py", "all partials at a constant factor of the value", check_baur_strassen),
    ("pebble_game.py", "red-blue pebble game, tiled matmul traffic", check_pebble_game),
    ("the_gap.md", "one page: why none of the four binds (prose)", check_the_gap),
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

    print(f"\n{BOLD}lower-bounds{RESET}  --  {len(CHECKS)} graded checks\n")
    counts = {PASS: 0, FAIL: 0, TODO: 0, ERROR: 0}
    for i, (name, desc, fn) in enumerate(CHECKS, 1):
        if not (lo <= i <= hi):
            continue
        status, msg = run(fn)
        counts[status] += 1
        mark = {PASS: f"{GREEN}OK{RESET}", FAIL: f"{RED}XX{RESET}",
                TODO: f"{GREY}--{RESET}", ERROR: f"{YELLOW}!!{RESET}"}[status]
        print(f"  {mark} {i}. {name:<20} {GREY}{desc}{RESET}")
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
