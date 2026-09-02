# Computational lower bounds: four real ones, and why none of them binds

Five steps. Steps 1, 3, 4 are the standard library only; step 2 needs `numpy`.

| # | File | You build | Acceptance test |
|---|---|---|---|
| 1 | `decision_tree.py` | The information-theoretic sorting bound, and an exhaustive search that meets it | `ceil(log2 n!)` comparisons suffice for n ≤ 5 and not fewer |
| 2 | `tensor_rank.py` | Strassen's algorithm derived from a rank-7 tensor decomposition | Reconstruction error `0.0`; rank-6 best fit plateaus at `1.000` |
| 3 | `baur_strassen.py` | All partial derivatives at a constant factor of the evaluation cost | Op ratio flat in n from 4 to 1024 |
| 4 | `pebble_game.py` | A cache simulator, and the `Ω(n³/√M)` memory-traffic bound | `traffic·√M/n³` constant for tiled, not for naive |
| 5 | `the_gap.md` | One page of prose: why steps 1–4 bound nothing you care about | You wrote it |

## Why this order

Each step weakens the model, and each weaker model buys a weaker bound. That progression
*is* the subject:

```
1  decision trees          strong bound, absurdly weak model  (only comparisons allowed)
2  arithmetic circuits      exact bound for ONE tiny tensor; the general case is open
3  arithmetic circuits      Omega(n log n) -- still the state of the art since 1983
4  red-blue pebble game     tight and USEFUL, but bounds one algorithm, not one function
5  ...                      and for the model you actually care about: nothing
```

Step 5 is not a coda. It is the point, and it is the only step with no code in it.

    DESIGN DECISION -- prove the bound before implementing the algorithm.
    Each step asks for the lower bound first and the matching construction second. The
    reverse order is more comfortable and teaches you nothing: once you have Strassen's
    seven products in front of you, "why not six" stops being a question you can feel.
    Cost: step 2 asks you to spend an hour failing to find a rank-6 decomposition before
    telling you the residual plateaus at exactly 1.0. That hour is the content.

## What this track does NOT contain

No proof of any barrier result. Razborov–Rudich and Williams are in `../READING.md`
because step 5 asks you to say why the techniques in steps 1–4 cannot be pushed, and you
cannot answer that from these four exercises alone. You have to read something.

## Running

```bash
python3 check.py          # all steps
python3 check.py 2        # step 2 only
python3 check.py --all    # do not stop at the first gap
```
