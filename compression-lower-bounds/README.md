# Compression and computational lower bounds

Two tracks. **They do not share a technique, a tool, or a proof method, and this
directory keeps them apart on purpose.**

| Track | Question | Ends at |
|---|---|---|
| [`compression/`](compression/) | How few bits per weight before the model breaks? | A layer-wise EXL3-shaped quantizer you understand line by line |
| [`lower-bounds/`](lower-bounds/) | What can be proved *impossible*, and for which machine? | Four real lower bounds, and one page on why none of them binds the thing you care about |

## Why they are in one directory but not one course

The honest connection is not technical. Rate–distortion theory gives you *tight*
limits — Shannon's `D(R) = σ²2^(-2R)` is achieved in the limit, and every step of the
compression track is a measurable fraction of that limit. Computational complexity
gives you *almost no* limits: the best lower bound for an explicit polynomial is still
Baur–Strassen's `Ω(n log n)` from 1983, and the barriers (relativisation, natural
proofs, algebrisation) explain why.

So the pairing is a contrast, not a synthesis: **one field knows its own limits and one
does not.** If you interleave them you will spend the compression track hunting for a
complexity-theoretic bound on quantization, and there isn't one. Finish one, then start
the other.

## Ground rules for this directory

Both tracks follow [`PHILOSOPHY.md`](../PHILOSOPHY.md), with one deliberate break:

> **DESIGN DECISION — no `solutions/` directory.**
> Every other directory in this repo ships reference implementations. This one does not.
> The material is short enough that reading a solution costs you the entire exercise:
> Lloyd's algorithm is nine lines and you will never re-derive it once you have seen
> them. The cost is real — when you are stuck you are stuck, and `check.py` tells you
> *which invariant broke*, never *what to write*.
> **Chosen:** no solutions, because the failure modes here are the content.

Every template states, before you write a line:

- **the acceptance test** — a number, an assertion, or a curve shape, that your code
  either produces or does not;
- **the predicted failure mode** — what will go wrong, written down in advance so that
  when it happens you learn something instead of debugging blind;
- **one prose question** — answer it in writing before moving on. If you cannot, you
  implemented the step without understanding it, which is the failure this repo exists
  to prevent.

## Running the checkers

```bash
cd compression   && python3 check.py       # 6 graded checks
cd lower-bounds  && python3 check.py       # 5 graded checks
```

`check.py` runs against **your** code. A step you have not written reports `TODO`, not
`FAIL`. `python3 check.py 3` runs one step; `python3 check.py --all` does not stop at
the first gap.

`compression/` needs `numpy`. `lower-bounds/` needs `numpy` only for step 2; the rest is
the standard library.

## Checkpoints

After compression steps 1–3, and again after 4–6, close every file and write the
explanation from memory: what the quantity is, why the previous step made this one
possible, and what number you measured. A gap in the written version is a gap, not a
lapse of memory — the checker cannot see it, so you have to.

## Reference numbers

Every number below was computed while writing this directory, not recalled. They are
what a correct implementation produces, and `check.py` tests against them.

**Scalar Lloyd–Max on a unit Gaussian** (converged to a fixed point, distortion measured
on held-out samples):

| N | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | → ∞ |
|---|---|---|---|---|---|---|---|---|---|
| `D` | .363380 | .117482 | .034548 | .009501 | .002505 | .000644 | .000163 | .0000412 | — |
| `D·N²` | 1.4535 | 1.8797 | 2.2111 | 2.4323 | 2.5648 | 2.6388 | 2.6785 | 2.7000 | **2.7207** |

`D·N²` rises monotonically to the Panter–Dite constant `√3·π/2 = 2.72070` and **does not
reach it at any N you will use.** Verifying a 4-bit quantizer "against Zador's bound"
therefore tests nothing: the asymptote is 12% away at N=16. `N=2` has the closed form
`1 − 2/π = 0.3633802276`, which is the only exact anchor in the table.

**Other measured quantities**

| Quantity | Value | Where |
|---|---|---|
| `G(Z)` normalised second moment | `1/12 = 0.083333` | step 3 |
| `G(E8)` | `0.0716821` → **0.654 dB** space-filling gain | step 3 |
| μ of a weight matrix with a planted outlier row, n=1024 | `97.6` → `4.72` after RHT | step 4 |
| μ of an adversarial matrix under **deterministic** Hadamard | `1.00` → **`256.0`** | step 4 |
| GPTQ proxy loss ÷ round-to-nearest, 4 bits | `0.28` | step 5 |
| Trellis distortion at R=2, register length L=2 / 6 / 10 | `0.243` / `0.111` / `0.079` | step 6 |
| Best rank-5 / 6 / 7 fit to the ⟨2,2,2⟩ matmul tensor | `1.414` / `1.000` / `0.000` | LB step 2 |
| Baur–Strassen op ratio, n = 4 … 1024 | `4.00`, flat | LB step 3 |
| Tiled matmul traffic · √M / n³ | `5.1 – 5.9`, flat | LB step 4 |

See [`READING.md`](READING.md) for the papers, with verification status marked per entry.
