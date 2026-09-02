# Compression: from rate–distortion to a layer-wise quantizer

Six steps. Each is an afternoon except step 5, which is two. Requires `numpy`.

| # | File | You build | Acceptance test |
|---|---|---|---|
| 1 | `objective.py` | The objective, and the two conditions any optimal quantizer satisfies | Conditions detect their own violation |
| 2 | `lloyd_max.py` | Scalar Lloyd–Max on a Gaussian | `D·N²` matches a table to 6% |
| 3 | `lattices.py` | Nearest-point decoders for Z⁸ and E8 | `G(E8) = 0.0717`, gain `0.654 dB` |
| 4 | `incoherence.py` | Random Hadamard transform, incoherence μ | μ: `1 → 256` deterministic, `1 → ~7` random |
| 5 | `gptq.py` | GPTQ on one linear layer | Update identity exact; loss ratio `≤ 0.6` vs RTN |
| 6 | `trellis.py` | Trellis quantizer, Viterbi encoding | `D(L=10) < 0.75 · D(scalar)` |

## The ladder

Every step exists because the previous one hit a wall you can name:

```
1  the objective                       ...but nothing tells you where to put the levels
2  Lloyd-Max, scalar                   ...but the cells are intervals; in d dims they are polytopes
3  lattice VQ, d=8                     ...but a codebook of size 2^(Rd) will not fit in cache
4  incoherence processing              ...(the enabling step: makes the weights look Gaussian)
5  GPTQ: quantize against a Hessian    ...but it still rounds each weight independently
6  trellis: codebook computed, not stored
```

Step 4 sits inside the ladder rather than after it because incoherence processing is
what licenses steps 3, 5 and 6 to assume the source is i.i.d. Gaussian. Without it they
are all quantizing something with heavy tails and directional structure, and every
distortion number you computed in steps 1–3 stops applying.

## Do not skip step 1

Steps 2–6 are algorithms. Step 1 is the objective those algorithms are approximate
solutions to. If you cannot state the two optimality conditions without looking, you
will not be able to tell whether step 5 is behaving correctly, because GPTQ's whole
claim is that it satisfies one of them exactly and the other not at all.

## What "from first principles" means here

None of these files hands you an algorithm to implement. Each hands you a *quantity to
minimise* and asks what the minimiser must satisfy. Lloyd's algorithm is not stated
anywhere in this directory; it is what you get when you alternate between the two
conditions in step 1, and you should notice that yourself. If you find yourself
reaching for a reference to remember "the algorithm", you have skipped the derivation.

## Running

```bash
python3 check.py          # all steps, stops at the first unwritten one
python3 check.py 4        # step 4 only
python3 check.py 2 4      # steps 2 through 4
python3 check.py --all    # do not stop at the first gap
```
