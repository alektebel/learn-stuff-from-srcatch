# Efficient Attention From Scratch

Implement the three main answers to "attention is quadratic", in numpy, one step at a
time — and **measure** each claim rather than reading it.

| # | Method | Paper |
|---|---|---|
| 1 | **Clustered attention** | Vyas, Katharopoulos, Fleuret, *Fast Transformers with Clustered Attention*, NeurIPS 2020 — [arXiv:2007.04825](https://arxiv.org/abs/2007.04825) |
| 2 | **Linear attention** | Katharopoulos, Vyas, Pappas, Fleuret, *Transformers are RNNs*, ICML 2020 — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 3 | **FlashAttention 1 & 2** | Dao et al., *FlashAttention*, NeurIPS 2022 — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · Dao, *FlashAttention-2*, 2023 — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) |

**[THEORY.md](THEORY.md)** is the conceptual companion: the core idea of each method in
plain language and in mathematics, what problem it solves, how it compares with standard
scaled-dot-product attention, its approximation guarantee (or its exactness), and its
asymptotic complexity. Read one section, implement it, come back.

## Why these three, in this order

They are not three variations on one idea. They fail differently, and the order makes
that visible:

* **Clustered attention** approximates the same function and comes with a bound you can
  compute without knowing the answer — and which turns out to be numerically vacuous
  unless your queries genuinely cluster. It cannot do causal masking at all.
* **Linear attention** does not approximate softmax attention; it *replaces* it with a
  different attention that happens to factorise. So there is no bound, and the right
  question changes from "how close" to "does it train as well". Causal masking becomes a
  prefix sum, and decoding becomes an RNN with constant state.
* **FlashAttention** changes nothing about the function at all. It is exact, bit for bit,
  and every gain comes from moving fewer bytes. It is the only one of the three that is
  strictly better than what it replaces, which is why it is the one that actually shipped.

By the end you should be able to say why only one of them can be exact, why only one of
them has an error bound, and why the one with neither is the one in your framework.

---

## How to use this directory

Top-level files are **templates**: every function has a docstring explaining what to
build and why, then `raise NotImplementedError`. You fill them in. `solutions/` holds
complete working versions for when you are stuck or want to compare afterwards.

```bash
cd efficient-attention
pip install numpy            # the only dependency
python3 check.py             # what to build next
# ... implement the functions the checker points at ...
python3 check.py             # re-run; it stops at the first thing not yet done
```

`check.py` runs **19 graded checks** against *your* code. It never imports `solutions/`,
and it grades against a small reference attention written inside the checker itself, so
a mistake in your baseline cannot quietly excuse a mistake later.

```
  ✓  1. baseline.py    stable softmax and attention
  ✓  2. baseline.py    backward pass and cost model
  ·  3. clustered.py   LSH bits and Hamming K-means
      not implemented yet — clustered.py:62 in lsh_bits()

  2/19 passing, 1 to write

  Next: step 3 — LSH bits and Hamming K-means (clustered.py)
```

| Command | Does |
|---|---|
| `python3 check.py` | Run in order, stop at the first unimplemented step |
| `python3 check.py 13` | Run only step 13, while you iterate on it |
| `python3 check.py 13 15` | Run steps 13 through 15 |
| `python3 check.py --all` | Run everything, skipping nothing |
| `python3 <file>.py` | Run that file's own demo, once it is implemented |

Grey `·` means not written yet. Red `✗` means written but wrong.

**Given, not exercises:** `common.py` (toy data, error metrics, numerical gradients) and
`io_model.py` (the memory hierarchy you will be charged by).

---

## The 19 steps

### `baseline.py` — steps 1-2 · *small*
Standard scaled dot-product attention, its backward pass, and an analytic cost model.
Everything later is measured against this, and two details here come back as load-bearing
parts of FlashAttention: the numerical stability of the softmax, and the row-coupling
term `dS = P * (dP - rowsum(P*dP))` — together with the identity
`rowsum(P*dP) = rowsum(dO*O)`, which is what lets flash-backward avoid a second `N x N`
intermediate.

### `clustered.py` — steps 3-6 · *the longest file*
LSH bits, K-means in Hamming space, attention computed once per cluster, and the
improved variant that is exact on each cluster's top-`k` keys.

Step 5 is the unusual one: you implement the **error bound itself**, and the checker
asserts it is never violated across seeds and cluster spreads. A bound you have watched
hold — and watched become vacuous when clusters are loose — is worth more than a bound
you have read.

### `linear.py` — steps 7-10 · *steps 7-9 short, step 10 hard*
The feature map, the associativity rearrangement, causal masking as a prefix sum, and
the RNN form that gives the paper its title. Step 10 is the backward pass in linear
memory: a forward scan and a reverse scan, with the ones-column trick that folds the
denominator into the numerator. Do it on paper first.

### `flash.py` — steps 11-15 · *the centre of the directory*
Step 11 writes standard attention **in the IO model** so its `Theta(N^2)` traffic is
measured, not quoted. Step 12 is the online softmax — twenty lines, and the whole idea.
Steps 13-15 assemble the tiled forward pass, the backward pass by recomputation, and
causal block skipping.

### `flash2.py` — steps 16-18
The loop swap and the deferred division (fewer elementwise ops, independent row blocks),
splitting the key axis with an exact merge of partial softmaxes (this is flash-decoding,
ring attention, and paged KV caches, all at once), and a backward pass that never
accumulates a gradient through HBM.

### `benchmark.py` — step 19 · *small*
Flops, memory and measured HBM bytes for everything, with fitted exponents. Half an
hour, and the most useful half-hour here: a technique you cannot cost is a technique you
cannot choose.

---

## Design decisions

**numpy, no torch, no CUDA.** You write every matmul, softmax and gradient yourself.
Nothing is hidden behind an operator that already implements the thing being taught.

**Single head, no batch, 2-D tensors throughout.** Multi-head attention is this run `H`
times on `H` slices. Carrying the extra dimensions would triple the index bookkeeping in
every file and teach nothing.

**float64 by default.** FlashAttention is exact in exact arithmetic, so the checker can
demand agreement at `1e-12` instead of the `1e-6` that float32 rounding would force.
The difference between "the algorithm is right" and "the algorithm is roughly right" is
worth 4 bytes per element.

**A simulated memory hierarchy instead of a GPU** — `io_model.py`. It charges for every
HBM transfer and raises `SRAMOverflow` if your blocks do not fit the budget. Two things
follow that a GPU-less directory could not otherwise deliver:

* the claim "FlashAttention moves fewer bytes" becomes a number **you** produce;
* block sizes are *derived* from the budget rather than chosen, which is where the
  paper's `B_c = ceil(M/4d)` comes from.

What it deliberately does not model: warps, coalescing, latency hiding, tensor cores.
It counts bytes crossing one boundary. That single number is what the FlashAttention
analysis is about — and the parts it cannot show (FA-2's warp partitioning, its
occupancy win) are named as such in THEORY.md rather than faked.

**The cost of each method is stated, not just its benefit.** Clustered attention cannot
do causal masking and is worthless on data that does not cluster; linear attention has a
rank limit and no error bound at all; FlashAttention-2's byte count is barely better than
FlashAttention-1's, because its gains are elsewhere. A directory that only lists
advantages has taught you marketing.

---

## What you will have measured by the end

All of these come out of the demos in this directory, not from the papers:

```
clustered attention, 256 queries in 8 real groups, 32 clusters
    clustered 0.084 relative error   +top-8 0.015   +top-32 0.0026
    on iid gaussian queries instead:  0.99  (i.e. useless — measure your data)
    the step-5 bound: vacuous at cluster spread 0.3, 15x better than trivial at 0.01

linear attention
    N=8192: softmax 975 ms, linear 37 ms, 26x
    peak intermediate: 8 KB at every N, against 512 MB for the score matrix at N=8192
    RNN state after 64 tokens: 272 numbers, versus a KV cache's 2048 and growing

FlashAttention, d=32, 64 KB SRAM
    HBM traffic at N=512: standard 14.3 MB, flash-1 4.2 MB, flash-2 3.5 MB
    largest single HBM tensor: 65536 elements standard, 8192 flash
    error vs plain numpy attention: 4e-16   (exact, not approximate)
    causal: 24 tiles computed, 16 skipped outright

FlashAttention-2 against FlashAttention-1
    elementwise accumulator ops: 122,880 -> 65,536
    dQ writes to HBM in backward: 144 -> 12
    HBM traffic: 1.1 MB -> 1.0 MB   (barely moved: its win is not in bytes)
```

---

## Where to go next

* **Sparse + IO-aware.** Block-sparse FlashAttention, and the long line after it —
  combining a sparsity pattern with tiling is harder than either alone.
* **FlashAttention-3.** Hopper-specific: asynchrony (TMA, warp specialisation) and FP8.
  Same algorithm again, arranged for different hardware — the third time the same lesson
  appears in this directory.
* **Modern linear RNNs.** GLA, RetNet, Mamba, DeltaNet. Every one is an answer to the
  saturating state you build in step 9.
* **Multi-query and grouped-query attention, and paged KV caches.** Orthogonal to all of
  this: they shrink the cache rather than the computation. `../context-caching/` and
  `../vllm-engine/` in this repository go there.
