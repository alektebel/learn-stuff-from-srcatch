# Solutions

Complete, working implementations of every template in the parent directory. They pass
all 19 checks.

Use them the way you would use the back of a textbook: when you are stuck on one
function, after you have finished one and want to compare, or to read the commentary in
the docstrings — the solutions say *why* each choice was made, where the templates say
what to build.

Reading these first will cost you most of the value of the directory. The bugs you are
about to write (a softmax that overflows, a `dS` missing its row-coupling term, an outer
loop that re-reads what it should have kept on chip, a reverse scan that runs forwards)
are the material.

```bash
cd solutions
python3 baseline.py       # standard attention, gradients, cost table
python3 clustered.py      # error vs cluster count; the bound holding
python3 linear.py         # associativity, the RNN, gradient checks, timings
python3 flash.py          # standard vs flash HBM traffic, exactness, causal skipping
python3 flash2.py         # FA-1 vs FA-2 counters, split-K merge
python3 benchmark.py      # everything, with fitted exponents
```

`common.py` and `io_model.py` are copies of the given files from the parent directory,
so this folder runs standalone.

To check the solutions against the grader:

```bash
cp ../check.py . && python3 check.py --all && rm check.py
```

## Layout

| File | Steps | Contents |
|---|---|---|
| `baseline.py` | 1-2 | stable softmax, attention, backward, analytic cost |
| `clustered.py` | 3-6 | LSH, Hamming K-means, clustered attention, the error bound, top-k improvement |
| `linear.py` | 7-10 | feature map, associativity, causal prefix sums, the RNN, linear-memory backward |
| `flash.py` | 11-15 | standard attention in the IO model, online softmax, tiled forward, recomputed backward, causal skipping |
| `flash2.py` | 16-18 | loop swap and deferred division, split-K with exact merge, two-pass backward |
| `benchmark.py` | 19 | flop/memory/byte counts and measured scaling |
