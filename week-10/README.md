# Week 10 · Oct 26–Nov 1, 2026

> **CUDA, most of the week.**
> One directory, sixty-three hours, minimal context switching. Memory hierarchy, coalescing, shared memory, occupancy, and then a matmul you tune yourself.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `cuda-from-scratch/` | 63 of 122 | core · spine | [`../week-09/cuda-from-scratch/`](../week-09/cuda-from-scratch/) |
| [`ml-inference/`](ml-inference/) | 16 of 137 | core | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `cuda-from-scratch` 45 h | 45 |
| **spine** | `aws-from-scratch` 8 h, `autograd` 19 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Keep working the directory's order. Naive matmul, then tiled, then register blocked, measuring GFLOP/s at each step.
2. Delete each `TODO` comment as you satisfy it — the C and CUDA bars in `progress.py` only move if you do.
3. Open `../week-10/ml-inference/` on Saturday and start with the quantisation material, because everything else in that directory is downstream of understanding what int8 actually costs you.
4. Benchmark fp32 against fp16 against int8 on the same model and record accuracy alongside latency. Latency without the accuracy column is a meaningless number.

## Done means

- A naive, a tiled and a register-blocked matmul, with measured GFLOP/s for each.
- You can say what fraction of peak bandwidth each of your kernels reaches, and why the gap is what it is.
- You can explain warp divergence in terms of the mask stack you built in `week-03/compiler-and-vgpu/`.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 10            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 9](../week-09/) · [Roadmap](../ROADMAP.md) · [Week 11 →](../week-11/)
