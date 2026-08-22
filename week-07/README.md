# Week 7 · Oct 5–Oct 11, 2026

> **CUDA, all week.**
> One directory, eighty-one hours, no context switching. Memory hierarchy, coalescing, shared memory, occupancy, and then a matmul you tune yourself.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`cuda-from-scratch/`](cuda-from-scratch/) | 81 of 122 | core · spine | here |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `aws-from-scratch` 33 h, `deploy-and-debug` 9 h | 42 |
| **spine** | `compiler-and-vgpu` 13 h, `dynamo-paper` 8 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Work strictly in the directory's own order. Every kernel builds on the previous one's understanding of memory.
2. **Profile every kernel you write.** `ncu` or `nsight-compute`. A CUDA kernel you have not profiled is a kernel you do not understand, and this is the week to make that a habit rather than a chore.
3. Keep a table as you go: kernel, GB/s achieved, GB/s theoretical, occupancy. That table is the week's real deliverable.
4. Delete each `TODO` comment as you satisfy it — the C bars in `progress.py` only move if you do.

## Done means

- A naive matmul, a tiled matmul and a register-blocked matmul, with measured GFLOP/s for each.
- You can say what fraction of peak bandwidth each of your kernels reaches, and why the gap is what it is.
- You can explain warp divergence in terms of the mask stack you built in `week-03/compiler-and-vgpu/`.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 7    # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 6](../week-06/) · [Roadmap](../ROADMAP.md) · [Week 8 →](../week-08/)
