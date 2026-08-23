# Week 9 · Oct 19–Oct 25, 2026

> **Rays, then the machine built to trace a billion of them.**
> `ray-tracer` finishes on Tuesday and it is deliberately placed here: it is the purest embarrassingly-parallel workload in the repo, and you meet it the week before you learn the hardware designed for exactly that shape.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `ray-tracer/` | 20 of 35 | full only | [`../week-08/ray-tracer/`](../week-08/ray-tracer/) |
| [`cuda-from-scratch/`](cuda-from-scratch/) | 59 of 122 | core · spine | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `deploy-and-debug` 6 h, `context-caching` 28 h, `cuda-from-scratch` 11 h | 45 |
| **spine** | `aws-from-scratch` 26 h | 26 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `ray-tracer`: `bvh.py`, `material.py`, `render.py`.
2. **Before running the sample sweep, write down what you expect the noise to do when samples go from 16 to 64.** Then read the table. Four times the work for half the noise is the entire cost structure of rendering, and it is the same 1/sqrt(N) you will meet again in `diffusion-models`.
3. `cuda-from-scratch` from Wednesday, strictly in the directory's own order. Every kernel builds on the previous one's understanding of memory.
4. **Profile every kernel you write.** `ncu` or `nsight-compute`. A CUDA kernel you have not profiled is a kernel you do not understand, and this is the week to make that a habit rather than a chore.

## Done means

- `week-08/ray-tracer/` prints 6/6, and `render.ppm` opens in an image viewer.
- You can say why a BVH is 1.5x at four objects and 12x at a thousand — and why that makes 'we tried an acceleration structure and it did not help' a statement about the benchmark.
- A measurement table started: kernel, GB/s achieved, GB/s theoretical, occupancy. That table is the next two weeks' real deliverable.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 9            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 8](../week-08/) · [Roadmap](../ROADMAP.md) · [Week 10 →](../week-10/)
