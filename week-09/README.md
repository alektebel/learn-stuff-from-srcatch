# Week 9 · Oct 19–Oct 25, 2026

> **Quantisation, batching, and what a server actually does.**
> The bridge week between a kernel and a service.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`ml-inference/`](ml-inference/) | 76 of 137 | core | here |
| `tensorrt-inference/` | 5 of 109 | full only | [`../week-10/tensorrt-inference/`](../week-10/tensorrt-inference/) |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `contextcite` 1 h, `cuda-from-scratch` 41 h | 42 |
| **spine** | `aws-from-scratch` 21 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. `ml-inference` is the whole week apart from a first look at TensorRT.
2. Dynamic batching is the concept to spend real time on: it is the same throughput-versus-latency trade you will meet again in vLLM, and meeting it twice from different directions is why both directories are here.
3. Open `../week-10/tensorrt-inference/` on Sunday and get its toolchain installed. TensorRT setup can eat a day; do not let it eat a Monday.

## Done means

- `ml-inference` complete.
- A latency-versus-throughput curve for your own server at several batch sizes, and you can point at the knee and explain it.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 9    # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 8](../week-08/) · [Roadmap](../ROADMAP.md) · [Week 10 →](../week-10/)
