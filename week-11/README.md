# Week 11 · Nov 2–Nov 8, 2026

> **Finish TensorRT, start vLLM.**
> No new directory this week. Both projects live in week 10 and week 12.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `tensorrt-inference/` | 23 of 109 | full only | [`../week-10/tensorrt-inference/`](../week-10/tensorrt-inference/) |
| `vllm-engine/` | 57 of 156 | core | [`../week-12/vllm-engine/`](../week-12/vllm-engine/) |

**80 block hours**, plus the daily Lean slot (~8.4 h) = 88 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `cuda-from-scratch` 39 h, `ml-inference` 2 h | 41 |
| **spine** | `deploy-and-debug` 3 h, `context-caching` 18 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `../week-10/tensorrt-inference/`.
2. Then `../week-12/vllm-engine/`. Re-read your own `../week-06/context-caching/paged_kv_cache.py` before you start — you have already built the core idea at small scale, and the fastest way through this directory is to notice that.
3. PagedAttention first, continuous batching second.

## Done means

- TensorRT complete.
- A block allocator with a per-sequence block table, and forking a sequence costs zero additional blocks.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 11   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 10](../week-10/) · [Roadmap](../ROADMAP.md) · [Week 12 →](../week-12/)
