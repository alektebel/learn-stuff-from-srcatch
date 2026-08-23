# Week 13 · Nov 16–Nov 22, 2026

> **TensorRT, then open vLLM.**
> The last week before the largest single directory in the repo.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `tensorrt-inference/` | 72 of 109 | full only | [`../week-12/tensorrt-inference/`](../week-12/tensorrt-inference/) |
| [`vllm-engine/`](vllm-engine/) | 7 of 156 | core | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `ml-inference` 45 h | 45 |
| **spine** | `deploy-and-debug` 10 h, `context-caching` 15 h | 25 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `../week-12/tensorrt-inference/`, and benchmark it against your own week-11 server on the same model and the same hardware.
2. The deliverable is not 'TensorRT is faster'. It is a speedup attributed to specific optimisations you can name.
3. Open `vllm-engine` on Sunday. Re-read `../week-08/context-caching/paged_kv_cache.py` before you start — you have already built the core idea at small scale, and the fastest way through this directory is to notice that.

## Done means

- A TensorRT engine beating your own server, with the speedup attributed to specific optimisations rather than to the brand.
- PagedAttention read and understood well enough to sketch the block table on paper before you implement it.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 13            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 12](../week-12/) · [Roadmap](../ROADMAP.md) · [Week 14 →](../week-14/)
