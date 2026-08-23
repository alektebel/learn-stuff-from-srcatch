# Week 11 · Nov 2–Nov 8, 2026

> **Quantisation, batching, and what a server actually does.**
> The bridge week between a kernel and a service. No new directory: the project lives in week 10.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `ml-inference/` | 79 of 137 | core | [`../week-10/ml-inference/`](../week-10/ml-inference/) |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `cuda-from-scratch` 45 h | 45 |
| **spine** | `autograd` 11 h, `llm-from-scratch` 16 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. `../week-10/ml-inference/` is the whole week.
2. Dynamic batching is the concept to spend real time on. It is the same throughput-versus-latency trade you will meet again in vLLM, and meeting it twice from different directions is why both directories are here.
3. Re-read your own `../week-08/context-caching/kv_cache.py` when you reach the serving material. You have already built the core of it.
4. Plot a latency-versus-throughput curve for your own server at several batch sizes and find the knee.

## Done means

- A quantised model running, with a measured latency AND accuracy delta versus fp32.
- A latency-versus-throughput curve for your own server, and you can point at the knee and explain what is saturating.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 11            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 10](../week-10/) · [Roadmap](../ROADMAP.md) · [Week 12 →](../week-12/)
