# Week 12 · Nov 9–Nov 15, 2026

> **PagedAttention and continuous batching.**
> The most valuable single directory in the repo if you want to work on inference. Eighty-one hours, one project.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`vllm-engine/`](vllm-engine/) | 81 of 156 | core | here |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `ml-inference` 42 h | 42 |
| **spine** | `context-caching` 10 h, `contextcite` 11 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Continuous batching is the week's centre: prefill and decode interleaved across many sequences in one forward pass.
2. Then the scheduler and preemption. Preemption is where paged memory earns its complexity, and where the design decision becomes obvious in retrospect.
3. Hold the invariant from week 6 the whole way: **a cache that changes the output is not a cache, it is a bug.** Assert it in a test on day one.

## Done means

- Concurrent requests served with bit-identical output to the unbatched path.
- A measured throughput gain over static batching, and you can say which part of it came from paging and which from the scheduler.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 12   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 11](../week-11/) · [Roadmap](../ROADMAP.md) · [Week 13 →](../week-13/)
