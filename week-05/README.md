# Week 5 · Sep 21–Sep 27, 2026

> **A naive server, then the machinery that stops it falling apart.**
> Make it work. Watch two requests fail as `single_flight`. Then batch, cache, schedule, page.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `inference-from-scratch/` | steps 1–6 | core | [`../week-10/inference-from-scratch/`](../week-10/inference-from-scratch/) |

On the narrower tracks this same week is:

core: steps 1–6. spine: skip; you already have the cache.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Naive server. Two overlapping requests → `reason='single_flight'`.
2. Continuous batching. Measure TTFT, TPOT, throughput.
3. KV runtime: cached decode of 256 tokens is bandwidth-bound.
4. Scheduler, then paged KV. Fork costs zero blocks until a write.
5. Do not open `vllm-engine/`.

## Done means

- Checks 1–6 green.
- You watched the naive server fall apart and named the failure.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 5            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 4](../week-04/) · [Roadmap](../ROADMAP.md) · [Week 6 →](../week-06/)
