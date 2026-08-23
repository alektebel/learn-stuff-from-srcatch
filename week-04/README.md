# Week 4 · Sep 14–Sep 20, 2026

> **The cache in front of attention, then the first serving step.**
> Bit-identical cached/uncached output, then the GPU path for one token.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `context-caching/` | 28 | core · spine | [`../week-08/context-caching/`](../week-08/context-caching/) |
| `inference-from-scratch/` | start of 60 | core | [`../week-10/inference-from-scratch/`](../week-10/inference-from-scratch/) |

On the narrower tracks this same week is:

core: context-caching 28 h, then inference step 1. spine: context-caching only.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. `kv_cache.py` until cached and uncached attention agree to **zero**.
2. Then prefix / radix / paged / serving_demo.
3. If 16/16 lands: `inference_path.py` — name every kernel for one prefill and one decode.

## Done means

- context-caching 16/16. `serving_demo.py` is bit-identical with the cache on and off.
- You can name the prefill kernels and the cached-decode kernels.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 4            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 3](../week-03/) · [Roadmap](../ROADMAP.md) · [Week 5 →](../week-05/)
