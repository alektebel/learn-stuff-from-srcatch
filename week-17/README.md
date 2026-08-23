# Week 17 · Dec 14–Dec 20, 2026

> **Haskell, then a real GPU.**
> Sunday is for nvidia-smi, so start the driver before Monday.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `haskell-projects/` | 61 | full | [`../week-04/haskell-projects/`](../week-04/haskell-projects/) |
| `cuda-from-scratch/` | 122 | core · spine | [`../week-09/cuda-from-scratch/`](../week-09/cuda-from-scratch/) |
| `ml-inference/` | 137 | core | [`../week-10/ml-inference/`](../week-10/ml-inference/) |

On the narrower tracks this same week is:

core · spine: CUDA. core: ml-inference too. full: Haskell first.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Full: JSONParser, Calculator, BuildTool, WebScraper. Delete each `-- TODO`.
2. Vector add, then tiled matmul. Profile every kernel.
3. A latency harness. fp32 vs fp16 vs int8, latency AND accuracy.

## Done means

- `nvidia-smi` works. A kernel ran.
- The accuracy you paid for the bytes you saved.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 17            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 16](../week-16/) · [Roadmap](../ROADMAP.md) · [Week 18 →](../week-18/)
