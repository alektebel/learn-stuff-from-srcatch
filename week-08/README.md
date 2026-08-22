# Week 8 · Oct 12–Oct 18, 2026

> **Finish CUDA, start the inference stack.**
> No new directory this week. The two projects live in week 7 and week 9.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `cuda-from-scratch/` | 19 of 122 | core · spine | [`../week-07/cuda-from-scratch/`](../week-07/cuda-from-scratch/) |
| `ml-inference/` | 61 of 137 | core | [`../week-09/ml-inference/`](../week-09/ml-inference/) |

**80 block hours**, plus the daily Lean slot (~8.4 h) = 88 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `deploy-and-debug` 1 h, `context-caching` 28 h, `contextcite` 12 h | 41 |
| **spine** | `dynamo-paper` 13 h, `aws-from-scratch` 8 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `../week-07/cuda-from-scratch/` in the first two days — the neural network kernels at the end.
2. Then `../week-09/ml-inference/`: start with the quantisation material, because everything else in that directory is downstream of understanding what int8 costs you.
3. Benchmark fp32 against fp16 against int8 on the same model and record accuracy alongside latency. Latency without the accuracy column is a meaningless number.

## Done means

- `cuda-from-scratch` complete, with its measurement table written up.
- A quantised model running, with a measured latency AND accuracy delta versus fp32.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 8    # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 7](../week-07/) · [Roadmap](../ROADMAP.md) · [Week 9 →](../week-09/)
