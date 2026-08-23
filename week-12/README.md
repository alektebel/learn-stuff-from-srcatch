# Week 12 · Nov 9–Nov 15, 2026

> **Finish the inference stack, then meet the vendor's version.**
> TensorRT is a vendor's answer to the two months you just spent building your own. The value is entirely in the comparison, not the API.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `ml-inference/` | 42 of 137 | core | [`../week-10/ml-inference/`](../week-10/ml-inference/) |
| [`tensorrt-inference/`](tensorrt-inference/) | 37 of 109 | full only | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `cuda-from-scratch` 21 h, `ml-inference` 24 h | 45 |
| **spine** | `llm-from-scratch` 29 h | 29 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `../week-10/ml-inference/` in the first half of the week.
2. `tensorrt-inference` after it: engine building, precision calibration, profiling.
3. For every optimisation TensorRT applies, write one line on what it is doing and whether you could have done it by hand in week 10. Layer fusion, kernel auto-tuning, precision calibration — these are not magic, and the point of having written CUDA first is that you can now see through them.
4. Get its toolchain installed before you need it. TensorRT setup can eat a day.

## Done means

- `ml-inference` complete.
- A TensorRT engine built and running on your own model.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 12            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 11](../week-11/) · [Roadmap](../ROADMAP.md) · [Week 13 →](../week-13/)
