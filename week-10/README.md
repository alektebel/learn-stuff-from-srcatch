# Week 10 · Oct 26–Nov 1, 2026

> **TensorRT.**
> A vendor's answer to the week you just spent building your own. The value is in the comparison, not the API.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`tensorrt-inference/`](tensorrt-inference/) | 81 of 109 | full only | here |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `cuda-from-scratch` 42 h | 42 |
| **spine** | `aws-from-scratch` 14 h, `deploy-and-debug` 7 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Engine building, precision calibration, and profiling.
2. For every optimisation TensorRT applies, write one line on what it is doing and whether you could have done it by hand in week 7. Layer fusion, kernel auto-tuning, precision calibration — these are not magic and the point of having written CUDA first is that you can now see through them.
3. Compare against your week-9 server on the same model and the same hardware.

## Done means

- A TensorRT engine beating your own server, with the speedup attributed to specific optimisations rather than to 'TensorRT is faster'.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 10   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 9](../week-09/) · [Roadmap](../ROADMAP.md) · [Week 11 →](../week-11/)
