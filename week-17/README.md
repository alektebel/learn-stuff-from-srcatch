# Week 17 · Dec 14–Dec 20, 2026

> **Generation and detection, as a pair.**
> Deliberately in the same week: build the forger and the detector together, because each one is the honest test of the other.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`deepfake-creation/`](deepfake-creation/) | 37 of 48 | full only | here |
| [`deepfake-detection/`](deepfake-detection/) | 33 | full only | here |
| `quantitative-trading/` | 10 of 52 | full only | [`../week-18/quantitative-trading/`](../week-18/quantitative-trading/) |

**80 block hours**, plus the daily Lean slot (~8.4 h) = 88 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `vllm-engine` 42 h | 42 |
| **spine** | `cuda-from-scratch` 21 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. `deepfake-creation` first — faceswap and reenactment.
2. `deepfake-detection` immediately after, and **test your detector on your own generator's output**, not only on a public dataset. A detector that only works on someone else's artefacts has learned the dataset, not the problem.
3. Start `../week-18/quantitative-trading/` on Sunday.
4. One line worth writing down before you start: this pairing is the reason to build both. Detection research that never faces a generator it did not expect is how detectors ship broken.

## Done means

- A working faceswap pipeline.
- A detector with reported accuracy on a public set AND on your own output. Expect the second number to be much worse. That gap is the finding.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 17   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 16](../week-16/) · [Roadmap](../ROADMAP.md) · [Week 18 →](../week-18/)
