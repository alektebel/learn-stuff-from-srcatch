# Week 14 · Nov 23–Nov 29, 2026

> **Finish SGLang, then training at scale and world models.**
> The pivot from serving to training.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `sgl-lang/` | 24 of 87 | full only | [`../week-13/sgl-lang/`](../week-13/sgl-lang/) |
| [`distributed-training/`](distributed-training/) | 10 | full only | here |
| `world-models/` | 46 of 106 | full only | [`../week-15/world-models/`](../week-15/world-models/) |

**80 block hours**, plus the daily Lean slot (~8.4 h) = 88 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `ml-inference` 42 h | 42 |
| **spine** | `cuda-from-scratch` 21 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `../week-13/sgl-lang/`.
2. `distributed-training` is short: data parallel, gradient accumulation, and the communication pattern underneath. Ten hours.
3. Then `../week-15/world-models/`, starting with the VAE. Paper 1 only this week — Dreamer versions 1 to 3 are week 15.

## Done means

- SGLang complete.
- A data-parallel training loop whose loss curve matches single-GPU training.
- A VAE reconstructing observations well enough that the latents are worth modelling.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 14   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 13](../week-13/) · [Roadmap](../ROADMAP.md) · [Week 15 →](../week-15/)
