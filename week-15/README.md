# Week 15 · Nov 30–Dec 6, 2026

> **World models.**
> The recurrent model, the controller, and then Dreamer.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`world-models/`](world-models/) | 60 of 106 | full only | here |
| `diffusion-models/` | 21 of 91 | full only | [`../week-16/diffusion-models/`](../week-16/diffusion-models/) |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `ml-inference` 10 h, `vllm-engine` 31 h | 41 |
| **spine** | `cuda-from-scratch` 21 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `world-models`: the RNN, the controller, then Dreamer v1 through v3 in order.
2. V1 to v3 is a lesson in incremental research — each version fixes a specific failure of the last. Write down what each one fixed. That note is worth more than the implementation.
3. Start `../week-16/diffusion-models/` on Sunday with the forward process, which is arithmetic and needs no training run.

## Done means

- An agent trained in imagination that transfers to the real environment.
- Three sentences: what v2 fixed in v1, and what v3 fixed in v2.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 15   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 14](../week-14/) · [Roadmap](../ROADMAP.md) · [Week 16 →](../week-16/)
