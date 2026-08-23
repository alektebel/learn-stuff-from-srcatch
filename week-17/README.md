# Week 17 · Dec 14–Dec 20, 2026

> **Finish world models, start diffusion.**
> Two ways of learning a generative model, back to back — one that compresses into a latent and one that learns to reverse noise.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `world-models/` | 28 of 106 | full only | [`../week-16/world-models/`](../week-16/world-models/) |
| [`diffusion-models/`](diffusion-models/) | 51 of 91 | full only | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `vllm-engine` 45 h | 45 |
| **spine** | `cuda-from-scratch` 27 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `../week-16/world-models/`: Dreamer v1 through v3, in order.
2. Three sentences before you move on: what v2 fixed in v1, what v3 fixed in v2.
3. `diffusion-models` from Thursday. The forward noising process first, and verify it analytically before you train anything — it is arithmetic and it needs no training run.
4. Then the reverse process and DDPM sampling.

## Done means

- `world-models` complete.
- The closed-form `q(x_t | x_0)` verified against iterated single-step noising, to floating-point agreement.
- A first sample out of your own trained model, however bad it looks.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 17            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 16](../week-16/) · [Roadmap](../ROADMAP.md) · [Week 18 →](../week-18/)
