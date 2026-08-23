# Week 16 · Dec 7–Dec 13, 2026

> **World models.**
> The VAE, the recurrent model, the controller, and then Dreamer.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`world-models/`](world-models/) | 78 of 106 | full only | here |

**78 block hours**, plus the daily Lean slot (~8.4 h) = 86 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `vllm-engine` 45 h | 45 |
| **spine** | `cuda-from-scratch` 27 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. `world-models` all week. VAE first, then the RNN, then the controller.
2. The measurement that matters: how far a random `N(0, I)` draw is from the nearest latent the model has actually seen. You measured exactly this in `../week-07/autograd/generative.py` — a plain autoencoder scores 2.22 and a VAE 1.49, and that gap is why sampling from an autoencoder produces noise.
3. Then Dreamer v1, and start v2 if there is time.
4. V1 to v3 is a lesson in incremental research: each version fixes a specific failure of the last. Write down what each one fixed. That note is worth more than the implementation.

## Done means

- A VAE reconstructing observations well enough that the latents are worth modelling.
- An agent trained partly in imagination that transfers to the real environment.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 16            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 15](../week-15/) · [Roadmap](../ROADMAP.md) · [Week 17 →](../week-17/)
