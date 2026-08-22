# Week 16 · Dec 7–Dec 13, 2026

> **Diffusion.**
> Forward process, reverse process, sampling, and the conditioning that makes it useful.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`diffusion-models/`](diffusion-models/) | 70 of 91 | full only | here |
| `deepfake-creation/` | 11 of 48 | full only | [`../week-17/deepfake-creation/`](../week-17/deepfake-creation/) |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `vllm-engine` 42 h | 42 |
| **spine** | `cuda-from-scratch` 21 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Forward noising first, and verify it analytically before you train anything.
2. Then the reverse process and DDPM sampling, then DDIM, then classifier-free guidance.
3. Sample at several guidance scales and look at the diversity collapse. It is the clearest quality-versus-diversity trade in the repo.
4. Start `../week-17/deepfake-creation/` on Sunday.

## Done means

- Samples from your own trained model.
- A guidance-scale sweep with the collapse visible, and you can say why it happens.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 16   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 15](../week-15/) · [Roadmap](../ROADMAP.md) · [Week 17 →](../week-17/)
