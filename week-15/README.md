# Week 15 · Nov 30–Dec 6, 2026

> **Finish vLLM, then training at scale.**
> The pivot from serving to training. Ten weeks of making inference fast, and now the other half of the problem.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `vllm-engine/` | 70 of 156 | core | [`../week-13/vllm-engine/`](../week-13/vllm-engine/) |
| [`distributed-training/`](distributed-training/) | 10 | full only | here |

**80 block hours**, plus the daily Lean slot (~8.4 h) = 88 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `ml-inference` 24 h, `vllm-engine` 21 h | 45 |
| **spine** | `cuda-from-scratch` 27 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `../week-13/vllm-engine/`.
2. `distributed-training` is short: data parallel, gradient accumulation, and the communication pattern underneath. You already wrote gradient accumulation in `../week-07/autograd/` — this is the same idea with a network in the middle.
3. Start `../week-16/world-models/` on Sunday with the VAE. You built one in `autograd/generative.py`; this is the same reparameterisation trick against real observations.

## Done means

- vLLM complete, with its throughput numbers written up.
- A data-parallel training loop whose loss curve matches single-GPU training.
- You can explain why `all_reduce` of gradients and averaging of weights give the same answer for SGD and different answers for Adam.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 15            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 14](../week-14/) · [Roadmap](../ROADMAP.md) · [Week 16 →](../week-16/)
