# Week 18 · Dec 21–Dec 27, 2026

> **Vendor engines, world models, the tail.**
> Re-read your paged_kv.py before you open vLLM. Reserve the last day.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `tensorrt-inference/` | 109 | full | [`../week-12/tensorrt-inference/`](../week-12/tensorrt-inference/) |
| `vllm-engine/` | 156 | core | [`../week-13/vllm-engine/`](../week-13/vllm-engine/) |
| `world-models/` | 106 | full | [`../week-16/world-models/`](../week-16/world-models/) |
| `diffusion-models/` | 91 | full | [`../week-17/diffusion-models/`](../week-17/diffusion-models/) |

On the narrower tracks this same week is:

core: vLLM if you have anything left. full: the rest, then five small directories. Last day: every checker.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. TensorRT against the guide, then vLLM Phase 1. Fork still costs zero.
2. World-models VAE (you measured this in autograd). Diffusion forward process first, analytically.
3. Five small wins. Last day: every checker. Re-read the journal from 23 August.

## Done means

- What you can re-derive today that you could not on 23 August.
- The journal is closed.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 18            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 17](../week-17/) · [Roadmap](../ROADMAP.md)
