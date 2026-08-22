# Week 13 · Nov 16–Nov 22, 2026

> **Finish vLLM, start SGLang.**
> Structured generation, and the radix cache you already built once.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `vllm-engine/` | 18 of 156 | core | [`../week-12/vllm-engine/`](../week-12/vllm-engine/) |
| [`sgl-lang/`](sgl-lang/) | 63 of 87 | full only | here |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `ml-inference` 42 h | 42 |
| **spine** | `contextcite` 2 h, `cuda-from-scratch` 18 h | 20 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `../week-12/vllm-engine/` in two days.
2. `sgl-lang` after it. RadixAttention will be familiar — you wrote `radix_cache.py` in week 6 — so spend the time on the parts that are not: the frontend language, constrained decoding, and the grammar-compiled state machine.
3. Constrained decoding is the genuinely new mechanism. Do not skim it.

## Done means

- vLLM complete.
- Generation constrained to a JSON schema, with the mask applied at the logits and no invalid token ever sampled.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 13   # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 12](../week-12/) · [Roadmap](../ROADMAP.md) · [Week 14 →](../week-14/)
