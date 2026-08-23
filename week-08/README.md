# Week 8 · Oct 12–Oct 18, 2026

> **A transformer, then the cache in front of it.**
> You build attention, then immediately build the thing production systems put in front of attention — which is the correct order and almost never the order people learn it in.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `llm-from-scratch/` | 13 of 45 | core · spine | [`../week-07/llm-from-scratch/`](../week-07/llm-from-scratch/) |
| [`deploy-and-debug/`](deploy-and-debug/) | 10 | core · spine | here |
| [`context-caching/`](context-caching/) | 28 | core · spine | here |
| [`contextcite/`](contextcite/) | 13 | full only | here |
| [`ray-tracer/`](ray-tracer/) | 15 of 35 | full only | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `llm-from-scratch` 41 h, `deploy-and-debug` 4 h | 45 |
| **spine** | `dynamo-paper` 19 h, `aws-from-scratch` 8 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `llm-from-scratch`: `attention.py`, `transformer.py`, `train.py`, `sample.py`. The causal mask ablation is the measurement to care about — a model that can see the future gets a perplexity that looks like a triumph.
2. `deploy-and-debug` next. It is short and it is the one that teaches you to read a fault signature instead of guessing.
3. `context-caching` is the week's largest piece. `kv_cache.py` first, and do not move past it until cached and uncached attention agree to **zero** — not small, zero. Every later file inherits that bug if it does not. You now have your own attention implementation from Tuesday to check it against.
4. `contextcite` after that, then start `ray-tracer` with `vec.py` and `shapes.py`. Install the CUDA toolkit on Sunday — do not spend a Monday fighting a driver.

## Done means

- Four checkers green: 8/8, 12/12, 16/16, and `ray-tracer` at 3/6.
- `serving_demo.py` reports bit-identical output with the cache on and off. If it does not, the cache is a bug, not a cache.
- You can explain why a transformer without positional information is permutation-equivariant, and demonstrate it by shuffling the input.
- `nvidia-smi` works and a hello-world kernel compiles and runs.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 8            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 7](../week-07/) · [Roadmap](../ROADMAP.md) · [Week 9 →](../week-09/)
