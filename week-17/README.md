# Week 17 · Dec 14–Dec 20, 2026

> **Patterns, light, and Haskell.**
> The heaviest week, and the most arbitrary — three unrelated things sharing a slot because they are what is left. If the plan slips, this is the week to cut from, and `system-design` is the first candidate: `aws-from-scratch` covers much of it from the mechanisms up.

## Finish this week

| Project | Hours | Track |
|---|---|---|
| [`system-design/`](system-design/) | 48 | full only |
| [`ray-tracer/`](ray-tracer/) | 35 | full only |
| [`haskell-projects/`](haskell-projects/) | 61 | full only |

**144 block hours**, plus the daily Lean slot (~8.4 h) = 152 h on the full track.

Plus the AWS drill, daily, since week 1 — [`../week-12/aws-certification/drill.py`](../week-12/aws-certification/drill.py). It is not graded and it cannot be crammed.

## What to do, in order

1. `system-design` in groups: caching, then queues, then reliability, then distribution, then the applications. Do `consistent_hash.py` without looking at week 10's version, then diff.
2. `ray-tracer`: `vec.py` → `shapes.py` → `bvh.py` → `material.py` → `render.py`. **Predict what noise does from 16 to 64 samples before reading the table.**
3. `haskell-projects` last: JSON parser first — parser combinators are the payoff — then the calculator, then the build tool. Delete each `-- TODO`.

## Done means

- `week-17/ray-tracer/` prints 6/6 and `render.ppm` opens in an image viewer.
- You can say why a BVH is 1.5x at four objects and 12x at a thousand.
- Your JSON parser round-trips a nested document with escapes and unicode.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists.
4. **Log, ten minutes** — one entry in [`../journal/`](../journal/).

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 17
python3 ../progress.py --checks
```

[← Week 16](../week-16/) · [Roadmap](../ROADMAP.md) · [Week 18 →](../week-18/)
