# Week 4 · Sep 14–Sep 20, 2026

> **Functional programming, and the paper the cloud is built on.**
> Haskell for the week's first half because purity makes the second half legible: a vector clock is a lattice, and reconciliation is a join.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`haskell-projects/`](haskell-projects/) | 58 of 61 | full only | here |
| [`dynamo-paper/`](dynamo-paper/) | 21 | core · spine | here |
| `system-design/` | 2 of 48 | core | [`../week-05/system-design/`](../week-05/system-design/) |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `communication-protocols` 3 h, `c-compiler` 27 h, `compiler-and-vgpu` 11 h | 41 |
| **spine** | `http-server` 21 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. `haskell-projects` Monday to Thursday. JSON parser first (parser combinators are the payoff), then the calculator, then the build tool.
2. `dynamo-paper` Friday to Sunday, with the paper open beside you. Do `partitioning.py` and `vector_clock.py` before anything else — the rest assumes them.
3. Read Section 4 of the SOSP 2007 paper before you write `quorum.py`, not after.
4. Start `system-design` only if you finish early.

## Done means

- Your JSON parser round-trips a nested document with escapes and unicode.
- `week-04/dynamo-paper/` prints 17/17.
- **Before running `dynamo_cluster.py`, write down the availability table you expect** for (3,2,2), (3,3,1) and (3,1,3), sloppy and strict. Then run it. The cells you got wrong are the ones you actually learned something from.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 4    # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 3](../week-03/) · [Roadmap](../ROADMAP.md) · [Week 5 →](../week-05/)
