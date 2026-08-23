# Week 4 · Sep 14–Sep 20, 2026

> **Haskell, then a database from the disk up.**
> Haskell first because purity makes the next two weeks legible: a vector clock is a lattice, a query plan is a fold, and reconciliation is a join. Then the week turns to storage.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`haskell-projects/`](haskell-projects/) | 61 | full only | here |
| [`database-engine/`](database-engine/) | 16 of 50 | core · spine | here |

**77 block hours**, plus the daily Lean slot (~8.4 h) = 85 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `database-engine` 43 h | 43 |
| **spine** | `http-server` 13 h, `c-compiler` 14 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. `haskell-projects` Monday to Friday. JSON parser first (parser combinators are the payoff), then the calculator, then the build tool.
2. `database-engine` from Saturday, and the order is fixed because each file is the substrate of the next: `pager.py`, then `btree.py`. Nothing else works until a page round-trips to disk and back byte-identical.
3. Get through `pager.py` this week and no further. Slotted pages, the free space calculation, and the buffer pool with its LRU. Sixteen hours is exactly enough for that and not enough for a B+tree on top of a shaky pager.

## Done means

- Your JSON parser round-trips a nested document with escapes and unicode.
- `week-04/database-engine/` prints at least 2/18 from `python3 check.py`.
- **Before running `pager.py`'s demo, predict how many rows fit in a 4096-byte page** for a fixed record size. Then run it. The gap is the page header and the slot array, and knowing that number by feel is most of what a storage engineer knows that an application developer does not.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 4            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 3](../week-03/) · [Roadmap](../ROADMAP.md) · [Week 5 →](../week-05/)
