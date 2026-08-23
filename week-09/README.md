# Week 9 · Oct 19–Oct 25, 2026

> **A database from the disk up.**
> Pages, an index, a log, isolation, a parser, a planner and an executor. The highest density of reusable mechanism per hour in the repo.

## Finish this week

| Project | Hours | Track |
|---|---|---|
| [`database-engine/`](database-engine/) | 50 | core · spine |

**50 block hours**, plus the daily Lean slot (~8.4 h) = 58 h on the full track.

Plus the AWS drill, daily, since week 1 — [`../week-12/aws-certification/drill.py`](../week-12/aws-certification/drill.py). It is not graded and it cannot be crammed.

## What to do, in order

1. `pager.py` first: slotted pages, free-space arithmetic, the buffer pool and its LRU. **Predict how many rows fit in a 4096-byte page before running the demo.**
2. Then `btree.py`, then **`wal.py` before `mvcc.py`** — durability is one writer and a disk, isolation is several writers and each other. The other order makes both confusing.
3. Then `sql.py` (recursive descent, precedence climbing), `executor.py` (Volcano iterators), `planner.py`, `database.py`.
4. The eighteen checks are the order.

## Done means

- `week-09/database-engine/` prints 18/18.
- You can say in one sentence what `wal.py` promises after a crash and what `mvcc.py` promises a concurrent reader — and why write skew slips past the second one.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists.
4. **Log, ten minutes** — one entry in [`../journal/`](../journal/).

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 9
python3 ../progress.py --checks
```

[← Week 8](../week-08/) · [Roadmap](../ROADMAP.md) · [Week 10 →](../week-10/)
