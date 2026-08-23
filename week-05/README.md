# Week 5 · Sep 21–Sep 27, 2026

> **Indexes, logs, and the paper that gave all three up.**
> The densest conceptual week in the plan. A B+tree, write-ahead logging, MVCC and a query planner — then Dynamo, which discards every guarantee you just built, and Raft, which insists on them.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `database-engine/` | 34 of 50 | core · spine | [`../week-04/database-engine/`](../week-04/database-engine/) |
| [`dynamo-paper/`](dynamo-paper/) | 21 | core · spine | here |
| [`raft/`](raft/) | 24 of 30 | core | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `database-engine` 7 h, `dynamo-paper` 21 h, `raft` 16 h | 44 |
| **spine** | `c-compiler` 13 h, `compiler-and-vgpu` 16 h | 29 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `database-engine`: `btree.py`, `wal.py`, `mvcc.py`, `sql.py`, `executor.py`, `planner.py`, `database.py`. Thirty-four hours, and the checker's eighteen steps are the order.
2. **Do `wal.py` before `mvcc.py`.** Durability is a property of one writer and a disk; isolation is a property of several writers and each other. Doing them the other way round makes both confusing.
3. `dynamo-paper` next, with the SOSP 2007 paper open. `partitioning.py` and `vector_clock.py` before anything else — the rest assumes them.
4. `raft` last, and read the paper's Figure 8 before you write `replication.py`. It is the counterexample that makes the obvious commit rule wrong.

## Done means

- `week-04/database-engine/` prints 18/18.
- `week-05/dynamo-paper/` prints 17/17.
- **Before running `dynamo_cluster.py`, write down the availability table you expect** for (3,2,2), (3,3,1) and (3,1,3), sloppy and strict. Then run it. The cells you got wrong are the ones you actually learned something from.
- You can state, in one sentence each, what `wal.py` promises after a crash and what `mvcc.py` promises to a concurrent reader — and why write skew slips past the second one.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 5            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 4](../week-04/) · [Roadmap](../ROADMAP.md) · [Week 6 →](../week-06/)
