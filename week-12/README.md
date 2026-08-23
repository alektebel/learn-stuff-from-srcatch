# Week 12 · Nov 9–Nov 15, 2026

> **The rest of the database.**
> WAL before MVCC. A Volcano iterator is a next() that may call next().

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `database-engine/` | rest of 50 | core · spine | [`../week-04/database-engine/`](../week-04/database-engine/) |

On the narrower tracks this same week is:

core · spine: 18/18.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. `btree.py`, then `wal.py`. Crash, recover.
2. `mvcc.py` — snapshot isolation, and the write skew it lets through.
3. SQL, executor, planner, `database.py`.

## Done means

- 18/18.
- One machine, ACID. Dynamo is next week.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 12            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 11](../week-11/) · [Roadmap](../ROADMAP.md) · [Week 13 →](../week-13/)
