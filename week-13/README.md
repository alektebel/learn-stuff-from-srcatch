# Week 13 · Nov 16–Nov 22, 2026

> **Same partition, opposite answers.**
> One machine ACID, then the paper that throws it away, then the consensus it refuses.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `dynamo-paper/` | 21 | core · spine | [`../week-05/dynamo-paper/`](../week-05/dynamo-paper/) |
| `raft/` | 30 | core | [`../week-05/raft/`](../week-05/raft/) |

On the narrower tracks this same week is:

core: both. spine: Dynamo only.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Preference lists and vector clocks. N/R/W, sloppy quorum, hinted handoff.
2. Raft: read Figure 8 before `replication.py`.
3. Write the Raft-versus-Dynamo table in the journal.

## Done means

- 17/17 and 7/7 if you can.
- The table exists.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 13            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 12](../week-12/) · [Roadmap](../ROADMAP.md) · [Week 14 →](../week-14/)
