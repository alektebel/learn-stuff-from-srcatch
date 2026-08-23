# Week 11 · Nov 2–Nov 8, 2026

> **Distributed training, then a pager.**
> Gradient accumulation you already wrote; now there is a network in the middle. Then slotted pages, nothing further.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `distributed-training/` | 10 | core | [`../week-15/distributed-training/`](../week-15/distributed-training/) |
| `database-engine/` | pager + start of 50 | core · spine | [`../week-04/database-engine/`](../week-04/database-engine/) |

On the narrower tracks this same week is:

core: distributed-training, then pager.py. spine: pager.py only.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. `template_data_loader.py` → `template_trainer.py`.
2. Then `pager.py` only. Slotted pages, free space, LRU. **Do not open btree.py this week.**

## Done means

- The data-parallel trainer runs.
- A page that cannot find free space is a pager bug.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 11            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 10](../week-10/) · [Roadmap](../ROADMAP.md) · [Week 12 →](../week-12/)
