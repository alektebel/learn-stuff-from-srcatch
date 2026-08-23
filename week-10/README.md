# Week 10 · Oct 26–Nov 1, 2026

> **Tokens and columns, after derivations.**
> ContextCite first. SPADE and MARS-SQL both end by citing through it.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `contextcite/` | 13 | core | [`../week-08/contextcite/`](../week-08/contextcite/) |
| `spade/` | 16 | core | [`../week-08/spade/`](../week-08/spade/) |
| `mars-sql/` | 20 | core | [`../week-08/mars-sql/`](../week-08/mars-sql/) |

On the narrower tracks this same week is:

core: all three, in that order. spine: skip.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. ContextCite end to end. 14/14 before you touch SPADE.
2. SPADE: deltas → taxonomy → candidates → selector → cite.
3. MARS-SQL: grounding → ReAct → generative validation → cite the SQL.

## Done means

- 14/14, 8/8, 8/8.
- An assertion you can point at a prompt delta, and a SQL you can point at a schema column.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 10            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 9](../week-09/) · [Roadmap](../ROADMAP.md) · [Week 11 →](../week-11/)
