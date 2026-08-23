# Week 7 · Oct 5–Oct 11, 2026

> **Evaluate the query once in ℕ[X].**
> Every other question is a homomorphism. If that fails, you built an annotation scheme.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `provenance-semirings/` | 45 | core | [`../week-08/provenance-semirings/`](../week-08/provenance-semirings/) |

On the narrower tracks this same week is:

core: the whole directory, 45 h. spine: skip; jump to the pager in week 11.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. The eight semirings, and they obey the laws.
2. Sparse ℕ[X]. Specialize ac+bd by hand.
3. Annotated RA. The running query must be ac+bd.
4. **The payoff:** `h(Q_How) = Q_K` for every K. Then Datalog: refuse the cycle in How.

## Done means

- 8/8.
- Lineage of (Ada, Bar) is 4, bag is 2, cost is 7 — from one polynomial.
- You raised on a cyclic path query in ℕ[X].

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 7            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 6](../week-06/) · [Roadmap](../ROADMAP.md) · [Week 8 →](../week-08/)
