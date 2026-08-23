# Week 8 · Oct 12–Oct 18, 2026

> **A query is a justification tree.**
> SLD diverges on the even loop. CoSLD succeeds. Duals make `not p(X)` a call.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `scasp/` | 45 | core | [`../week-08/scasp/`](../week-08/scasp/) |

On the narrower tracks this same week is:

core: the whole engine, 45 h. spine: skip.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Unification with the occurs check.
2. SLD: member/2; even loop returns None.
3. Dual rules: fact → neq; conjuncts De Morgan.
4. CoSLD, then the tree. opus flies; tweety does not.

## Done means

- 8/8.
- The even-loop tree is marked `coinductive`.
- `atoms_used(flies(opus))` cites sparrow, not penguin(tweety).

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 8            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 7](../week-07/) · [Roadmap](../ROADMAP.md) · [Week 9 →](../week-09/)
