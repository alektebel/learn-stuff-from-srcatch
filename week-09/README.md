# Week 9 · Oct 19–Oct 25, 2026

> **The LLM only parses.**
> No model in this repo can emit FOL. Gold parser + fault injector + a prover that emits a proof.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `linc/` | 30 | core | [`../week-08/linc/`](../week-08/linc/) |

On the narrower tracks this same week is:

core: 30 h. spine: skip.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Ground and NNF. GoldParser looks up the fixtures.
2. Four operational faults, L1/L2/L3 bins. FaultyParser is seeded.
3. Prover: p1 True, p2 False, p3 Uncertain.
4. Sweep the error rate (predict first). Trace the proof through ℕ[X].

## Done means

- 8/8.
- Gold is 3/3. The sweep is monotone.
- Lineage of p1 is {p0,p1,p2} via specialize, not a list you appended.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 9            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 8](../week-08/) · [Roadmap](../ROADMAP.md) · [Week 10 →](../week-10/)
