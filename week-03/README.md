# Week 3 · Sep 7–Sep 13, 2026

> **A transformer, then distillation.**
> Build the model, watch the causal-mask ablation, then teach a smaller one.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `llm-from-scratch/` | rest of 55 | core · spine | [`../week-07/llm-from-scratch/`](../week-07/llm-from-scratch/) |

On the narrower tracks this same week is:

core · spine: attention through distill.py.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. `attention.py` — causal mask and sqrt(d_k). Measure the entropy table.
2. `transformer.py`, `train.py`, `sample.py`. Run the mask ablation on purpose.
3. `distill.py` — forward vs reverse KL, on vs off policy, OPD vs RL vs SFT, OPSD, Privilege Illusion.

## Done means

- `python3 check.py` prints 15/15.
- Removing the causal mask made loss better and the model worthless.
- You can separate a privilege tell from a capability token.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 3            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 2](../week-02/) · [Roadmap](../ROADMAP.md) · [Week 4 →](../week-04/)
