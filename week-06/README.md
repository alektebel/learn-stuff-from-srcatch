# Week 6 · Sep 28–Oct 4, 2026

> **Optimise, speculate, observe, then read the engines.**
> Step 11 is the first time you open vLLM / SGLang / TensorRT-LLM.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `inference-from-scratch/` | steps 7–12 | core | [`../week-10/inference-from-scratch/`](../week-10/inference-from-scratch/) |
| `deploy-and-debug/` | 10 | core · spine | [`../week-08/deploy-and-debug/`](../week-08/deploy-and-debug/) |

On the narrower tracks this same week is:

core: finish inference 12/12, then deploy-and-debug. spine: deploy-and-debug.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. GPU path: graphs, fusion, quant, one sync per step.
2. Speculative decoding — include a case that is *slower*.
3. Observability, then the load-test knee.
4. **Now** fill `compare.py`. Then `deeper.py`.
5. `deploy-and-debug` if there is time.

## Done means

- inference-from-scratch 12/12.
- A concurrency where throughput stopped scaling, and a reason.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 6            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 5](../week-05/) · [Roadmap](../ROADMAP.md) · [Week 7 →](../week-07/)
