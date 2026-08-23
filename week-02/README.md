# Week 2 · Aug 31–Sep 6, 2026

> **Gradients from nothing, then a tokenizer.**
> The highest-leverage thirty hours in the repo, then BPE in rank order.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `autograd/` | 30 | core · spine | [`../week-07/autograd/`](../week-07/autograd/) |
| `llm-from-scratch/` | start of 55 | core · spine | [`../week-07/llm-from-scratch/`](../week-07/llm-from-scratch/) |

On the narrower tracks this same week is:

core · spine: autograd 30 h, then tokenizer.py.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. `tensor.py` first. Do not move on until every gradient matches central differences.
2. Then `nn.py`, `optim.py`, `train.py`, `generative.py`. Aim at 10/10.
3. Saturday: `tokenizer.py` — merges in **rank order**, not position order.

## Done means

- `autograd` is 10/10.
- A tensor used twice gets the SUM of both paths.
- `decode(encode(text))` is exact, and reversing merge rank changes the tokenisation.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 2            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 1](../week-01/) · [Roadmap](../ROADMAP.md) · [Week 3 →](../week-03/)
