# Week 7 · Oct 5–Oct 11, 2026

> **Finish AWS, then build gradients from nothing.**
> The pivot week. Everything before it is systems; everything after it is machine learning systems, and this is where you build the arithmetic that all of it runs on.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `aws-from-scratch/` | 17 of 42 | core · spine | [`../week-06/aws-from-scratch/`](../week-06/aws-from-scratch/) |
| [`autograd/`](autograd/) | 30 | core · spine | here |
| [`llm-from-scratch/`](llm-from-scratch/) | 32 of 45 | core · spine | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `aws-from-scratch` 11 h, `autograd` 30 h, `llm-from-scratch` 4 h | 45 |
| **spine** | `database-engine` 25 h, `dynamo-paper` 2 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `aws-from-scratch` — DynamoDB, Lambda, SNS, KMS, VPC, the capstone, and the three billing files. Aim for 24/24 by Tuesday.
2. **Before running `optimize.py`, write down where you think provisioned DynamoDB overtakes on-demand.** The answer is in the price sheet and you can derive it. Most people are off by an order of magnitude.
3. `autograd` from Wednesday, and it is the highest-leverage thirty hours in the repo. `tensor.py` first: reverse mode over arrays, topological sort, gradient accumulation, and `_unbroadcast`. Do not move to `nn.py` until every operation's gradient matches central differences.
4. Then `nn.py`, `optim.py`, `train.py`, `generative.py`. Start `llm-from-scratch` on Saturday with `tokenizer.py`, which is pure string processing and needs none of the above.

## Done means

- `week-06/aws-from-scratch/` prints 24/24.
- `week-07/autograd/` prints 10/10.
- A tensor used twice gets a gradient that is the SUM of both paths, and you can say why running `backward` before the topological sort finishes produces a model that trains slightly worse forever without ever raising an error.
- Your BPE tokenizer round-trips arbitrary text, and merges apply in rank order — not in whatever order the loop happens to find them.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 7            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 6](../week-06/) · [Roadmap](../ROADMAP.md) · [Week 8 →](../week-08/)
