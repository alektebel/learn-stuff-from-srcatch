# Week 1 · Aug 24–Aug 30, 2026

> **AWS, the eight mechanisms and the bill.**
> Do not open `autograd/` until `check.py` prints 24/24. `iam.py` is always first.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `aws-from-scratch/` | 42 | core · spine | [`../week-06/aws-from-scratch/`](../week-06/aws-from-scratch/) |

On the narrower tracks this same week is:

core: the whole AWS directory, 42 h. spine: IAM → S3 → SQS → as far as 24/24 will go.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. `iam.py` first. Everything else is gated by it.
2. Then `s3.py` → `sqs.py` → `dynamodb.py` → `lambda_svc.py` → `sns.py` → `kms.py` → `vpc.py` → `capstone.py`.
3. Then the bill: `pricing.py` → `billing.py` → `optimize.py`.
4. **Predict the provisioned-vs-on-demand DynamoDB crossover before running `optimize.py`.**

## Done means

- `week-06/aws-from-scratch/` prints 24/24.
- You can name the two rounding rules AWS bills by, and you wrote the crossover down before the demo.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 1            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[Roadmap](../ROADMAP.md) · [Week 2 →](../week-02/)
