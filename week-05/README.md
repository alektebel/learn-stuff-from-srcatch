# Week 5 · Sep 21–Sep 27, 2026

> **The cloud, from its mechanisms up.**
> Two directories that cover the same ground from opposite ends: `system-design` as patterns in isolation, `aws-from-scratch` as the services those patterns became.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`system-design/`](system-design/) | 46 of 48 | core | here |
| [`aws-from-scratch/`](aws-from-scratch/) | 34 of 42 | core · spine | here |

**80 block hours**, plus the daily Lean slot (~8.4 h) = 88 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `compiler-and-vgpu` 5 h, `dynamo-paper` 21 h, `system-design` 16 h | 42 |
| **spine** | `http-server` 11 h, `c-compiler` 10 h | 21 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. `system-design` first, Monday to Thursday. Rate limiting, consistent hashing, caching, queues, the reliability patterns. You built consistent hashing last week in `dynamo-paper` — do it again here without looking, then diff.
2. `aws-from-scratch` from Thursday. Order is fixed and the checker enforces it: `iam.py` first, always. Everything else in the directory is gated by it.
3. Get through `iam.py`, `s3.py`, `sqs.py` and `dynamodb.py` — checks 1 to 9 — this week. The rest is week 6.

## Done means

- `week-05/aws-from-scratch/` prints at least 9/24.
- You can state IAM's evaluation rule from memory and explain why policy order cannot change the answer.
- You can explain why a hot partition key throttles a table that looks idle.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 5    # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 4](../week-04/) · [Roadmap](../ROADMAP.md) · [Week 6 →](../week-06/)
