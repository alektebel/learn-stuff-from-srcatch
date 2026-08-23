# Week 6 · Sep 28–Oct 4, 2026

> **Consensus, then the cloud built on top of it.**
> Raft and Dynamo are the same partition with opposite answers, and this is the week you can hold both. Then `system-design` and `aws-from-scratch` cover the same ground from opposite ends: patterns in isolation, and the services those patterns became.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `raft/` | 6 of 30 | core | [`../week-05/raft/`](../week-05/raft/) |
| [`system-design/`](system-design/) | 48 | full only | here |
| [`aws-from-scratch/`](aws-from-scratch/) | 25 of 42 | core · spine | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `raft` 14 h, `aws-from-scratch` 31 h | 45 |
| **spine** | `database-engine` 25 h | 25 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `raft` in the first day: `cluster.py` against a hostile network, with safety asserted after every operation.
2. Write the Raft-versus-Dynamo table in your `LOG.md` before you move on. Same partition, opposite answers — a bank ledger wants one, a shopping cart wants the other, and being able to say which you need is the point of having built both.
3. `system-design` Tuesday to Friday. Rate limiting, consistent hashing, caching, queues, the reliability patterns. You built consistent hashing last week in `dynamo-paper` — do it again here without looking, then diff.
4. `aws-from-scratch` from Saturday. Order is fixed and the checker enforces it: `iam.py` first, always. Everything else in the directory is gated by it. Get to `iam.py`, `s3.py` and `sqs.py` — checks 1 to 7.

## Done means

- `week-05/raft/` prints 7/7.
- `week-06/aws-from-scratch/` prints at least 7/24.
- You can state IAM's evaluation rule from memory and explain why policy order cannot change the answer.
- You can explain why Raft's obvious commit rule — 'a majority has it' — is wrong, using Figure 8 and not a hand-wave.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 6            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 5](../week-05/) · [Roadmap](../ROADMAP.md) · [Week 7 →](../week-07/)
