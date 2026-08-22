# Week 6 · Sep 28–Oct 4, 2026

> **Four checkers green in one week.**
> The densest verification week in the plan. Finish AWS, then LLM serving, then attribution, and end the week with four graded checkers passing.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `aws-from-scratch/` | 8 of 42 | core · spine | [`../week-05/aws-from-scratch/`](../week-05/aws-from-scratch/) |
| [`deploy-and-debug/`](deploy-and-debug/) | 10 | core · spine | here |
| [`context-caching/`](context-caching/) | 28 | core · spine | here |
| [`contextcite/`](contextcite/) | 13 | core · spine | here |
| `cuda-from-scratch/` | 22 of 122 | core · spine | [`../week-07/cuda-from-scratch/`](../week-07/cuda-from-scratch/) |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `system-design` 32 h, `aws-from-scratch` 9 h | 41 |
| **spine** | `c-compiler` 17 h, `compiler-and-vgpu` 3 h | 20 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `aws-from-scratch` — Lambda, SNS, KMS, VPC, the capstone, and the three billing files. Aim for 24/24 by Tuesday.
2. `deploy-and-debug` next: it is short and it is the one that teaches you to read a fault signature instead of guessing.
3. `context-caching` is the week's largest piece. `kv_cache.py` first, and do not move past it until cached and uncached attention agree to **zero** — not small, zero. Every later file inherits that bug if it does not.
4. `contextcite` last, then open `cuda-from-scratch` and get the toolchain working before week 7 starts. Do not spend Sunday night fighting a driver.

## Done means

- Four checkers: 24/24, 12/12, 16/16, 14/14.
- **Before running `aws-from-scratch/solutions/optimize.py`, write down where you think provisioned DynamoDB overtakes on-demand.** The answer is in the price sheet, and you can derive it.
- `serving_demo.py` reports bit-identical output with the cache on and off. If it does not, the cache is a bug, not a cache.
- `nvidia-smi` works and a hello-world kernel compiles and runs.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 6    # where the plan says you should be
python3 ../progress.py --checks    # what actually passes
```

[← Week 5](../week-05/) · [Roadmap](../ROADMAP.md) · [Week 7 →](../week-07/)
