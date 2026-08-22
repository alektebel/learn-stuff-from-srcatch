# Roadmap — all of it, before 2027

Start **Monday 24 August 2026**. Finish **Sunday 27 December 2026**. Eighteen weeks.

The repo is laid out to match: **`week-01/` … `week-18/`**, each with a README
carrying that week's objective, the order to work in, and a concrete *done
means*. This file is the whole plan and the arithmetic behind it; the week
folders are what you actually open on a Monday. **[Start at `week-01/`](week-01/).**

## The arithmetic, before the calendar

Read this first, because it changes what you decide.

| | |
|---|---|
| Directories | **34** |
| Implementable units left (stubs, `sorry`s, TODO markers) | **2,227** |
| Estimated effort | **~1,600 hours** |
| Weeks available | **18** |
| Therefore | **89 h/week — 12.7 hours a day, seven days a week, for 126 consecutive days** |

Where the hours come from: 17 directories state their own estimate in their
README (`ESTIMATED TIME: 6-8 hours`, and so on) — those are used directly. The
rest are units × a rate calibrated against the ones that do state a number
(≈0.45 h per unit; ≈0.30 h for the six directories whose templates are unusually
specific). `python3 progress.py` prints the same table any time you want to
re-check it.

**89 h/week is not a plan.** It is more than a full-time job with no weekends
and no slack for a single bad day, and 1,600 hours is the *optimistic* end,
because none of it includes the day you lose to a CUDA driver. Attempting it is
how people finish week 6 and stop.

So the plan below is one schedule at **three sizes**. Same order, same method,
same daily structure — you pick the width, and `progress.py --track` scores you
against the one you picked. Pick honestly on day one; switching down in November
feels like failure, and switching down in August is just planning.

| Track | Directories | Hours | Per week | Per day | Who it is for |
|---|---|---|---|---|---|
| **full** | 34 | 1,603 | 89 h | 12.7 h × 7 | This is your full-time job and you have no other commitments |
| **core** | 14 | 748 | 42 h | 6 h × 6, one day off | You have a job, and you are serious |
| **spine** | 10 | 373 | 21 h | 3 h × 6, one day off | You have a job and a life, and you would rather finish |

All three finish on 27 December. They differ only in what they contain.

**The recommendation is `core`.** It holds every directory that other
directories reference, it is the largest of the three you can actually sustain
next to employment, and finishing it means the remaining twenty directories are
variations on mechanisms you already own — which is exactly the claim
[`PHILOSOPHY.md`](PHILOSOPHY.md) makes about this repo.

---

## Week 0 — this weekend (21–23 August)

Do not start week 1 without this. Every hour here buys back three in November.

- [ ] **Toolchain.** `gcc`, `make`, Python 3.11+, and a venv per ML directory.
- [ ] **Haskell** (if you are on `full`): `ghcup`, then `cabal install http-conduit tagsoup async`.
- [ ] **Lean 4** (if you are on `full`): `elan`, then a Mathlib-enabled project. This
      alone can eat an afternoon. Do it now, not in week 4.
- [ ] **A GPU you can actually reach.** Weeks 7–17 of the full track — CUDA,
      TensorRT, vLLM, SGLang, world-models, diffusion — are not doable on a
      laptop. Colab Pro, Lambda, RunPod, or a local card. **Budget the money in
      August**, because "I will sort the GPU out later" is the single most common
      way this plan dies. The `core` track needs a GPU from week 7; `spine`
      needs one only for `cuda-from-scratch`.
- [ ] **Baseline.** `python3 progress.py --track core --checks`. Every bar should
      read 0%. That is the point — you want the zero on the record.
- [ ] **Pick your track and write it down** in this file, on the line below.

> My track: `________`  · started: `________`

Then open [`week-01/`](week-01/) and read its README before Monday.

---

## The daily structure

The hours matter less than the shape. Every tier uses the same four moves, in
this order, and the order is the method:

1. **Implement.** Longest block, first thing, hardest unfinished stub. Do not
   open `solutions/` — not to "check the approach", not to "see the signature".
   The solution is a comparison you make *afterwards*, and reading it first
   converts an exercise into a transcription.
2. **Predict, then run.** Before you run a demo, write down the number you
   expect. Hit rate, speedup, throttle count, dollars. Then run it. This is the
   entire method of the repo: a result that surprises you is a gap in your model
   that a passing test did not reveal, and you only get that signal if you
   committed to a number first.
3. **Make it green.** Where a `check.py` exists, that is the stop condition —
   not "it looks right". Where one does not, the stop condition is the file's own
   demo printing a table you predicted.
4. **Log, ten minutes.** One line per unit in `LOG.md`: what you built, what
   surprised you, which design decision you would now defend differently. Ten
   minutes a day is four hours over the plan, and it is the only artifact that
   still exists in a year.

And one rule per week:

> **Sunday is regression day. No new code.** Re-run *every* checker you have
> built so far — `python3 progress.py --track <yours> --checks` does it in one
> command — and write two sentences on what you can now re-derive that you could
> not last Sunday. When week 12's work breaks week 4's checker, you want to find
> out in one hour, not in December.

### Full — 89 h/week, 7 days

| | |
|---|---|
| 05:30–08:30 | implement (3 h) |
| 09:00–12:00 | implement (3 h) |
| 13:00–16:00 | implement (3 h) |
| 16:30–18:00 | predict-then-run, make it green (1.5 h) |
| 20:00–21:15 | Lean (1.25 h) |
| | **11.75 h/day × 7 = 82 h**, plus a 7 h Saturday overflow block |

### Core — 42 h/week, 6 days (recommended)

| | |
|---|---|
| Mon–Fri 05:45–08:15 | implement (2.5 h) |
| Mon–Fri 18:30–22:00 | implement, then green (3.5 h) |
| Saturday | 8 h, one long session — this is where hard things get finished |
| Sunday | 4 h: 3 h implement, then 1 h regression + log |
| **Off** | one weekday evening of your choice. Take it. Every week. |

### Spine — 21 h/week, 6 days

| | |
|---|---|
| Mon–Fri 19:00–21:30 | implement, then green (2.5 h) |
| Saturday | 6 h |
| Sunday | 2.5 h: 1.5 h implement, 1 h regression + log |

---

## The 18 weeks

Hours are the **full** track. `C` marks a directory in `core`, `S` in `spine`.
A directory spanning several weeks is listed in each of them.

Lean sits outside the table on purpose: **504 proof obligations, ~8.5 h/week,
every week of the full track**, as a fixed daily slot rather than a block. The
`block h` column below therefore reads about 81; 81 + 8.5 is the 89 h headline.
Proofs are the one thing here that goes better in ninety-minute pieces every day
than in a marathon, and it keeps a hard, unrelated muscle warm while the rest of
the plan is C and CUDA.

### Phase 1 · Systems in C — weeks 1–3

You cannot reason about a serving stack, a NAT gateway bill, or a CUDA memory
copy without sockets, buffers and the kernel boundary. This is why it is first.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**1**](week-01/) | Aug 24–30 | `bash-from-scratch` 8 **CS** · `http-server` 73 **CS** | 81 |
| [**2**](week-02/) | Aug 31–Sep 6 | `http-server` 13 · `dns-server` 9 · `cryptographic-library` 5 · `communication-protocols` 34 **C** · `toralizer` 19 | 80 |
| [**3**](week-03/) | Sep 7–13 | `toralizer` 2 · `firewall-from-scratch` 25 · `c-compiler` 27 **CS** · `compiler-and-vgpu` 16 **CS** · `quantum-computing-lang` 8 · `haskell-projects` 3 | 81 |

**Done means:** your HTTP server serves a real browser and survives `ab -c 100`;
your compiler compiles a program with a loop and a function call; `check.py` in
`compiler-and-vgpu` is 12/12.

### Phase 2 · Distributed systems and the cloud — weeks 4–6

Four of these six have graded checkers, so this is the phase where the plan is
most objectively scorable. Use that.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**4**](week-04/) | Sep 14–20 | `haskell-projects` 58 · `dynamo-paper` 21 **CS** · `system-design` 2 **C** | 81 |
| [**5**](week-05/) | Sep 21–27 | `system-design` 46 **C** · `aws-from-scratch` 34 **CS** | 80 |
| [**6**](week-06/) | Sep 28–Oct 4 | `aws-from-scratch` 8 · `deploy-and-debug` 10 **CS** · `context-caching` 28 **CS** · `contextcite` 13 **CS** · `cuda-from-scratch` 22 **CS** | 81 |

**Done means:** 17/17, 24/24, 12/12, 16/16 and 14/14 on five checkers, and you
can predict the Dynamo availability table and the DynamoDB cost crossover
*before* running either demo.

### Phase 3 · GPUs and inference — weeks 7–13

The largest and most expensive phase, in both hours and dollars. It is also the
one every ML-serving job description is actually asking about.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**7**](week-07/) | Oct 5–11 | `cuda-from-scratch` 81 **CS** | 81 |
| [**8**](week-08/) | Oct 12–18 | `cuda-from-scratch` 19 · `ml-inference` 61 **C** | 80 |
| [**9**](week-09/) | Oct 19–25 | `ml-inference` 76 **C** · `tensorrt-inference` 5 | 81 |
| [**10**](week-10/) | Oct 26–Nov 1 | `tensorrt-inference` 81 | 81 |
| [**11**](week-11/) | Nov 2–8 | `tensorrt-inference` 23 · `vllm-engine` 57 **C** | 80 |
| [**12**](week-12/) | Nov 9–15 | `vllm-engine` 81 **C** | 81 |
| [**13**](week-13/) | Nov 16–22 | `vllm-engine` 18 **C** · `sgl-lang` 63 | 81 |

**Done means:** a hand-written kernel within a stated factor of cuBLAS and you
can say *why* the factor is what it is; a paged-attention engine serving
concurrent requests with **bit-identical output** to the unbatched path — the
one invariant `context-caching` exists to teach.

### Phase 4 · Models and applications — weeks 14–18

| Week | Dates | Work | block h |
|---|---|---|---|
| [**14**](week-14/) | Nov 23–29 | `sgl-lang` 24 · `distributed-training` 10 · `world-models` 46 | 80 |
| [**15**](week-15/) | Nov 30–Dec 6 | `world-models` 60 · `diffusion-models` 21 | 81 |
| [**16**](week-16/) | Dec 7–13 | `diffusion-models` 70 · `deepfake-creation` 11 | 81 |
| [**17**](week-17/) | Dec 14–20 | `deepfake-creation` 37 · `deepfake-detection` 33 · `quantitative-trading` 10 | 80 |
| [**18**](week-18/) | Dec 21–27 | `quantitative-trading` 42 · `spectral-graphs` 5 · `sas-lineage-tool` 8 · `web-scraping` 6 · `ml-in-production` 8 · `mlops` 12 | 81 |

Week 18 is deliberately a long tail of small directories. Finishing on six small
wins in the last week is worth more than finishing on one heroic one.

---

## Scoring yourself

```bash
python3 progress.py                      # core track, today's week
python3 progress.py --week 3             # what week 3 expects of you
python3 progress.py --track full         # all 34
python3 progress.py --track spine        # the 10-directory minimum
python3 progress.py --checks             # also run every check.py — the exact number
```

It prints per-directory bars, flags anything past its due week with a red `!`,
and closes with the only line that matters:

```
  0 h of work done; the plan says 208 h by the end of week 5.
  Behind by 208 h (5.0 weeks).
```

Two honesty notes it will also print at you, because they are easy to forget:

- The Python bars are trustworthy — the marker is `raise NotImplementedError`,
  and it disappears when the function stops raising.
- The **C, CUDA and Haskell bars only move if you delete each `TODO` comment as
  you satisfy it.** Do that. It costs nothing and it is the only thing keeping
  those bars from being decoration.
- `--checks` cannot be gamed at all. When the bars and the checkers disagree,
  the checkers are right.

---

## When you slip — and you will

Slip is the normal state of a four-month plan; the failure mode is not slipping,
it is responding to slip by cutting the *daily block* instead of the *scope*.
The block is the only thing producing progress. Cut in this order:

| Cut | Saves | Why it is the right thing to lose |
|---|---|---|
| 1. `deepfake-creation` + `deepfake-detection` | 81 h | The least transferable mechanisms in the repo |
| 2. `quantitative-trading` | 52 h | Self-contained domain; nothing else depends on it |
| 3. `haskell-projects` | 61 h | Orthogonal to every other track here |
| 4. `lean-proofs` | 151 h | The biggest single saving, and the easiest to pick up later — proofs do not go stale |
| 5. `sgl-lang` | 87 h | Genuinely subsumed by `vllm-engine` + `context-caching` |
| 6. `tensorrt-inference` | 109 h | Vendor-specific; `ml-inference` already taught the ideas |

That is 541 hours — the difference between `full` and `core`, almost exactly.
Which is the honest way to read the `core` track: it is `full` with the six cuts
already made, made in August by someone calm rather than in November by someone
tired.

**Never cut these**, at any tier, because everything else in the repo is written
against them: `cuda-from-scratch`, `context-caching`, `dynamo-paper`,
`aws-from-scratch`, `c-compiler`, `http-server`.

---

## What "done" means

Not "I implemented the stubs". This repo's own standard, from
[`PHILOSOPHY.md`](PHILOSOPHY.md), is higher and more useful:

> You could **re-derive** the design decision, name the alternative that was
> rejected, and say what limit case forced the complication.

The concrete test, at the end of each directory: **close the files and explain
to someone why it is built that way.** Why a preference list must skip virtual
nodes on the same physical machine. Why an explicit Deny cannot be order
dependent. Why only leaves of a radix cache are evictable. Why more Lambda
memory can be free speed and can also be a 42× bill.

If you can do that, the directory is done even with a stub left in it. If you
cannot, it is not done even at 24/24 — and the fix is not more code, it is
`git log -p` on your own week and the `DESIGN DECISION` blocks you skimmed.
