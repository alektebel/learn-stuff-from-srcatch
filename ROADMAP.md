# Roadmap — all of it, before 2027

Start **Monday 24 August 2026**. Finish **Sunday 27 December 2026**. Eighteen weeks.

The repo is laid out to match: **`week-01/` … `week-18/`**, each with a README
carrying that week's objective, the order to work in, and a concrete *done
means*. This file is the whole plan and the arithmetic behind it; the week
folders are what you actually open on a Monday, and
[`IMPLEMENTATION_ORDER.md`](IMPLEMENTATION_ORDER.md) is the file-by-file list
underneath both. **[Start at `week-01/`](week-01/).**

## The arithmetic, before the calendar

Read this first, because it changes what you decide.

| | |
|---|---|
| Directories | **35** |
| Implementable units left (stubs, `sorry`s, TODO markers) | **2,382** |
| Estimated effort | **~1,570 hours** |
| Weeks available | **18** |
| Therefore | **87 h/week — 12.5 hours a day, seven days a week, for 126 consecutive days** |

Where the hours come from: the directories that state their own estimate in
their README (`ESTIMATED TIME: 6-8 hours`, and so on) are used directly. The
rest are units × a rate calibrated against the ones that do state a number
(≈0.45 h per unit; ≈0.30 h for the directories whose templates are unusually
specific). `python3 progress.py` prints the same table any time you want to
re-check it.

**87 h/week is not a plan.** It is more than a full-time job with no weekends
and no slack for a single bad day, and 1,570 hours is the *optimistic* end,
because none of it includes the day you lose to a CUDA driver. Attempting it is
how people finish week 6 and stop.

So the plan below is one schedule at **three sizes**. Same order, same method,
same daily structure — you pick the width, and `progress.py --track` scores you
against the one you picked. Pick honestly on day one; switching down in November
feels like failure, and switching down in August is just planning.

| Track | Directories | Hours | Per week | Per day | Who it is for |
|---|---|---|---|---|---|
| **full** | 35 | 1,573 | 87 h | 12.5 h × 7 | This is your full-time job and you have no other commitments |
| **core** | 15 | 808 | 45 h | 7.5 h × 6, one day off | You have a job, and you are serious |
| **spine** | 12 | 485 | 27 h | 4.5 h × 6, one day off | You have a job and a life, and you would rather finish |

All three finish on 27 December. They differ only in what they contain.

**The recommendation is `core`.** It holds every directory that other
directories reference, it is the largest of the three you can actually sustain
next to employment, and finishing it means the remaining twenty directories are
variations on mechanisms you already own — which is exactly the claim
[`PHILOSOPHY.md`](PHILOSOPHY.md) makes about this repo.

### What changed, and why the narrow tracks got heavier

Four directories were cut from this plan — `deepfake-creation`,
`deepfake-detection`, `quantitative-trading` and `sgl-lang`, 220 hours — and five
built to replace them: `database-engine`, `autograd`, `llm-from-scratch`, `raft`
and `ray-tracer`, about 190 hours.

The trade is deliberate. The four that left were the least transferable material
in the repo: two of them teach one media pipeline, one teaches one financial
domain, and `sgl-lang` is genuinely subsumed by `vllm-engine` plus
`context-caching`. The five that arrived are all *substrates* — things that other
directories in this repo are written on top of. You cannot reason about
`context-caching` without having built attention, or about `dynamo-paper`'s
refusal of consensus without having built consensus, or about `vllm-engine`'s
block allocator without having built a pager.

The consequence to notice: **`core` moved from 42 h/week to 45, and `spine` from
21 to 27.** All four cut directories were on the full track only, so dropping
them saved the narrow tracks nothing while the new material costs them
something. `system-design` and `communication-protocols` came off `core` to
absorb part of it — the first because `aws-from-scratch` covers the same ground
from the mechanisms up, the second because framing a UART packet is not on the
path to anything else here. That is a real cost, stated rather than hidden.

---

## Week 0 — this weekend (21–23 August)

Do not start week 1 without this. Every hour here buys back three in November.

- [ ] **Toolchain.** `gcc`, `make`, Python 3.11+, and a venv per ML directory.
      Note that the five newest directories — `database-engine`, `autograd`,
      `llm-from-scratch`, `raft`, `ray-tracer` — are pure Python 3 standard
      library on purpose, and need nothing installed at all.
- [ ] **Haskell** (if you are on `full`): `ghcup`, then `cabal install http-conduit tagsoup async`.
- [ ] **Lean 4** (if you are on `full`): `elan`, then a Mathlib-enabled project. This
      alone can eat an afternoon. Do it now, not in week 4.
- [ ] **A GPU you can actually reach.** Weeks 9–18 of the full track — CUDA,
      TensorRT, vLLM, world-models, diffusion — are not doable on a laptop.
      Colab Pro, Lambda, RunPod, or a local card. **Budget the money in
      August**, because "I will sort the GPU out later" is the single most common
      way this plan dies. The `core` track needs a GPU from week 9; `spine`
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
   expect. Hit rate, speedup, throttle count, dollars, noise at 64 samples. Then
   run it. This is the entire method of the repo: a result that surprises you is
   a gap in your model that a passing test did not reveal, and you only get that
   signal if you committed to a number first.
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
> not last Sunday. When week 14's work breaks week 5's checker, you want to find
> out in one hour, not in December.

### Full — 87 h/week, 7 days

| | |
|---|---|
| 05:30–08:30 | implement (3 h) |
| 09:00–12:00 | implement (3 h) |
| 13:00–16:00 | implement (3 h) |
| 16:30–18:00 | predict-then-run, make it green (1.5 h) |
| 20:00–21:15 | Lean (1.25 h) |
| | **11.75 h/day × 7 = 82 h**, plus a 5 h Saturday overflow block |

### Core — 45 h/week, 6 days (recommended)

| | |
|---|---|
| Mon–Fri 05:45–08:15 | implement (2.5 h) |
| Mon–Fri 18:30–22:00 | implement, then green (3.5 h) |
| Saturday | 9 h, one long session — this is where hard things get finished |
| Sunday | 6 h: 5 h implement, then 1 h regression + log |
| **Off** | one weekday evening of your choice. Take it. Every week. |

### Spine — 27 h/week, 6 days

| | |
|---|---|
| Mon–Fri 19:00–22:00 | implement, then green (3 h) |
| Saturday | 8 h |
| Sunday | 4 h: 3 h implement, 1 h regression + log |

---

## The 18 weeks

Hours are the **full** track. `C` marks a directory in `core`, `S` in `spine`.
A directory spanning several weeks is listed in each of them.

Lean sits outside the table on purpose: **504 proof obligations, ~8.4 h/week,
every week of the full track**, as a fixed daily slot rather than a block. The
`block h` column below therefore reads about 79; 79 + 8.4 is the 87 h headline.
Proofs are the one thing here that goes better in ninety-minute pieces every day
than in a marathon, and it keeps a hard, unrelated muscle warm while the rest of
the plan is C and CUDA.

### Phase 1 · Systems in C — weeks 1–3

You cannot reason about a serving stack, a NAT gateway bill, or a CUDA memory
copy without sockets, buffers and the kernel boundary. This is why it is first.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**1**](week-01/) | Aug 24–Aug 30 | `bash-from-scratch` 8 **CS** · `http-server` 71 **CS** | 79 |
| [**2**](week-02/) | Aug 31–Sep 6 | `http-server` 15 **CS** · `dns-server` 9 · `cryptographic-library` 5 · `communication-protocols` 34 · `toralizer` 16 | 79 |
| [**3**](week-03/) | Sep 7–Sep 13 | `toralizer` 5 · `firewall-from-scratch` 25 · `c-compiler` 27 **CS** · `compiler-and-vgpu` 16 **CS** · `quantum-computing-lang` 8 | 81 |

**Done means:** your HTTP server serves a real browser and survives `ab -c 100`;
your compiler compiles a program with a loop and a function call; `check.py` in
`compiler-and-vgpu` is 12/12.

### Phase 2 · Storage, consensus and the cloud — weeks 4–6

Three graded checkers and 60 checks, so this is the phase where the plan is most
objectively scorable. Use that.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**4**](week-04/) | Sep 14–Sep 20 | `haskell-projects` 61 · `database-engine` 16 **CS** | 77 |
| [**5**](week-05/) | Sep 21–Sep 27 | `database-engine` 34 **CS** · `dynamo-paper` 21 **CS** · `raft` 24 **C** | 79 |
| [**6**](week-06/) | Sep 28–Oct 4 | `raft` 6 **C** · `system-design` 48 · `aws-from-scratch` 25 **CS** | 79 |

The three-way pairing is the point of putting these together. `database-engine`
gives you ACID on one machine. `dynamo-paper` throws all of it away for
availability, and its own README says so: a globally agreed membership view
"needs consensus, which is the availability cost the paper refuses." `raft` is
the thing it is refusing. Same partition, opposite answers — and having built
both, you can say which one a given problem needs.

**Done means:** 18/18, 17/17 and 7/7 on three checkers, plus at least 7/24 on
AWS; and you can predict the Dynamo availability table *before* running the demo.

### Phase 3 · Gradients, transformers and light — weeks 7–9

The pivot. Everything before it is systems; everything after it is machine
learning systems, and this phase is where you build the arithmetic underneath
all of it — from `y = w @ x + b` and a topological sort to a working transformer,
and then the KV cache that production puts in front of one.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**7**](week-07/) | Oct 5–Oct 11 | `aws-from-scratch` 17 **CS** · `autograd` 30 **CS** · `llm-from-scratch` 32 **CS** | 79 |
| [**8**](week-08/) | Oct 12–Oct 18 | `llm-from-scratch` 13 **CS** · `deploy-and-debug` 10 **CS** · `context-caching` 28 **CS** · `contextcite` 13 · `ray-tracer` 15 | 79 |
| [**9**](week-09/) | Oct 19–Oct 25 | `ray-tracer` 20 · `cuda-from-scratch` 59 **CS** | 79 |

`ray-tracer` is in this phase rather than on its own because of where it sits in
the *order*: it is the purest embarrassingly-parallel workload in the repo, and
you meet it in the week before the hardware designed for exactly that shape.

**Done means:** 10/10 on `autograd`, 8/8 on `llm-from-scratch`, 16/16 on
`context-caching` — and `serving_demo.py` reporting **bit-identical** output with
the cache on and off, checked against the attention you wrote yourself the week
before.

### Phase 4 · GPUs and inference — weeks 10–15

The largest and most expensive phase, in both hours and dollars. It is also the
one every ML-serving job description is actually asking about.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**10**](week-10/) | Oct 26–Nov 1 | `cuda-from-scratch` 63 **CS** · `ml-inference` 16 **C** | 79 |
| [**11**](week-11/) | Nov 2–Nov 8 | `ml-inference` 79 **C** | 79 |
| [**12**](week-12/) | Nov 9–Nov 15 | `ml-inference` 42 **C** · `tensorrt-inference` 37 | 79 |
| [**13**](week-13/) | Nov 16–Nov 22 | `tensorrt-inference` 72 · `vllm-engine` 7 **C** | 79 |
| [**14**](week-14/) | Nov 23–Nov 29 | `vllm-engine` 79 **C** | 79 |
| [**15**](week-15/) | Nov 30–Dec 6 | `vllm-engine` 70 **C** · `distributed-training` 10 | 80 |

**Done means:** a hand-written kernel within a stated factor of cuBLAS and you
can say *why* the factor is what it is; a paged-attention engine serving
concurrent requests with **bit-identical output** to the unbatched path — the
one invariant `context-caching` exists to teach.

### Phase 5 · Generative models and the long tail — weeks 16–18

| Week | Dates | Work | block h |
|---|---|---|---|
| [**16**](week-16/) | Dec 7–Dec 13 | `world-models` 78 | 78 |
| [**17**](week-17/) | Dec 14–Dec 20 | `world-models` 28 · `diffusion-models` 51 | 79 |
| [**18**](week-18/) | Dec 21–Dec 27 | `diffusion-models` 40 · `spectral-graphs` 5 · `sas-lineage-tool` 8 · `web-scraping` 6 · `ml-in-production` 8 · `mlops` 12 | 79 |

Both of these are generative models you have already met a small version of:
`autograd/generative.py` builds the VAE that `world-models` starts from, and the
1/√N noise economics of diffusion sampling are the ones you measured in
`ray-tracer` in week 9.

Week 18 is deliberately a long tail of small directories. Finishing on five small
wins in the last week is worth more than finishing on one heroic one.

---

## Scoring yourself

```bash
python3 progress.py                      # core track, today's week
python3 progress.py --week 3             # what week 3 expects of you
python3 progress.py --track full         # all 35
python3 progress.py --track spine        # the 12-directory minimum
python3 progress.py --checks             # also run every check.py — the exact number
```

It prints per-directory bars, flags anything past its due week with a red `!`,
and closes with the only line that matters:

```
  0 h of work done; the plan says 397 h by the end of week 5.
  Behind by 397 h (5.0 weeks).
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
| 1. `lean-proofs` | 151 h | The biggest single saving, and the easiest to pick up later — proofs do not go stale |
| 2. `tensorrt-inference` | 109 h | Vendor-specific; `ml-inference` already taught the ideas |
| 3. `world-models` + `diffusion-models` | 197 h | The most GPU-hungry hours in the plan, and the last to be reached |
| 4. `haskell-projects` | 61 h | Orthogonal to every other track here |
| 5. `system-design` | 48 h | `aws-from-scratch` covers the same ground from the mechanisms up |
| 6. `ray-tracer` | 35 h | Self-contained; nothing else in the repo depends on it |

That is 601 hours. The remaining 164 between `full` and `core` is the week-2
protocol work and the week-18 long tail — cut those last, individually, as
needed. Which is the honest way to read the `core` track: it is `full` with the
six cuts already made, made in August by someone calm rather than in November by
someone tired.

**Never cut these**, at any tier, because everything else in the repo is written
against them: `http-server`, `c-compiler`, `database-engine`, `dynamo-paper`,
`aws-from-scratch`, `autograd`, `llm-from-scratch`, `context-caching`,
`cuda-from-scratch`.

---

## What "done" means

Not "I implemented the stubs". This repo's own standard, from
[`PHILOSOPHY.md`](PHILOSOPHY.md), is higher and more useful:

> You could **re-derive** the design decision, name the alternative that was
> rejected, and say what limit case forced the complication.

The concrete test, at the end of each directory: **close the files and explain
to someone why it is built that way.** Why a preference list must skip virtual
nodes on the same physical machine. Why an explicit Deny cannot be order
dependent. Why "a majority has it" is the wrong commit rule for Raft. Why only
leaves of a radix cache are evictable. Why `t_min = 0.001` is a guess about
scene scale rather than a derived constant.

If you can do that, the directory is done even with a stub left in it. If you
cannot, it is not done even at 24/24 — and the fix is not more code, it is
`git log -p` on your own week and the `DESIGN DECISION` blocks you skimmed.
