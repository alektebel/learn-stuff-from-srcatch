# Roadmap — all of it, before 2027

Start **Monday 24 August 2026**. Finish **Sunday 27 December 2026**. Eighteen weeks.

The week folders still hold the original layout; **this file and the journal
are the authority on order**. A directory is found wherever it lives
(`progress.py` locates it). **[Start at `week-01/`](week-01/)** — that README
now points at AWS, not at a shell.

### The order, as of 23 August 2026

1. **AWS from scratch** — the eight mechanisms and the bill
2. **LLM work** — autograd, a transformer, distillation, the KV cache, then a
   serving stack you build before reading vLLM
3. **Provenance-reasoning** — the free semiring, then s(CASP), then LINC
   (a proof you can replay; the LLM only parses). ContextCite / SPADE /
   MARS-SQL after that, for tokens and columns
4. **Distributed training**
5. **Database engine**
6. **The rest** — Dynamo/Raft, systems in C, CUDA, the vendor engines, world
   models, the tail

## The arithmetic, before the calendar

Read this first, because it changes what you decide.

| | |
|---|---|
| Directories | **41** |
| Implementable units left (stubs, `sorry`s, TODO markers) | **~2,590** |
| Estimated effort | **~1,800 hours** |
| Weeks available | **18** |
| Therefore | **~93 h/week on `full` — still not a plan, see the tracks** |

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
| **full** | 41 | ~1,800 | 100 h | 14 h × 7 | This is your full-time job and you have no other commitments |
| **core** | 23 | ~1,050 | 58 h | 9.5 h × 6, one day off | You have a job, and you are serious |
| **spine** | 12 | ~495 | 28 h | 4.5 h × 6, one day off | You have a job and a life, and you would rather finish |

All three finish on 27 December. They differ only in what they contain.

**The recommendation is `core`.** It holds every directory that other
directories reference, including the 120 h provenance stack that makes an
LLM answer a proof you can replay. Finishing it means the remaining
directories are variations on mechanisms you already own — which is exactly
the claim [`PHILOSOPHY.md`](PHILOSOPHY.md) makes about this repo. Core is
heavier than it was before those three directories existed; that is the
trade you made on 23 August.

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
- [ ] **A GPU you can actually reach.** CUDA, TensorRT, vLLM, world-models and
      diffusion are now in weeks 14–18. Still budget the money in August.
      `inference-from-scratch` (weeks 4–6) runs on a simulated GPU and does
      **not** need a card. The `core` track needs a real GPU from week 17;
      `spine` only for `cuda-from-scratch`.
- [ ] **Baseline.** `python3 progress.py --track core --checks`. Every bar should
      read 0%. That is the point — you want the zero on the record.
- [ ] **The journal.** `python3 journal/serve.py` and write today's post
      (`journal/posts/2026-08-23.md`). From today on, the ten-minute log is a
      blog post with an expected title, not a loose `LOG.md`.
- [ ] **Pick your track and write it down** in this file, on the line below.

> My track: `________`  · started: `________`

Then open [`week-01/`](week-01/) — AWS, not a shell — and read its README before Monday.

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
4. **Log, ten minutes.** Open the journal (`python3 journal/serve.py`) and write
   the day's post. The expected title is already there. What you built, what
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

### Phase 1 · AWS — week 1

The cloud as eight mechanisms and a bill. Do the whole directory before you
open `autograd/`. `iam.py` is always first; everything else is gated by it.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**1**](week-01/) | Aug 24–Aug 30 | `aws-from-scratch` 42 **CS** · start `autograd` if 24/24 lands early | 42+ |

**Done means:** 24/24 on `python3 check.py`, and you predicted the
provisioned-vs-on-demand DynamoDB crossover *before* running `optimize.py`.

### Phase 2 · LLM work — weeks 2–6

Gradients, a transformer, distillation, the cache in front of attention, then
a serving stack on a simulated GPU. Do not open `vllm-engine/` until step 11
of `inference-from-scratch`.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**2**](week-02/) | Aug 31–Sep 6 | `autograd` 30 **CS** · `llm-from-scratch` tokenizer + attention **CS** | — |
| [**3**](week-03/) | Sep 7–Sep 13 | `llm-from-scratch` transformer, train, sample, `distill.py` **CS** | — |
| [**4**](week-04/) | Sep 14–Sep 20 | `context-caching` 28 **CS** · start `inference-from-scratch` **C** | — |
| [**5**](week-05/) | Sep 21–Sep 27 | `inference-from-scratch` steps 1–6 **C** | — |
| [**6**](week-06/) | Sep 28–Oct 4 | `inference-from-scratch` steps 7–12 **C** · `deploy-and-debug` 10 **CS** | — |

**Done means:** 10/10 autograd, 15/15 llm (including distillation), 16/16
context-caching with bit-identical cached/uncached output, 12/12 inference.

### Phase 3 · Provenance-reasoning — weeks 7–10

An LLM answer is traceable when it is a proof object. It is replicable
when the prover is deterministic given the parse. The LLM is only
allowed to parse — and in this repo it is a fault-injected stand-in,
because there is no model here that can emit FOL.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**7**](week-07/) | Oct 5–Oct 11 | `provenance-semirings` 45 **C** — evaluate once in ℕ[X] | 45 |
| [**8**](week-08/) | Oct 12–Oct 18 | `scasp` 45 **C** — unification, duals, even loops, the tree | 45 |
| [**9**](week-09/) | Oct 19–Oct 25 | `linc` 30 **C** — pluggable parse, prover, sweep the error rate | 30 |
| [**10**](week-10/) | Oct 26–Nov 1 | `contextcite` 13 **C** · `spade` 16 **C** · `mars-sql` 20 **C** | 49 |

**Done means:** 8/8 on the semirings (the homomorphism holds), 8/8 on
s(CASP) (opus flies; tweety does not; the even loop is a tree), 8/8 on
LINC (gold is 3/3; the sweep is monotone; lineage of p1 is `{p0,p1,p2}`
via `specialize`). Then 14/14, 8/8, 8/8 on the citation layer.

Spine skips this phase and jumps to the pager in week 11.

### Phase 4 · Distributed training, then a database — weeks 11–12

| Week | Dates | Work | block h |
|---|---|---|---|
| [**11**](week-11/) | Nov 2–Nov 8 | `distributed-training` 10 **C** · `database-engine` pager only **CS** | — |
| [**12**](week-12/) | Nov 9–Nov 15 | `database-engine` MVCC, SQL, planner, executor **CS** | — |

**Done means:** the data-parallel trainer runs; `database-engine` is 18/18.

### Phase 5 · The rest — weeks 13–18

Compressed: three weeks of provenance took three weeks from the tail.
Dynamo and Raft still sit next to the database. The C / CUDA / vendor
stack is now the first thing you cut if you slip — not the proof stack.

| Week | Dates | Work | block h |
|---|---|---|---|
| [**13**](week-13/) | Nov 16–Nov 22 | `dynamo-paper` 21 **CS** · `raft` 30 **C** | 51 |
| [**14**](week-14/) | Nov 23–Nov 29 | `bash-from-scratch` 8 **CS** · `http-server` start **CS** | — |
| [**15**](week-15/) | Nov 30–Dec 6 | finish `http-server` · `dns-server` · crypto · protocols | — |
| [**16**](week-16/) | Dec 7–Dec 13 | `c-compiler` **CS** · `compiler-and-vgpu` **CS** · firewall · toralizer | — |
| [**17**](week-17/) | Dec 14–Dec 20 | Haskell · system-design · `cuda-from-scratch` **CS** · `ml-inference` **C** | — |
| [**18**](week-18/) | Dec 21–Dec 27 | TensorRT · vLLM **C** · world-models · diffusion · the tail | — |

---

## Scoring yourself

```bash
python3 progress.py                      # core track, today's week
python3 progress.py --week 3             # what week 3 expects of you
python3 progress.py --track full         # all 41
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
`cuda-from-scratch`, `inference-from-scratch`, `provenance-semirings`,
`scasp`, `linc`.

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
