# TODO — the work, in steps

Skeletons are in place. Nothing below is implemented. Each step says what to
build and how you will know it is right; the *why* for each is in
[`REMAINING.md`](REMAINING.md).

Steps 1–2 block everything else. Do them before the plan starts.

---

## 1. Verify the 59 existing checks · ~1 day · BLOCKING

Seven directories are check-only. None of their checkers has been run against a
working implementation, so it is not known whether they are passable. An
impossible check is worse than no check: the method here is to trust the
checker over your own reading, and one bad assertion turns that into a trap.

Skeleton: [`tools/verify_checks.py`](tools/verify_checks.py).

- [ ] **1.1** Implement `stage()` — copy templates + `check.py` to a temp dir,
      overlay references from a path **outside the repo**. The checker must
      never run from the repo, so an interrupt cannot leave a reference behind.
- [ ] **1.2** Implement `run_checker()` — parse `"N/M passing"`. A missing line
      is a failure, not a zero; a crashed checker must not look like a clean run.
- [ ] **1.3** Implement `apply_injection()` — assert the target string appears
      **exactly once** before replacing. A silent no-op injection reports as
      "caught" and is the main way a harness like this lies to you.
- [ ] **1.4** Implement `verify()` — pass N/N, then one injection per check,
      each required to make its own check FAIL. Exit non-zero on any MISS.
- [ ] **1.5** Write references (outside the repo) and run it, in this order.
      Delete each reference set as soon as its directory is green.

      | Directory | Stubs | Checks | Pass N/N | Injections caught |
      |---|---|---|---|---|
      | `week-08/provenance-semirings/` | 76 | 8 | ☐ | ☐ |
      | `week-10/inference-from-scratch/` | 41 | 12 | ☐ | ☐ |
      | `week-08/linc/` | 22 | 8 | ☐ | ☐ |
      | `week-08/scasp/` | 22 | 8 | ☐ | ☐ |
      | `week-08/spade/` | 19 | 8 | ☐ | ☐ |
      | `week-08/mars-sql/` | 15 | 8 | ☐ | ☐ |
      | `week-07/llm-from-scratch/distill.py` | 12 | 7 | ☐ | ☐ |

- [ ] **1.6** Add a line to each directory's README recording the date it was
      verified and at what counts. An unverified checker should be visibly
      unverified.

**Done means:** `verify_checks.py` exits 0 for all seven, and no reference
implementation exists anywhere under the repo.

---

## 2. Replace the one check that grades recall · ~2 h

`week-07/llm-from-scratch/check.py` step 14 grades `paper_choices()`, a dict of
strings — it tests whether you remember what a paper said. Every other check in
this repo tests a mechanism.

- [ ] **2.1 MiniLM** — attention distributions and value-relation matrices are
      `L × L`. Check with a **width-8 teacher and a width-4 student**: the
      targets stay comparable with no layer mapping and no learned projection.
      That property *is* the paper.
- [ ] **2.2 GKD** — generalised JSD with `m = β·teacher + (1−β)·student`. Check
      both limits numerically: `D/β → KL(t‖s)` as `β→0`, `D/(1−β) → KL(s‖t)` as
      `β→1`. A version matching only one limit has the mixture coefficient wrong.
- [ ] **2.3 OPSD** — teacher identical to student ⇒ gradient **exactly zero**,
      both directions. Also: EMA the *logits*, not the probabilities — check
      that the probability-space average drifts toward uniform and the
      logit-space one does not.
- [ ] **2.4 SDPO** — several distinct methods use that name. Do not assert one.
      Check the identity they share: `π* ∝ π_ref·exp(r/β)`, inverted to
      `r = β·log(π*/π_ref) + β·log Z`, with the constant cancelling in a
      pairwise margin. Then read the specific paper you meant and add a check
      for whatever it changes.
- [ ] **2.5** Delete `paper_choices()` and its check.

---

## 3. Close the three unreachable claims in `distill.py` · ~4 h

Named in docstrings, not reachable by any check.

- [ ] **3.1** The gradients: `∂KL(t‖s)/∂z = s − t` forward, and
      `s·(log(s/t) − KL(s‖t))` reverse. The leading `s` is *why* mode collapse
      happens — check it against finite differences.
- [ ] **3.2** Mass-covering vs mode-seeking, measured: fit a deliberately
      **unimodal** student to a bimodal teacher under each divergence. Forward
      lands in the valley; reverse commits to one mode. Assert the valley mass
      differs by a wide margin. They only diverge when the student cannot match
      the teacher — which is always, so the student family must be underpowered
      or the check is vacuous.
- [ ] **3.3** The privilege floor in closed form: best blind student is the
      marginal `E_z[p(y|x,z)]`, its loss is exactly `I(y;z|x)`. Assert the two
      agree to floating point, then sweep student capacity and assert the loss
      stops at that wall rather than approaching zero.

Draft templates for all three, plus MiniLM/GKD/OPSD/DPO, are in the scratchpad
at `.../scratchpad/superseded/llm/`. Merge or discard.

---

## 4. RL post-training · ~30 h · SKELETON IN PLACE

[`week-05/rl-posttraining/`](week-05/rl-posttraining/) — 9 files, 32 stubs,
9 checks named and unwritten. No `solutions/`.

- [ ] **4.0** Decide what happens to
      `origin/claude/rl-posttraining-llm-exercises-wzfe8w`, which has 8 phases,
      GRPO machinery, a tiny SQL env, tests **and solutions**. Merge `common/`
      and the phase structure into this directory and strip the solutions, or
      keep it separate and accept the overlap. **Nothing below should start
      until this is decided**, or you will build the same thing twice.
- [ ] **4.1** Write the nine checks in `check.py` first, each so it fails
      against a deliberately wrong implementation. The TODO in each check's
      docstring says what to assert.
- [ ] **4.2** `env.py`, then `policy_gradient.py` — the identity against
      central differences. Nothing later is meaningful until this passes.
- [ ] **4.3** `baselines.py` — include the **action-dependent** baseline and
      show it biased. That is the one people ship by accident.
- [ ] **4.4** `kl_estimators.py` — k1/k2/k3, with the negative-sample fraction.
- [ ] **4.5** `ppo.py` — clip fraction rising monotonically with drift.
- [ ] **4.6** `grpo.py` — group baseline, then Dr. GRPO's length bias measured
      and removed. The bias must be clearly non-zero before removal, or the
      check is testing nothing.
- [ ] **4.7** `async_rl.py` — ESS against lag, and the threshold crossing.
- [ ] **4.8** `reward_hacking.py` — the true reward must **turn over**.
      Asserting only that the proxy rises tests nothing; that is what
      optimisers do.
- [ ] **4.9** `dpo.py` — round-trip the identity; the recovered reward differs
      by a **constant**, not by zero.
- [ ] **4.10** Add to `progress.py`'s `PLAN` and to the week-05 README, and
      wire the cross-links to `distill.py` and `mars-sql/`.
- [ ] **4.11** Verify with `tools/verify_checks.py`, then delete the references.

---

## 5. Mine and delete `provenance-reasoning` · ~2 h

In the scratchpad at `.../scratchpad/superseded/provenance-reasoning/` — five
templates **and** five working solutions with running demos. Overlaps
`week-08/provenance-semirings/` and `week-08/linc/`. Two results in it are not
in the shipped version:

- [ ] **5.1** The **universality check** — evaluate a query once in `N[X]`, then
      derive nine semantics by mapping the answer through a homomorphism, and
      assert it agrees with having evaluated natively in each. That property is
      what makes provenance one feature instead of nine, and an operator that
      does not use `+` and `×` where the definition says passes every row-count
      test and fails exactly this one.
- [ ] **5.2** The **absorption/termination result** — `a + a×b == a` predicts
      whether a recursive program reaches a fixpoint. Measured: `Boolean`,
      `Tropical` and `PosBool` converge on cyclic data; `Counting` and `N[X]`
      never do, because the answer really is infinite.
- [ ] **5.3** Delete the directory.

---

## 6. Housekeeping · ~1 h

- [ ] **6.1** `IMPLEMENTATION_SUMMARY.md` and
      `LEAN_PROOFS_EXPANSION_SUMMARY.md` are pre-reorganisation artifacts at the
      root — same category as the deepfake summary already deleted.
- [ ] **6.2** `_codeql_detected_source_root` at the root: gitignore or remove.
- [ ] **6.3** **Week folders no longer match `PLAN`'s due weeks.** `raft` and
      `dynamo-paper` live in `week-05/` and are due week 13; `ray-tracer` is in
      `week-08/` and due 17. Either re-home the directories or state in the
      READMEs that the folder is a name, not a date. Right now both readings
      look supported and one of them is wrong.
- [ ] **6.4** `LOG.md` still does not exist, and every week README instructs
      that it be written daily. If `journal/` is now that thing, say so in the
      week READMEs so the instruction points somewhere real.
- [ ] **6.5** Confirm `week-11` and `week-14` READMEs point at where their work
      actually lives — neither has a directory of its own.

---

## 7. The decision that is not a task

The full track reads **41 directories, 1,799 h, 100 h/week**, up from 87 two
commits ago, with weeks 14–18 holding 900 h between them. The twelve-week,
ten-project cut discussed on 23 Aug was never applied.

Steps 1–6 add roughly 40 hours and remove none. Nothing in this file fixes the
number, and nothing should until you decide what the plan actually is.
