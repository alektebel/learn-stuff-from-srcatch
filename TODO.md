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

## 8. Bitcoin / Ethereum · ~55 h · SKELETON IN PLACE

[`week-05/blockchain-from-scratch/`](week-05/blockchain-from-scratch/) — 9 files,
46 stubs, 11 checks named and unwritten. No `solutions/`.

It sits beside `raft/` and `dynamo-paper/` on purpose. Those two cover
**crash faults with known membership**; this is the third corner —
**Byzantine faults with open membership**, where a participant profits by
lying and you do not know who they are. Sybil resistance is the only genuinely
new idea in the subject, and it is worth having next to the other two.

- [ ] **8.1** Write the eleven checks first. Each docstring in `check.py` says
      what to assert and names the weak version to avoid.
- [ ] **8.2** `chain.py` — Merkle root and SPV proof. Assert proof length grows
      as `log2(n)`, and that changing **any** transaction moves the root — per
      transaction, not just the first.
- [ ] **8.3** `pow.py` — assert hashes-to-block is **geometric**, not merely
      that the mean is `2^d`. A constant would pass a mean-only check.
- [ ] **8.4** `utxo.py` — the second spend must be refused **by the set**, not
      by a history scan. That is the whole double-spend defence.
- [ ] **8.5** `script.py` — P2PKH runs, and assert there is no jump opcode, so
      validation cost is bounded by script length.
- [ ] **8.6** `fork.py` — heaviest **work**, not most blocks. Build a case where
      the longer chain has less work; equal difficulty tests nothing.
- [ ] **8.7** `fork.py` — Nakamoto's `(q/p)^k`, simulated against the closed
      form. Include `q > 0.5`, where the probability is 1 and the formula stops
      applying.
- [ ] **8.8** `fork.py` — selfish mining, sweeping **gamma** (the share of
      honest miners that build on the attacker's released block). Assert the
      profitability threshold moves with it; a single-gamma check hides the
      result.
- [ ] **8.9** `accounts.py` — replay the same signed transaction twice with
      nonce checking off, then on. That is the exact price of dropping the UTXO
      set.
- [ ] **8.10** `evm.py` — an infinite loop must **terminate** out of gas, the
      sender must still be charged, and state must revert.
- [ ] **8.11** `trie.py` — inclusion proofs, and an **exclusion** proof for a
      key that is absent. The exclusion proof is what a plain Merkle tree
      cannot do and why the trie is a trie.
- [ ] **8.12** `pos.py` — hand-build conflicting finality and assert `slashable`
      names at least a third of the stake. Same shape as Raft's Figure 8: a
      safety checker that has never caught anything is not evidence.
- [ ] **8.13** Add to `progress.py`'s `PLAN`, cross-link from `raft/` and
      `dynamo-paper/`, and verify with `tools/verify_checks.py`.

---

## 9. AWS: three services, not thirty · ~22 h

**Mostly no — and the directory already argues why.** Its own "The rest of AWS"
table maps roughly twenty services onto the eight mechanisms already built:
Kinesis is `sqs.py` plus partitioning, Fargate is `lambda_svc.py` with a longer
container, Cognito is `iam.py` with a user directory, Secrets Manager is
`kms.py` with rotation. Adding a seventh variant of S3 would teach nothing and
would dilute a directory that currently sits at a verified 24/24.

Three mechanisms are genuinely absent. The README names the first two itself.

- [ ] **9.1 `cloudformation.py` — declarative desired state · ~8 h.**
      A dependency graph with **rollback**, which nothing in the eight has.
      Topological ordering, partial failure, rollback to last-known-good, drift
      detection, and cycle detection. This is the mechanism under Terraform and
      Kubernetes too, so it pays out well beyond AWS.
      *Checks:* a cycle is refused rather than deadlocking; a mid-way failure
      leaves **no** partially-applied resources; drift is detected on a
      resource changed out of band; and a rollback that itself fails is
      reported rather than swallowed — that last one is where real deployments
      get stuck.
- [ ] **9.2 `kinesis.py` — an ordered log with replay · ~8 h.**
      `sqs.py` deliberately has neither ordering nor replay, so this is a real
      contrast rather than a variant: per-shard ordering, a retention window,
      consumer checkpointing and iterators, and resharding. It is also Kafka.
      *Checks:* order holds **within** a shard and explicitly does not across
      shards; a consumer resuming from a checkpoint replays exactly the
      un-processed suffix; records expire out of the retention window even
      unread; and a reshard preserves per-key ordering across the split — the
      one everybody gets wrong.
- [ ] **9.3 `autoscaling.py` — a control loop · ~6 h.**
      The one the README's table misses. None of the eight has feedback
      control, and the failure modes are measurable and instructive:
      oscillation, and scaling on a **lagging** metric.
      *Checks:* the loop converges to the target under steady load; with too
      short a cooldown it **oscillates** (assert the amplitude, do not just
      assert it settles); scaling on queue depth beats scaling on CPU for a
      queue-driven workload; and scale-in on a lagging metric overshoots. Pair
      it with `optimize.py` so the cost of the overshoot is in dollars.

Then update `billing.py` to meter all three, and the README's "rest of AWS"
table to point the relevant rows at the new files.

**Explicitly not worth adding:** RDS/Aurora failover (it is `raft/` plus
replication lag), ElastiCache (`system-design/`), Route 53 (`dns-server/`),
CloudTrail (already written by `capstone.py`), Step Functions (a state machine
over `lambda_svc.py`), Organizations/SCPs (another deny layer in `iam.py`),
Glue/Athena (`database-engine/`'s planner over `s3.py`). Each is a packaging
difference, and the map in the README is the right place to say so.

---

## 10. AWS certification · ~35 h + daily drill · SKELETON IN PLACE

[`week-06/aws-certification/`](week-06/aws-certification/) — 7 files, 40 stubs,
10 checks named and unwritten.

**Do not plan on `aws-from-scratch/` clearing an exam by itself.** This repo is
deliberately anti-recall — its standard is re-deriving a decision and naming the
rejected alternative — and the exams are substantially recall: service names,
quotas, defaults, retrieval times. An arbitrary quota is not derivable.

Having done `aws-from-scratch/` you hold the harder half, and it is the half
that separates reasoning from cramming. It is not sufficient. The split is kept
visible on purpose: `check.py` grades the derivable half, `drill.py` carries the
arbitrary half and is **not** graded.

- [ ] **10.1** Write the ten checks. Two matter most: `check_elimination`
      (every distractor rejected by a **named** constraint) and
      `check_decision_is_not_a_lookup` (change one requirement, the answer
      changes at a boundary you can compute). A decision table that survives
      the second has memorised scenarios rather than rules.
- [ ] **10.2** `decide.py` — the elimination procedure. This is the shape of
      nearly every scenario question, and it is derivable given the eight
      mechanisms you already built.
- [ ] **10.3** `storage.py` — reuse the minimum-object-size and
      minimum-duration rules already checked in `pricing.py`. Include a
      lifecycle transition that **loses** money for small objects and one that
      loses it for short-lived ones — both are standard traps and both are
      arithmetic.
- [ ] **10.4** `storage.py` — EBS: assert IOPS binds for small blocks and
      throughput for large ones. A check that only tests IOPS passes a wrong
      answer on every sequential workload.
- [ ] **10.5** `network.py` — reachability over real route tables, including
      the **asymmetric-routing** case people answer from intuition. Then the
      six connectivity options, with the N where Transit Gateway overtakes a
      peering mesh computed rather than asserted.
- [ ] **10.6** `resilience.py` — RTO and standing cost order **opposite** ways
      across backup-restore / pilot light / warm standby / multi-site, and
      `pattern_for` returns the **cheapest that clears** the stated RPO/RTO.
      Returning the best pattern is the most common wrong answer.
- [ ] **10.7** `mlstack.py` — real-time, serverless, async and batch inference
      as four points on one curve, with the serverless-vs-provisioned crossover
      derived the same way as DynamoDB's in `optimize.py`. Map Feature Store
      onto `dynamodb.py`'s partition key and Pipelines onto the dependency
      graph from §9.1 — build that first if you want this to land.
- [ ] **10.8** `wellarchitected.py` — a review returning six green ticks is a
      review that was not done. Grade on reporting a **trade-off** and naming
      the pillar that fails first. This is the professional-level skill and the
      one your preparation is unusually good for.
- [ ] **10.9** `drill.py` — spaced repetition over quotas, defaults, names and
      limits. Keep it out of `check.py`; `check_drill_is_separate` enforces
      that as the directory grows. Run it **daily**, starting well before the
      exam date — this is the part that cannot be crammed in the last week and
      the part this repo's method does not help with.
- [ ] **10.10** Before any of the above, pull the **current official exam
      guides** and diff them against the check list here. Exam codes, blueprints
      and service names change, and everything above was written against a
      knowledge cutoff. Treat any conflict as the guide being right.
- [ ] **10.11** Order to sit them in: Solutions Architect Associate first (it
      shares the most with what you have built), then ML Engineer Associate,
      then Solutions Architect Professional. AI Practitioner is cheapest to
      clear and worth least — sit it only if someone is asking you for it.

---

## Last: the decision that is not a task

The full track reads **41 directories, 1,799 h, 100 h/week**, up from 87 two
commits ago, with weeks 14–18 holding 900 h between them. The twelve-week,
ten-project cut discussed on 23 Aug was never applied.

Every step above adds roughly **152 hours** (≈40 for 1–6, 55 for the blockchain,
22 for the three AWS services, 35 for the certification layer) plus a daily
drill, and removes none. Nothing in this file fixes the number,
and nothing should until you decide what the plan actually is.
