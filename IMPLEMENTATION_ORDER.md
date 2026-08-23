# What you implement, week by week, file by file

Every file you will actually open and write, in the order you will open it, with
its stub count in brackets. This is the bullet-point version of
[`ROADMAP.md`](ROADMAP.md); that file has the arithmetic and the reasoning, this
one has the list.

Hours are the **full** track. `C` marks a directory on `core`, `S` on `spine` —
if you are on a narrower track, skip the unmarked bullets and everything else
below stays in the same order. `python3 progress.py --track <yours>` is the
authority on pace; this file is the authority on **order**.

Three directories (`tensorrt-inference`, `vllm-engine`, `web-scraping`) are
design briefs with no templates: you write the files yourself against their
`IMPLEMENTATION_GUIDE.md`, so their bullets name phases rather than filenames.

---

## Phase 1 · AWS — week 1

### Week 1 · Aug 24–30 — `aws-from-scratch` **CS** 42 h
- `week-06/aws-from-scratch/` — order is fixed and the checker enforces it:
  - `iam.py` **[10]** — **always first**, everything else is gated by it
  - `s3.py` **[16]** → `sqs.py` **[8]** → `dynamodb.py` **[13]** → `lambda_svc.py` **[8]** → `sns.py` **[9]** → `kms.py` **[12]** → `vpc.py` **[14]** → `capstone.py` **[5]**
  - then the billing layer: `pricing.py` **[17]** → `billing.py` **[13]** → `optimize.py` **[15]**
  - **predict the provisioned-vs-on-demand DynamoDB crossover before running `optimize.py`**
  - stop condition: `python3 check.py` prints 24/24
- **Not this week:** `autograd/`. 24/24 first.

---

## Phase 2 · LLM work — weeks 2–6

### Week 2 · Aug 31–Sep 6 — `autograd` **CS** 30 h, `llm-from-scratch` start **CS**
- `week-07/autograd/` — the highest-leverage 30 hours in the repo:
  - `tensor.py` **[27]** — reverse mode over arrays, topological sort, gradient accumulation, `_unbroadcast`. Do not move on until every gradient matches central differences
  - `nn.py` **[13]** → `optim.py` **[7]** → `train.py` **[7]** → `generative.py` **[13]**
- `week-07/llm-from-scratch/`:
  - `tokenizer.py` **[10]** — BPE, merges applied in **rank order**
  - start `attention.py` if 10/10 on autograd is already green

### Week 3 · Sep 7–13 — `llm-from-scratch` **CS**
- `attention.py` **[4]** → `transformer.py` **[8]** → `train.py` **[4]** → `sample.py` **[9]**
- `distill.py` **[11]** — OPD, on vs off policy, forward/reverse KL/JSD, RL vs OPD vs SFT, OPSD, the papers, Privilege Illusion
- stop condition: `python3 check.py` prints 15/15

### Week 4 · Sep 14–20 — `context-caching` **CS** 28 h, start `inference-from-scratch` **C**
- `week-08/context-caching/`:
  - `tiny_transformer.py` **[10]** → `kv_cache.py` **[17]** — **do not move past `kv_cache.py` until cached and uncached attention agree to zero**, not to small
  - `prefix_cache.py` **[11]** → `radix_cache.py` **[12]** → `paged_kv_cache.py` **[13]** → `semantic_cache.py` **[12]** → `cache_router.py` **[13]** → `serving_demo.py` **[6]**
- `week-10/inference-from-scratch/inference_path.py` if 16/16 is green

### Week 5 · Sep 21–27 — `inference-from-scratch` steps 1–6 **C**
- **do not open vllm-engine yet**
  1. `inference_path.py` — what the GPU does for one prefill token and one decode token
  2. `naive_server.py` — make it work, then watch two requests fail as `single_flight`
  3. `batching.py` — continuous batching; measure TTFT, TPOT, throughput
  4. `kv_runtime.py` — decode becomes memory-bandwidth bound
  5. `scheduler.py` — queues, priorities, backpressure, cancellation, timeouts
  6. `paged_kv.py` — blocks, fragmentation, prefix sharing, CoW fork

### Week 6 · Sep 28–Oct 4 — `inference-from-scratch` steps 7–12 **C**, `deploy-and-debug` **CS**
  7. `gpu_opt.py` — CUDA graphs, fusion, quant, one sync per step
  8. `speculate.py` — when speculation is actually faster
  9. `observe.py` — TTFT, ITL, throughput, GPU util, KV, queue time
  10. `traffic.py` — find the concurrency where throughput stops scaling, and why
  11. `compare.py` — **now** read vLLM, SGLang, TensorRT-LLM
  12. `deeper.py` — multi-GPU, disagg, KV offload, cache-aware routing
  - stop condition: 12/12
- `week-08/deploy-and-debug/`: `capacity.py` **[10]** → `metrics.py` **[11]** → `diagnose.py` **[4]** → `rollout.py` **[9]**

---

## Phase 3 · Provenance-reasoning — weeks 7–10

An LLM answer is a proof, or it is not traceable. The parse is untrusted;
the prover is not. ContextCite comes *after* this, and answers a different
question (which tokens, not which derivation).

### Week 7 · Oct 5–11 — `provenance-semirings` **C** 45 h
- `week-08/provenance-semirings/` — `instance.py` is provided
  - `semiring.py` — the eight instances, one interface
  - `polynomial.py` — sparse ℕ[X]
  - `homomorphism.py` — `specialize`
  - `relation.py` → `ra.py` — σ, π, ⋈, ∪; the running query is ac + bd
  - **check 5 is the payoff:** `h(Q_How) = Q_K` for every K
  - `datalog.py` — absorptive K terminate; How on a cycle raises
  - stop condition: 8/8

### Week 8 · Oct 12–18 — `scasp` **C** 45 h
- `week-08/scasp/` — `program.py` is provided
  - `term.py` → `unify.py` (occurs on) → `sld.py` (even loop returns None)
  - `dual.py` — Clark duals, `neq`, De Morgan on conjuncts
  - `coinductive.py` — even ancestor succeeds, odd fails
  - `justify.py` — the tree; `atoms_used` is the lineage
  - stop condition: 8/8. opus flies; tweety does not

### Week 9 · Oct 19–25 — `linc` **C** 30 h
- `week-08/linc/` — `fixtures.py` is provided. No LLM.
  - `fol.py` → `parser.py` (`Parser` protocol + `GoldParser`)
  - `faults.py` — scope invert, drop ¬, arity drift, hallucinated const
  - `prover.py` — `{True, False, Uncertain, Error}` and a proof object
  - `pipeline.py` → `sweep.py` — predict the table, then run it
  - `trace.py` — how-polynomial of the proof; do week 7 first
  - stop condition: 8/8. Gold is 3/3; the sweep is monotone

### Week 10 · Oct 26–Nov 1 — citation layer **C**
- `week-08/contextcite/` 14/14 first
- `week-08/spade/` 8/8
- `week-08/mars-sql/` 8/8. Tokens and columns, after derivations

---

## Phase 4 · Distributed training, then a database — weeks 11–12

### Week 11 · Nov 2–8 — `distributed-training` **C** 10 h, pager **CS**
- `week-15/distributed-training/phase1_data_parallel/template_data_loader.py` **[4]** → `template_trainer.py` **[6]**
- `week-04/database-engine/pager.py` **[13]**. **This week and no further.**

### Week 12 · Nov 9–15 — `database-engine` **CS**
- `btree.py` → `wal.py` → `mvcc.py` → `sql.py` → `executor.py` → `planner.py` → `database.py`
- stop condition: 18/18

---

## Phase 5 · The rest — weeks 13–18

Compressed. Cut from here if you slip, not from Phase 3.

### Week 13 · Nov 16–22 — `dynamo-paper` **CS**, `raft` **C**
- preference lists and vector clocks first → quorum → handoff → merkle → gossip
- Raft: read Figure 8 before `replication.py`
- **Write the Raft-versus-Dynamo table in the journal**

### Week 14 · Nov 23–29 — `bash-from-scratch` **CS**, `http-server` start **CS**
- `shell.c`; HTTP accept / parse / static files / a real 404

### Week 15 · Nov 30–Dec 6 — finish HTTP, then the protocols
- keep-alive, `ab -n 10000 -c 100`
- DNS, SHA-256, UART → SPI → I2C → CAN

### Week 16 · Dec 7–13 — compilers **CS**
- `c-compiler`, `compiler-and-vgpu` 12/12
- full: firewall, toralizer, quantum

### Week 17 · Dec 14–20 — Haskell, CUDA **CS**, `ml-inference` **C**
- Haskell TODOs deleted; `nvidia-smi`; profile every kernel
- Quantisation: latency AND accuracy. Ray tracer if you have the Saturday

### Week 18 · Dec 21–27 — vendor engines, world models, the tail
- TensorRT, vLLM (re-read your `paged_kv.py`), world-models, diffusion
- five small directories. **Reserve the last day.** Every checker. Re-read the journal from 23 August

---

## Running alongside: `lean-proofs/` — every day, ~1.2 h

Not a week — a daily slot across all eighteen. Proofs go better in ninety-minute
pieces than in a marathon, and they keep an unrelated muscle warm while the rest
of the plan is C and CUDA. Roughly 28 `sorry`s a week.

The ladder to Galois, in order:

- `BasicLogic.lean` **[13]** — weeks 1
- `SetTheory.lean` **[31]** — weeks 1–2
- `NaturalNumbers.lean` **[29]** — weeks 2–3
- `Groups.lean` **[36]** — weeks 3–4
- `Rings.lean` **[70]** — weeks 5–7
- `Fields.lean` **[42]** — weeks 7–8
- `Polynomials.lean` **[47]** — weeks 8–10
- `FieldExtensions.lean` **[79]** — weeks 10–13
- `SplittingFields.lean` **[38]** — weeks 14–15
- `GaloisTheory.lean` **[31]** — weeks 15–16

And the analysis side track, if the algebra ladder stalls:

- `Limits.lean` **[19]**, `PartialDerivatives.lean` **[15]**, `BanachSpaces.lean` **[17]**, `NumberTheory.lean` **[37]**

---

## The stop conditions, in one place

| Directory | Command | Green means |
|---|---|---|
| `week-03/compiler-and-vgpu/` | `python3 check.py` | 12/12 |
| `week-04/database-engine/` | `python3 check.py` | 18/18 |
| `week-05/dynamo-paper/` | `python3 check.py` | 17/17 |
| `week-05/raft/` | `python3 check.py` | 7/7 |
| `week-06/aws-from-scratch/` | `python3 check.py` | 24/24 |
| `week-07/autograd/` | `python3 check.py` | 10/10 |
| `week-07/llm-from-scratch/` | `python3 check.py` | 15/15 |
| `week-08/deploy-and-debug/` | `python3 check.py` | 12/12 |
| `week-08/context-caching/` | `python3 check.py` | 16/16 |
| `week-08/provenance-semirings/` | `python3 check.py` | 8/8 |
| `week-08/scasp/` | `python3 check.py` | 8/8 |
| `week-08/linc/` | `python3 check.py` | 8/8 |
| `week-08/contextcite/` | `python3 check.py` | 14/14 |
| `week-08/spade/` | `python3 check.py` | 8/8 |
| `week-08/mars-sql/` | `python3 check.py` | 8/8 |
| `week-08/ray-tracer/` | `python3 check.py` | 6/6 |
| `week-10/inference-from-scratch/` | `python3 check.py` | 12/12 |

Everywhere else the stop condition is the file's own demo printing a table you
predicted first. All of it at once:

```bash
python3 progress.py --checks
```

---

[Roadmap](ROADMAP.md) · [Philosophy](PHILOSOPHY.md) · [Start at week 1](week-01/)
