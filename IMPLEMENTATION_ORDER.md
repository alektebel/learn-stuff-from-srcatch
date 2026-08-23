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

## Phase 1 · Systems in C — weeks 1–3

### Week 1 · Aug 24–30 — `bash-from-scratch` **CS** 8 h, `http-server` **CS** 71 h
- `week-01/bash-from-scratch/shell.c` **[18]** — tokenising, `fork`/`exec`/`wait`, pipes, redirection, signal handling, exit statuses
- `week-01/http-server/http_server.c` **[16]**, in this order and no other:
  - the accept loop — `socket`, `bind`, `listen`, `accept`
  - request line and header parsing
  - static file serving with the right `Content-Type`
  - a real 404 rather than a hang
- **Not this week:** keep-alive, chunked encoding, concurrency. Week 2.

### Week 2 · Aug 31–Sep 6 — `http-server` 15 h **CS**, `dns-server` 9 h, `cryptographic-library` 5 h, `communication-protocols` 34 h, `toralizer` 16 h
- `week-01/http-server/http_server.c` — finish it: concurrency and keep-alive, then hold it under `ab -n 10000 -c 100`
- `week-02/dns-server/dns_server.c` **[20]** — UDP sockets, the DNS wire format, A and CNAME records, TCP retry on a truncated response
- `week-02/cryptographic-library/sha256.c` **[12]** — padding, the message schedule, the compression function
- `week-02/communication-protocols/` — framing first, the buses that build on it after:
  - `uart.c` **[22]**
  - `spi.c` **[15]**
  - `i2c.c` **[26]**
  - `can.c` **[23]**
  - `rs232_485.c` **[29]**
- `week-02/toralizer/network.c` **[10]**, then `socks.c` **[19]** — the SOCKS5 handshake

### Week 3 · Sep 7–13 — `toralizer` 5 h, `firewall-from-scratch` 25 h, `c-compiler` **CS** 27 h, `compiler-and-vgpu` **CS** 16 h, `quantum-computing-lang` 8 h
- `week-02/toralizer/toralizer.c` **[17]** — finish Monday, then close it
- `week-03/firewall-from-scratch/firewall.c` **[68]** — raw sockets, packet parsing, rule matching
- `week-03/c-compiler/`, and get `int main(){return 2+3;}` running end to end on day one:
  - `lexer.c` **[9]**
  - `parser.c` **[41]**
  - `semantic.c` **[42]**
  - `codegen.c` **[33]** — linear-scan register allocation and spilling
  - `optimizer.c` **[24]**
- `week-03/compiler-and-vgpu/` — one 32-bit ISA, two execution models:
  - `isa.py` **[3]** → `assembler.py` **[9]** → `cpu.py` **[6]** → `frontend.py` **[12]** → `codegen.py` **[12]** → `vgpu.py` **[8]** → `capstone.py` **[3]**
  - stop condition: `python3 check.py` prints 12/12
- `week-03/quantum-computing-lang/quantum.c` **[17]** — an interpreter over complex amplitudes

---

## Phase 2 · Storage, consensus and the cloud — weeks 4–6

### Week 4 · Sep 14–20 — `haskell-projects` 61 h, `database-engine` **CS** 16 h
- `week-04/haskell-projects/`:
  - `JSONParser.hs` **[15]** — parser combinators, and the payoff of the whole directory
  - `Calculator.hs` **[14]**
  - `BuildTool.hs` **[22]**
  - `WebScraper.hs` **[16]**
  - delete each `-- TODO` as you satisfy it, or the bars never move
- `week-04/database-engine/pager.py` **[13]** — slotted pages, free-space arithmetic, the buffer pool and its LRU. **This week and no further.** A B+tree on a shaky pager is a week you lose.

### Week 5 · Sep 21–27 — `database-engine` **CS** 34 h, `dynamo-paper` **CS** 21 h, `raft` **C** 24 h
- `week-04/database-engine/`, and the checker's 18 steps are the order:
  - `btree.py` **[12]**
  - `wal.py` **[10]** — **before** `mvcc.py`. Durability is one writer and a disk; isolation is several writers and each other
  - `mvcc.py` **[12]**
  - `sql.py` **[15]** — recursive descent, precedence climbing
  - `executor.py` **[19]** — the Volcano iterator model
  - `planner.py` **[20]**
  - `database.py` **[13]**
- `week-05/dynamo-paper/`, with the SOSP 2007 paper open:
  - `partitioning.py` **[13]** and `vector_clock.py` **[11]** first — everything else assumes them
  - `quorum.py` **[8]** → `hinted_handoff.py` **[7]** → `merkle_sync.py` **[8]** → `gossip.py` **[10]** → `dynamo_cluster.py` **[13]**
- `week-05/raft/`, and read Figure 8 before you write `replication.py`:
  - `log.py` **[9]** → `election.py` **[7]** → `replication.py` **[8]**

### Week 6 · Sep 28–Oct 4 — `raft` **C** 6 h, `system-design` 48 h, `aws-from-scratch` **CS** 25 h
- `week-05/raft/cluster.py` **[6]** — safety asserted after every operation, against a network that partitions and crashes
- **Then write the Raft-versus-Dynamo table in `LOG.md`** before moving on. Same partition, opposite answers.
- `week-06/system-design/` — 18 small files, roughly a day per group:
  - caching: `cache.py` **[12]**, `cdn_caching.py` **[7]**, `data_locality.py` **[12]**
  - queues: `message_queue.py` **[8]**, `idempotency_keys.py` **[7]**, `request_batching.py` **[12]**
  - reliability: `circuit_breaker.py` **[9]**, `throttling.py` **[7]**, `rate_limiter.py` **[4]**, `connection_pooling.py` **[6]**
  - distribution: `consistent_hash.py` **[5]** (do it without looking at last week's, then diff), `hot_partition_mitigation.py` **[8]**, `eventual_consistency.py` **[13]**, `session_stickiness.py` **[11]**, `edge_computing.py` **[10]**
  - applications: `url_shortener.py` **[9]**, `leaderboard.py` **[14]**, `capacity_planner.py` **[5]**
- `week-06/aws-from-scratch/` — order is fixed and the checker enforces it:
  - `iam.py` **[10]** — **always first**, everything else is gated by it
  - `s3.py` **[16]** → `sqs.py` **[8]** — checks 1 to 7

---

## Phase 3 · Gradients, transformers and light — weeks 7–9

### Week 7 · Oct 5–11 — `aws-from-scratch` **CS** 17 h, `autograd` **CS** 30 h, `llm-from-scratch` **CS** 32 h
- `week-06/aws-from-scratch/`, to 24/24 by Tuesday:
  - `dynamodb.py` **[13]** → `lambda_svc.py` **[8]** → `sns.py` **[9]** → `kms.py` **[12]** → `vpc.py` **[14]** → `capstone.py` **[5]**
  - then the billing layer: `pricing.py` **[17]** → `billing.py` **[13]** → `optimize.py` **[15]**
  - **predict the provisioned-vs-on-demand DynamoDB crossover before running `optimize.py`**
- `week-07/autograd/` — the highest-leverage 30 hours in the repo:
  - `tensor.py` **[27]** — reverse mode over arrays, topological sort, gradient accumulation, `_unbroadcast`. Do not move on until every gradient matches central differences
  - `nn.py` **[13]** → `optim.py` **[7]** → `train.py` **[7]** → `generative.py` **[13]**
- `week-07/llm-from-scratch/`:
  - `tokenizer.py` **[10]** — BPE, merges applied in **rank order**
  - `engine.py` **[2]** → `attention.py` **[4]** → `transformer.py` **[8]**

### Week 8 · Oct 12–18 — `llm-from-scratch` **CS** 13 h, `deploy-and-debug` **CS** 10 h, `context-caching` **CS** 28 h, `contextcite` 13 h, `ray-tracer` 15 h
- `week-07/llm-from-scratch/train.py` **[4]** → `sample.py` **[9]**
- `week-08/deploy-and-debug/`: `capacity.py` **[10]** → `metrics.py` **[11]** → `diagnose.py` **[4]** → `rollout.py` **[9]**
- `week-08/context-caching/`:
  - `tiny_transformer.py` **[10]** → `kv_cache.py` **[17]** — **do not move past `kv_cache.py` until cached and uncached attention agree to zero**, not to small
  - `prefix_cache.py` **[11]** → `radix_cache.py` **[12]** → `paged_kv_cache.py` **[13]** → `semantic_cache.py` **[12]** → `cache_router.py` **[13]** → `serving_demo.py` **[6]**
- `week-08/contextcite/`: `partition.py` **[5]** → `ablation.py` **[5]** → `logit_probs.py` **[7]** → `lasso.py` **[7]** → `contextcite.py` **[8]** → `evaluate.py` **[5]** → `applications.py` **[5]**
- `week-08/ray-tracer/vec.py` **[13]** → `shapes.py` **[8]**
- **Sunday:** install the CUDA toolkit. Do not spend a Monday fighting a driver.

### Week 9 · Oct 19–25 — `ray-tracer` 20 h, `cuda-from-scratch` **CS** 59 h
- `week-08/ray-tracer/bvh.py` **[5]** → `material.py` **[7]** → `render.py` **[8]**
  - **predict what the noise does from 16 to 64 samples before you read the table**
- `week-09/cuda-from-scratch/`, strictly in order, profiling every kernel:
  - `01_vector_addition.cu` **[16]** → `02_matrix_addition.cu` **[15]** → `03_matrix_multiplication.cu` **[16]** → `04_reduction.cu` **[5]** → `05_convolution.cu` **[7]**
  - start the table now: kernel, GB/s achieved, GB/s theoretical, occupancy

---

## Phase 4 · GPUs and inference — weeks 10–15

### Week 10 · Oct 26–Nov 1 — `cuda-from-scratch` **CS** 63 h, `ml-inference` **C** 16 h
- `week-09/cuda-from-scratch/06_neural_network_forward.cu` **[9]** → `07_neural_network_backward.cu` **[8]** → `08_complete_neural_network.cu` **[29]**
- `week-10/ml-inference/` — Phase 1, basic inference: a forward pass, a latency harness, a baseline you will beat for the next five weeks
- `week-10/ml-inference/phase2_optimization/template_quantization.py` **[6]** — start here in Phase 2, because everything downstream depends on knowing what int8 costs

### Week 11 · Nov 2–8 — `ml-inference` **C** 79 h
- `week-10/ml-inference/` Phase 2 — pruning, distillation, operator fusion, the rest of quantisation
- `week-10/ml-inference/` Phase 3 — advanced serving, and **dynamic batching is the thing to spend real time on**; it is the same throughput-versus-latency trade vLLM will hand you from the other direction
- re-read your own `week-08/context-caching/kv_cache.py` when you reach the serving material
- deliverable: a latency-versus-throughput curve at several batch sizes, with the knee identified

### Week 12 · Nov 9–15 — `ml-inference` **C** 42 h, `tensorrt-inference` 37 h
- `week-10/ml-inference/phase4_production/` — the full serving system, monitoring, and the production harness
- `week-12/tensorrt-inference/` against its `IMPLEMENTATION_GUIDE.md`:
  - Phase 1, core infrastructure — the engine, the builder, the runtime
  - Phase 2, graph optimisations — layer fusion, constant folding, dead-layer elimination
  - for each optimisation, one line on whether you could have done it by hand in week 10

### Week 13 · Nov 16–22 — `tensorrt-inference` 72 h, `vllm-engine` **C** 7 h
- `week-12/tensorrt-inference/`:
  - Phase 3, mixed precision — fp16 and INT8 calibration
  - Phase 4, advanced features, then benchmark against your own week-11 server on the same model and hardware
  - the deliverable is a speedup **attributed to named optimisations**, not to the brand
- `week-13/vllm-engine/` — read Phase 1 and sketch the block table on paper. Re-read `week-08/context-caching/paged_kv_cache.py` first; you have already built the core idea at small scale

### Week 14 · Nov 23–29 — `vllm-engine` **C** 79 h
- `week-13/vllm-engine/` Phase 1 — PagedAttention: the block allocator and per-sequence block table. **Forking a sequence must cost zero additional blocks**
- Phase 2 — continuous batching, prefill and decode interleaved across sequences in one forward pass
- Phase 3 — model parallelism
- assert on day one, and hold it all week: **a cache that changes the output is not a cache, it is a bug**

### Week 15 · Nov 30–Dec 6 — `vllm-engine` **C** 70 h, `distributed-training` 10 h
- `week-13/vllm-engine/` Phase 4 (quantisation), Phase 5 (advanced optimisations), Phase 6 (production serving)
- `week-15/distributed-training/phase1_data_parallel/template_data_loader.py` **[4]** → `template_trainer.py` **[6]**
  - you already wrote gradient accumulation in `week-07/autograd/`; this is the same idea with a network in the middle
- **Sunday:** open `week-16/world-models/` and start the VAE

---

## Phase 5 · Generative models and the long tail — weeks 16–18

### Week 16 · Dec 7–13 — `world-models` 78 h
- `week-16/world-models/common/`: `env_wrapper.py` **[12]** → `replay_buffer.py` **[9]** → `metrics.py` **[9]** → `video.py` **[7]**
- `week-16/world-models/paper1_world_models/`: `vae.py` **[8]** → `rnn.py` **[8]** → `controller.py` **[13]** → `train.py` **[7]** → `eval.py` **[10]**
  - measure the distance from a random `N(0, I)` draw to the nearest latent the model has seen. You measured exactly this in `autograd/generative.py`
- `week-16/world-models/paper2_dreamerv1/`: `rssm.py` **[14]** → `networks.py` **[13]** → `actor_critic.py` **[11]** → `buffer.py` **[11]** → `train.py` **[13]**

### Week 17 · Dec 14–20 — `world-models` 28 h, `diffusion-models` 51 h
- `week-16/world-models/paper3_dreamerv2/`: `rssm.py` **[11]** → `networks.py` **[8]** → `actor_critic.py` **[7]** → `train.py` **[7]**
- `week-16/world-models/paper4_dreamerv3/`: `symlog.py` **[2]** → `world_model.py` **[11]** → `actor_critic.py` **[8]** → `train.py` **[3]**
- `week-16/world-models/paper5_iris/`: `tokenizer.py` **[10]** → `transformer.py` **[8]** → `actor_critic.py` **[6]** → `train.py` **[5]**
- **three sentences before you move on:** what v2 fixed in v1, what v3 fixed in v2
- `week-17/diffusion-models/diffusion.py` **[22]** — the forward process first, verified **analytically** against iterated single-step noising before you train anything
- `week-17/diffusion-models/unet.py` **[19]**

### Week 18 · Dec 21–27 — `diffusion-models` 40 h, then five small directories
- `week-17/diffusion-models/train.py` **[11]** → `sample.py` **[11]** — DDPM, then DDIM, then classifier-free guidance
  - sweep the guidance scale and watch the diversity collapse. The noise you are fighting is the same 1/√N you measured in `ray-tracer`
- `week-18/spectral-graphs/spectral_graphs.py` **[11]**
- `week-18/sas-lineage-tool/template_lineage_parser.py` **[23]**
- `week-18/web-scraping/` — against its `IMPLEMENTATION_GUIDE.md`: fetch, parse, queue, politeness, the distributed layer
- `week-18/ml-in-production/phase1_basic_serving/template_model_server.py` **[7]**
- `week-18/mlops/phase1_tracking/template_experiment_logger.py` **[13]**
- **Reserve the last day.** Run every checker in the repo, re-read `LOG.md` from week 1, write the closing entry.

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
| `week-07/llm-from-scratch/` | `python3 check.py` | 8/8 |
| `week-08/deploy-and-debug/` | `python3 check.py` | 12/12 |
| `week-08/context-caching/` | `python3 check.py` | 16/16 |
| `week-08/contextcite/` | `python3 check.py` | 14/14 |
| `week-08/ray-tracer/` | `python3 check.py` | 6/6 |

Everywhere else the stop condition is the file's own demo printing a table you
predicted first. All of it at once:

```bash
python3 progress.py --checks
```

---

[Roadmap](ROADMAP.md) · [Philosophy](PHILOSOPHY.md) · [Start at week 1](week-01/)
