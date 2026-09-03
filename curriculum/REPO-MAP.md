# Repository map — 34 project directories against six curriculum blocks

Which directory to build after which block, and — the part usually left out — **which
directories no block covers at all**.

## The headline: the curriculum covers about half this repository

All six blocks are machine learning. Roughly half of these directories are systems,
languages, networking, formal methods and finance, and **no course on the list touches
them**. That is not a defect in either the curriculum or the repository; it is worth
stating so you do not finish the six blocks believing you have worked through the repo.

---

## Covered by the curriculum

### Block 0 — prerequisites (build these first)

| Directory | Why here |
|---|---|
| `cuda-from-scratch/` | Kernels from vector add to a full network. **Before CS336 assignment 2**, not during |
| `compiler-and-vgpu/` | SIMT execution: warps, divergence, mask stacks, barrier deadlock. 12 graded checks |

### Block 1 — CS229 + Géron

| Directory | Why here |
|---|---|
| `quantitative-trading/` | Non-stationary data with unambiguous feedback. The best antidote to CS229's implication that models are well-understood estimators |
| `spectral-graphs/` | Spectral methods; closest to mathematics you already have |
| `ml-in-production/` | Serving, monitoring, A/B testing |
| `mlops/` | Experiment tracking, CI/CD, feature stores |

### Block 2 — CS224N + NLP with Transformers

| Directory | Why here |
|---|---|
| `contextcite/` | ContextCite from scratch. 14 graded checks. Requires holding a real LM and reasoning about ablated-context scores |
| `sgl-lang/` | Structured generation, grammar enforcement, constrained decoding |

### Block 3 — CS230 + Goodfellow (demoted; see the block file)

| Directory | Why here |
|---|---|
| `diffusion-models/` | DDPM, DDIM, U-Net. The direct answer to Goodfellow Part III being obsolete: you build what replaced it |
| `world-models/` | Five papers, World Models through DreamerV3 and IRIS. The best PyTorch-fluency route in the repo |
| `deepfake-creation/`, `deepfake-detection/` | Generative and forensic methods; optional |

### Block 4 — CS336 + Raschka (the spine — maps onto more of this repo than any other)

| Directory | When |
|---|---|
| `cuda-from-scratch/`, `compiler-and-vgpu/` | **Before** assignment 2 |
| `distributed-training/` | **Alongside** assignment 2 — data and model parallelism, multi-node |
| `web-scraping/` | **Alongside** assignment 4 — crawling, parsing, rate limiting at scale |
| `context-caching/` | **After** assignment 1. 16 graded checks: KV cache, prefix and radix caching, paged KV with copy-on-write, cache-aware routing |
| `vllm-engine/` | After `context-caching/`. PagedAttention, continuous batching |
| `ml-inference/` | Inference optimisation, quantization, edge |
| `tensorrt-inference/` | Graph optimisation, kernel auto-tuning |

### Block 5 — CME295 + Hugging Face

| Directory | Why here |
|---|---|
| `deploy-and-debug/` | 12 graded checks: capacity maths, percentiles, error budgets, root-cause diagnosis of 11 injected faults, safe rollout |
| `ml-inference/` | Quantization, revisited with CME295's empirical framing |

### Block 6 — CS329A + Hur & Song

| Directory | Why here |
|---|---|
| `system-design/` | The reliability substrate an agent runs on: caching, queues, circuit breakers, backpressure, idempotency |
| `aws-from-scratch/` | IAM policy evaluation, in `iam.py`, if you build the authorisation layer rather than delegating it |
| `dynamo-paper/` | Vector clocks, quorums, anti-entropy. 17 graded checks |

---

## Not covered by any block

These are the other half of the repository. Nothing in CS229 → CS329A goes near them.

### Systems and languages

`c-compiler/` · `bash-from-scratch/` · `quantum-computing-lang/` · `haskell-projects/` ·
`toralizer/`

### Networking

`http-server/` · `dns-server/` · `firewall-from-scratch/` · `communication-protocols/`

### Security and formal methods

`cryptographic-library/` · `lean-proofs/` (Lean 4, toward the Fundamental Theorem of
Galois Theory — the one directory here that is straightforwardly in your existing
mathematical territory)

### Data engineering

`sas-lineage-tool/`

**If you want a second curriculum for this half**, the repository's own
[README](../README.md) has a "Video Courses & Learning Resources" section covering
operating systems, compilers, cryptography, networking and formal verification. It is
not organised into blocks the way `curriculum/` is; that would be a separate piece of
work.

---

## On other branches

Two directories exist on branches that are not merged into `main`:

| Directory | Branch | Relates to |
|---|---|---|
| `compression-lower-bounds/` | `claude/compression-lower-bounds-cuxikk` | Block 4 — rate–distortion theory under quantization, and a separate computational-lower-bounds track |
| `enterprise-ai-projects/` | `claude/enterprise-ai-project-guides` | Blocks 5 and 6 — twelve enterprise AI project guides: multi-tenant RAG, PII proxy, MCP over legacy ERPs, shadow traffic evaluation |

---

## Graded directories

Directories shipping `check.py`, which is where feedback is sharpest:

| Directory | Checks |
|---|---|
| `aws-from-scratch/` | 18 |
| `dynamo-paper/` | 17 |
| `context-caching/` | 16 |
| `contextcite/` | 14 |
| `compiler-and-vgpu/` | 12 |
| `deploy-and-debug/` | 12 |

Start with a graded one. A directory that tells you which invariant you broke is worth
three that do not.
