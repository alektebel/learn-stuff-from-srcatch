# 4 — CS336 + Raschka, *Build a Large Language Model (From Scratch)*

| | |
|---|---|
| **Course** | Stanford CS336, *Language Modeling from Scratch* — <https://stanford-cs336.github.io/spring2025/>. Tatsunori Hashimoto and Percy Liang, with Marcel Röd, Neil Band, Rohith Kudipudi. Lectures on YouTube, no enrolment required |
| **Book** | Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024 — <https://www.manning.com/books/build-a-large-language-model-from-scratch> |

## Does the pairing work

Yes, and it is the best block on the list — but they are not equals and treating them as
equals wastes the course.

**Raschka's book is the ramp.** It walks you through one GPT-style model end to end in
PyTorch: tokenizer, attention, the block, pretraining loop, fine-tuning for classification,
instruction fine-tuning. It is careful, incremental, and correct. It is also *small* — the
models are toy-scale and the book does not go near distributed training, kernels, data
pipelines at scale, or RL.

**CS336 is the real thing.** Liang's own framing of why it exists: researchers have become
detached from the technical details of how language models work, and the fix is to have
students build everything.

Read Raschka first, in about two weeks. Then do CS336, which is months.

## The scale of CS336, stated plainly

Five assignments:

| # | What |
|---|---|
| 1 | BPE tokenizer, Transformer architecture, Adam optimizer — **from PyTorch primitives only**. No `torch.nn.Transformer`, no `torch.nn.Linear`. Train on TinyStories and OpenWebText |
| 2 | **Flash Attention 2 in Triton**, plus distributed data parallel and optimizer sharding |
| 3 | Fit **scaling laws** using IsoFLOP |
| 4 | Common Crawl **HTML → text**: quality filtering, harmful-content filtering, PII removal, deduplication |
| 5 | **Alignment**: supervised fine-tuning, expert iteration, GRPO variants, RL on Qwen 2.5 Math 1.5B |

Assignment 2 assumes GPU access and CUDA competence that nothing else in this curriculum
provides. This is the reason `cuda-from-scratch/` is in
[block 0](00-prerequisites.md) rather than here — arriving at Triton without a mental
model of the memory hierarchy turns a hard assignment into an impossible one.

Assignment 4 is the one people underrate. It is not glamorous and it is where most of the
actual quality of a model comes from, which is a lesson you can only learn by doing it.

## What is distinctive about assignment 3, for you specifically

Fitting scaling laws is elementary mathematics — a power law, a few runs, a fit. The
interesting part is epistemological, and it is one you are unusually equipped to be
careful about: you fit a curve to a handful of training runs and extrapolate three orders
of magnitude in compute. Ask what licenses that. Ask what the confidence interval on the
extrapolation is, and whether anyone reports it. Ask what would falsify the law.

That scepticism is a contribution, not an obstacle. Most of the field treats these curves
as more solid than the evidence supports.

## The trap

Reading the CS336 lectures and doing Raschka's book, and calling that the block. The
lectures are excellent and watching them produces the feeling of understanding without any
of it. The assignments are the course. If you do only one thing here, do assignment 1
without `torch.nn.Linear`.

## Build after — and during — this block

This block maps onto more of this repository than any other:

```
cuda-from-scratch/        BEFORE assignment 2. Not optional.
compiler-and-vgpu/        BEFORE assignment 2. SIMT, warps, divergence, mask stacks.
distributed-training/     ALONGSIDE assignment 2. Data and model parallelism, multi-node.
context-caching/          AFTER assignment 1. 16 graded checks: KV cache, prefix caching,
                          paged KV blocks with copy-on-write, cache-aware routing.
vllm-engine/              AFTER context-caching. PagedAttention, continuous batching.
ml-inference/             quantization and inference optimisation.
tensorrt-inference/       graph optimisation, kernel auto-tuning.
web-scraping/             ALONGSIDE assignment 4. Crawling, parsing, rate limiting at scale.
```

And on the other branch: `compression-lower-bounds/compression/` is the rate–distortion
foundation under the quantization that `ml-inference/` and CME295 both treat empirically.
Do it after assignment 1 if quantization interests you.
