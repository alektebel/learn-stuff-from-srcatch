# Curriculum — six course-and-book blocks

Six pairings, each a Stanford course with a book that covers the same ground from the
other direction. Every block file says what the course actually is, what the book
actually is, **whether the pairing works**, what to skip, and which directory in this
repository to build once you have finished.

| # | Course | Book | Pairing |
|---|---|---|---|
| 0 | — | — | [Prerequisites, and what a mathematician skips](00-prerequisites.md) |
| 1 | CS229 Machine Learning | Géron, *Hands-On ML*, 3rd ed. | [Good](01-cs229-homl.md) — theory and practice, correctly opposed |
| 2 | CS224N NLP with Deep Learning | Tunstall, von Werra, Wolf, *NLP with Transformers* | [Good](02-cs224n-nlp-transformers.md) — but overlaps blocks 4 and 5 |
| 3 | CS230 Deep Learning | Goodfellow, Bengio, Courville, *Deep Learning* | [**Weak**](03-cs230-goodfellow.md) — see below |
| 4 | CS336 Language Modeling from Scratch | Raschka, *Build a Large Language Model (From Scratch)* | [Excellent](04-cs336-raschka.md) — and the one that matters |
| 5 | CME295 Transformers & LLMs | Hugging Face LLM Course | [Good as reference](05-cme295-huggingface.md), weak as a course-shaped thing |
| 6 | CS329A Self-Improving AI Agents | Hur & Song, *Build an AI Agent (From Scratch)* | [**Mismatched**](06-cs329a-agents.md) — the book is far below the course |

[**REPO-MAP.md**](REPO-MAP.md) maps all 34 project directories of this repository onto these six
blocks: what to build after what, and which directories no block covers.

---

## Read this before starting block 1

### Three honest problems with this list

**1. Block 3 is the weak one, and it is weak in both directions.**

Goodfellow, Bengio & Courville was published by MIT Press in **November 2016**.
*Attention Is All You Need* is **2017**. The book therefore contains no transformers at
all — not a light treatment, none — and blocks 2, 4, 5 and 6 of this curriculum are
entirely about transformers.

Meanwhile CS230 is the deeplearning.ai specialization with a project wrapper: Coursera
videos, quizzes, programming assignments, a midterm, and a final project worth 40% of the
grade. Its sequence-models section is largely RNN/LSTM-era.

So you have a practical course whose architecture coverage predates the field's current
architecture, paired with a theory book that predates it too. Neither compensates for the
other. Block 3's file says exactly which parts of each are still worth your time — the
answer is "less than half of each" — and what to read instead.

**2. Blocks 2, 4 and 5 cover overlapping ground.**

CS224N is now substantially transformer-and-LLM focused. CME295 is *entirely* transformers
and LLMs. CS336 has you build one. Running all three linearly means learning attention
three times.

The structure that is not wasteful:

- **CS224N** for the *pre-transformer history* — word vectors, seq2seq, the attention
  mechanism as a fix for a specific bottleneck in encoder-decoder RNNs. This is where the
  intuition for *why* attention won lives, and it is the one thing CS336 and CME295 both
  assume rather than teach.
- **CS336** for the engineering. This is the load-bearing block.
- **CME295** as a **reference**, not a course. Its cheatsheets and study guide are what it
  is best at; sitting through it as a third pass over attention is not a use of a month.

**3. Block 4 is far larger than the other five.**

CS336's five assignments, for scale:

1. BPE tokenizer, Transformer, and Adam, from PyTorch primitives — you may not call
   `torch.nn.Transformer` or even `torch.nn.Linear`. Train on TinyStories and OpenWebText.
2. Flash Attention 2 in Triton, plus distributed data parallel and optimizer sharding.
3. Fit scaling laws with IsoFLOP.
4. Common Crawl HTML → text: quality filtering, harmful-content filtering, PII removal,
   deduplication.
5. Alignment: SFT, expert iteration, GRPO variants, RL on Qwen 2.5 Math 1.5B.

That is a semester on its own, and assignment 2 alone assumes GPU access and CUDA
competence the other blocks never ask for. Do not plan block 4 as "one of six".

### The order to actually use

Not 1 → 6. This one:

```
0  prerequisites            skip most of it, see the file
1  CS229 + HOML             targeted, not complete — you are a mathematician
2  CS224N (history half)    stop at the transformer lectures
4  CS336 + Raschka          the spine. Months, not weeks.
5  CME295 + HF course       as reference, consulted during 4, not before
3  CS230 + Goodfellow       only the chapters block 3's file names, and only if a gap showed up
6  CS329A + Hur & Song      last, because agents are built on everything above
```

Block 3 is demoted rather than dropped because its surviving parts — Goodfellow Part II
on optimization and regularization, CS230's "structuring ML projects" week — are genuinely
useful and are not covered anywhere else in the list. They are just not a *block*.

### What "finishing" a block means here

Not watching the lectures. Every block file ends with a **build**: a directory in this
repository, or a piece of one. Per [`PHILOSOPHY.md`](../PHILOSOPHY.md), the point is that
you get it wrong and find out precisely how. A course you completed and did not build from
is a course you will not be able to use.
