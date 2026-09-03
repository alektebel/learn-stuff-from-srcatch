# 2 — CS224N + Tunstall, von Werra & Wolf, *NLP with Transformers*

| | |
|---|---|
| **Course** | Stanford CS224N, *Natural Language Processing with Deep Learning* — <https://web.stanford.edu/class/cs224n/>. Winter 2025-26 instructors: Tatsunori Hashimoto and Diyi Yang |
| **Book** | Lewis Tunstall, Leandro von Werra, Thomas Wolf, *Natural Language Processing with Transformers*, **Revised Edition**, O'Reilly, July 2022, 406 pp., ISBN 9781098136796 |

## Does the pairing work

Yes, and the book is written by three of the people who built Hugging Face Transformers,
so it is the library's own account of itself — useful and worth knowing as you read it.

But note the date: **July 2022**. It is a fine-tuning-era book. Its centre of gravity is
BERT-style encoders, task-specific heads, and classification / NER / QA — which is what
NLP looked like before instruction-tuned decoders ate the field. The chapters on making
models efficient for deployment (distillation, quantization, pruning) have aged well; the
task-specific fine-tuning workflow has aged less well.

That is not a reason to skip it. It is a reason to read it for the *mechanics* (tokenizers,
the `datasets`/`transformers` API surface, how a training loop is actually assembled) and
not as a description of current practice.

## The part of CS224N you actually need — and it is the front half

This is the argument from the [curriculum README](README.md): blocks 2, 4 and 5 all teach
attention, and doing all three linearly means learning it three times.

CS224N's distinctive contribution is the part the other two skip: **the history**.

- Word vectors, word2vec, GloVe, and what a distributional representation is.
- Sequence-to-sequence with RNNs, and the specific failure that made attention necessary:
  a fixed-size bottleneck vector between encoder and decoder.
- Attention introduced as a **fix for that bottleneck** — which is what it originally was,
  and is the only framing under which "attention" is a natural name rather than a slogan.
- Subword tokenization and why it exists.

CS336 and CME295 both start from "here is a transformer" and never explain what it
replaced. If you skip this half, attention will always feel like an arbitrary construction
that happened to work.

**Stop when the course reaches the pretraining and LLM lectures.** Block 4 does that
better and in more depth.

## The trap

Reading the book as a to-do list of tasks to fine-tune models on. Its chapter structure —
classification, NER, QA, summarisation — invites this, and it is how the field was
organised in 2022. Reproducing all of them teaches you the `Trainer` API and not much else.
Do one end to end, carefully, and move on.

## Build after this block

```
contextcite/              14 graded checks. ContextCite from scratch: attribution by
                          ablating context sources and fitting a sparse LASSO surrogate
sgl-lang/                 structured generation, grammar enforcement, constrained decoding
```

`contextcite/` is the right one. It requires you to hold a real language model, score
sequences under ablated contexts, and reason about what the scores mean — which exercises
the whole block without being another fine-tuning exercise.
