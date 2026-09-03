# 5 — CME295 + the Hugging Face LLM Course

| | |
|---|---|
| **Course** | Stanford CME295, *Transformers & Large Language Models* — <https://cme295.stanford.edu/>. Afshine Amidi and Shervine Amidi. Lectures on YouTube; cheatsheets and a study guide at <https://github.com/afshinea/stanford-cme-295-transformers-large-language-models> |
| **Course 2** | Hugging Face LLM Course — <https://huggingface.co/learn/llm-course> |

## Does the pairing work

As a **reference pair**, yes. As a *block*, no — and the distinction matters for how you
spend the month.

CME295 covers: transformers (tokenization, embeddings, attention, architecture), LLM
foundations (mixture of experts, decoding strategies), training and tuning (supervised and
reinforcement finetuning, LoRA), evaluation, tricks (RoPE, attention approximation,
quantization), reasoning (train-time and test-time scaling, context awareness), and agentic
workflows (RAG, tool calling). Its stated prerequisites are calculus, linear algebra and
basic ML.

Read that list against block 4. There is very little in it that CS336 does not do in more
depth and with an assignment attached. The exception is breadth: CME295 surveys MoE, RoPE,
decoding strategies, and evaluation in a compact form that CS336 does not stop to
summarise.

**So use it as a reference consulted during block 4, not as a course taken before it.**
The Amidis' cheatsheets are the artefact here — they are genuinely good at compressing a
topic to one page, which is what you want when CS336 assumes RoPE and you need the
definition in ninety seconds rather than a lecture.

The Hugging Face course is the practical complement: the library, the ecosystem, the
`datasets`/`transformers`/`peft`/`trl` surface. It is a manual for tools you will use,
and it is best read the way you read a manual — when you need something.

## The genuinely distinctive part

**Evaluation.** CME295 gives it real time, and it is the topic most under-taught across
this entire list. CS336 has you build a model; almost nothing in the six blocks teaches
you to establish whether it is good. Take that section seriously, and note that
benchmark-driven evaluation has failure modes — contamination, construct validity, the
gap between a leaderboard number and a working product — that the material will not fully
dwell on.

## The trap

This block is where "I have watched a lot of lectures about LLMs" becomes indistinguishable
from "I can build one". It is the third pass over attention. If you find yourself watching
a lecture on the attention mechanism for the third time in this curriculum, stop — that is
the signal that this should be a reference and not a course.

## Build after this block

```
sgl-lang/                 structured generation, grammar enforcement, compilation
ml-inference/             quantization, edge deployment
deploy-and-debug/         12 graded checks: capacity maths, percentiles, error budgets,
                          root-cause diagnosis of 11 injected faults, safe rollout
```

And, on other branches:

```
compression-lower-bounds/compression/     the rate-distortion theory under quantization —
                                          CME295 tells you quantization works; this tells
                                          you what the floor is and why
enterprise-ai-projects/11-shadow-evaluator.md   evaluation as a production practice rather
                                                than a benchmark score
```

If evaluation is the distinctive part of CME295, then the shadow-traffic guide is its
natural build: it is the same question asked where the answer has consequences.
