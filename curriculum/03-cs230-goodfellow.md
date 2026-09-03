# 3 — CS230 + Goodfellow, Bengio & Courville, *Deep Learning*

| | |
|---|---|
| **Course** | Stanford CS230, *Deep Learning* — <https://cs230.stanford.edu/>. Andrew Ng and Kian Katanforoosh |
| **Book** | Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, MIT Press, **November 2016** — free at <https://www.deeplearningbook.org/> |

## Does the pairing work

**No, and this is the block to demote.** The reasoning is arithmetic, not taste.

Goodfellow was published in **November 2016**. *Attention Is All You Need* is **2017**.
The book contains **no transformer coverage at all** — not thin coverage, none — and
blocks 2, 4, 5 and 6 of this curriculum are entirely about transformers.

CS230, meanwhile, is the deeplearning.ai specialization with a Stanford project wrapper:
Coursera videos, quizzes, programming assignments, a midterm (25%) and a final project
(40%). Its coverage is CNNs, RNNs, LSTM, Adam, dropout, batch norm, Xavier/He
initialisation — good material, and a sequence-models section from the pre-transformer era.

So the practical course and the theory book are dated in the *same* direction. Normally a
pairing works because the two halves fail differently. This one does not.

There is a second mismatch: CS230's mathematical level is well below yours, and
Goodfellow's Part I is a maths review you do not need. You would be spending a block on
material that is simultaneously too easy and too old.

## What survives, and it is worth having

Do not skip these — they are genuinely good and are covered nowhere else on the list:

**From Goodfellow:**
- **Ch. 7, Regularization.** Still the best single treatment of why regularisation works,
  including the equivalence of early stopping and L2 for the quadratic case, and dataset
  augmentation as a prior.
- **Ch. 8, Optimization for Training Deep Models.** Ill-conditioning, plateaus, cliffs,
  the actual reason batch normalisation helps, and why the optimisation problem is not
  the convex one you are used to. This chapter has not been superseded.
- **Ch. 9, Convolutional Networks.** The invariance/equivariance framing is cleaner here
  than in most later sources.
- **Ch. 11, Practical Methodology.** Short, and the closest the book comes to
  transmitting judgement.

**Skip Part I entirely** (ch. 2–5 are the maths review) and **skip Part III** (ch. 16–20:
structured probabilistic models, Monte Carlo methods, the partition function, approximate
inference, deep generative models). Part III is about restricted Boltzmann machines, deep
belief networks and Boltzmann machines. It is intellectually interesting and it is
history — the generative-model lineage that actually won runs through VAEs, GANs,
autoregressive transformers and diffusion, and the book predates most of that.

**From CS230:** the *Structuring Machine Learning Projects* module (roughly week 5).
Error analysis, train/dev/test mismatch, when to change the metric versus the model,
human-level performance as a reference point. It is the least mathematical thing on the
list and one of the most useful, and it exists nowhere else in this curriculum.

## What to read instead of the rest

- For modern architectures with the same textbook care: Zhang, Lipton, Li & Smola,
  *Dive into Deep Learning* — <https://d2l.ai/> — which does cover attention and
  transformers, and is free and executable.
- For the theory that has appeared since 2016: Prince, *Understanding Deep Learning*
  (MIT Press, 2023), free at <https://udlbook.github.io/udlbook/>.

Confidence note: I am recommending both from their scope and reputation. I have not read
either cover to cover recently enough to vouch for specific chapters, and you should
sample before committing a month.

## Where this block belongs

**After block 4, not before, and only if a gap appeared.** Do CS336 first. When you hit
something in it you cannot diagnose — a training run that will not converge, a
regularisation choice you cannot justify — come back and read the specific Goodfellow
chapter that addresses it. That is what this book is now: an excellent reference, not a
course.

## Build after this block

```
diffusion-models/         DDPM, DDIM, U-Net — the generative lineage that actually won,
                          which is precisely what Goodfellow Part III does not contain
distributed-training/     data and model parallelism, multi-node
```

Building `diffusion-models/` is the direct answer to Part III being obsolete: you
implement the thing that replaced it.
