# 1 — CS229 + Géron, *Hands-On Machine Learning*

| | |
|---|---|
| **Course** | Stanford CS229, *Machine Learning* — <https://cs229.stanford.edu/> |
| **Book** | Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, **3rd edition**, O'Reilly, November 2022, 861 pp., ISBN 9781098125974 |

## Does the pairing work

Yes. This is the best-constructed pair on the list. CS229 is derivation-heavy — you get
the exponential family, the GLM construction, the SVM dual, the EM derivation — and Géron
is a practitioner's book with minimal theory and working code. They fail in opposite
directions, which is what a pairing is for.

Use the 3rd edition specifically. It is the one updated for the modern Keras API and it
does reach transformers and diffusion models; the 2nd edition does not.

## What to skip, given who you are

CS229's mathematical scaffolding is your home ground and reading it is a slow way to learn
nothing. Target these instead — they are the parts where CS229 is teaching *ML judgement*
rather than mathematics:

- **Generative vs discriminative**, and why GDA and logistic regression relate the way they
  do. This is the cleanest statement in the course of a distinction that recurs everywhere.
- **The bias–variance decomposition as a practical diagnostic**, not as an identity. You
  know the identity. What is new is using it to decide what to do next on a Tuesday.
- **Regularisation as MAP.** You will find this obvious and it is worth five minutes to
  confirm the correspondence, because half the field's vocabulary assumes it.
- **The learning-theory lectures** (uniform convergence, VC dimension). Mathematically
  satisfying, practically inert — modern deep networks violate every hypothesis. Read them
  and then note that they explain nothing about why a 70B model generalises. That gap is
  real and unresolved, and being clear-eyed about it early prevents a lot of confusion later.

From Géron, skip nothing in Part I. The chapters on end-to-end project structure, data
preparation and evaluation are the parts a mathematician most reliably underrates, and
they are the ones that determine whether your results mean anything.

## The trap

CS229's version of ML is pre-deep-learning in its emphasis, and it is easy to leave it
believing that the field is a collection of well-understood estimators with derivable
properties. It is not, and blocks 4 onward will be disorienting if you arrive expecting
that. The honest framing: CS229 teaches the part of ML that has theory, which is a
shrinking fraction of the part of ML that works.

## Build after this block

```
spectral-graphs/          spectral methods; closest to your existing mathematics
quantitative-trading/     statistical arbitrage and ML strategies, on data that fights back
ml-in-production/         model serving, monitoring, A/B testing
```

`quantitative-trading/` is the one to do. It supplies the thing CS229 cannot: a domain
where your model is wrong, the data is non-stationary, and the feedback is unambiguous.
