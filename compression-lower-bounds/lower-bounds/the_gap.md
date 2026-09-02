# Step 5 — why steps 1–4 give no lower bound for the model you care about

<!-- UNWRITTEN -->

**This step has no code. Delete the HTML comment on the line above, then write one page below.**
`check.py` checks that the marker is gone and that there is prose under each heading. It
cannot check whether the prose is any good; that is the point of writing it rather than
answering a quiz.

Write it with all four implementations closed. If you need to reopen one, note which —
that is the part you did not actually understand.

---

## 1. What each of the four bounds is a statement about

For each of steps 1–4, in one sentence: **the model**, **the quantity bounded**, and
**the class of objects quantified over**. Be exact about the last one. Three of the four
quantify over algorithms in a restricted model; one quantifies over schedules of a single
fixed algorithm, which is a much weaker statement than it first reads as.

> _your answer_

## 2. The transformer forward pass

Take the object you actually care about: a forward pass of a decoder-only transformer,
or one layer of it, on a real machine. Now try to apply each of the four bounds to it and
say precisely where each one fails to apply.

Some of the failures are about the model (the computation is not comparison-based, not
bilinear, not a single polynomial). At least one is about the *quantification*: the bound
holds but says nothing you did not already know. Distinguish those two kinds of failure —
they have different consequences for whether more work would help.

> _your answer_

## 3. The one bound that survives

One of the four does transfer, in modified form, and is used in practice to decide things
like tile sizes and whether an operation is memory-bound. Name it, state what it gives
you for a matmul of the shapes in an actual attention layer, and say what it does **not**
give you.

> _your answer_

## 4. Why the obvious approach to a real bound cannot work

You now want a lower bound on the number of operations needed to compute the function a
trained transformer computes. Before reading anything, write down the approach you would
take. Then read Razborov–Rudich (*Natural proofs*, JCSS 1997) and say which of the two
properties — constructivity and largeness — your approach has, and what that implies.

Then look at Williams' ACC⁰ result (JACM 2014) and note what a hard-won modern lower
bound actually covers. Compare the class it applies to with the class your transformer
lives in.

> _your answer_

## 5. The asymmetry with the other track

The compression track has a *tight* limit — Shannon's `D(R) = σ²2^(-2R)`, achieved in the
limit — and every step of it measures a known fraction of a known gap. This track has
`Ω(n log n)`, from 1983, for explicit polynomials, and open problems everywhere else.

Say what is structurally different about the two fields that produces this. It is not that
one is harder in the informal sense. Rate–distortion theory bounds a quantity defined by
an *average over a known distribution*; complexity bounds a *worst case over all inputs*
for an object that has to be *explicitly constructed*. Develop that, and say which of the
two situations you think the question "how few bits per weight" actually lives in.

> _your answer_

## 6. What you got wrong

Which prediction in this directory did you get wrong before implementing it? Be specific.
If the answer is "none", you did not write the predictions down in advance.

> _your answer_
