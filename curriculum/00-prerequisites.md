# 0 — Prerequisites, and what a mathematician skips

## The honest version

If you have measure theory, linear algebra, probability and numerical analysis, then most
of what these six courses call "prerequisites" is material you already own, taught more
slowly than you would like. The prerequisite you are actually missing is different, and
naming it correctly saves you two months.

**What you already have and can skip outright:**

- Goodfellow Part I (ch. 2–5): linear algebra, probability, numerical computation. This is
  a review chapter for people who did not do a maths degree.
- CS229's probability and linear algebra review sections and their handouts.
- Every "here is what a gradient is" segment in every course on the list.
- Convex optimisation framing — you will find the ML treatment less careful than what you
  know, not more.

**What you do not have, and no amount of mathematics substitutes for:**

1. **Engineering intuition for the empirical loop.** Not "what is SGD" but: how you tell a
   bug from a bad hyperparameter, what a broken loss curve looks like versus a converging
   one, why your reproduction is 3% off. This is not derivable. It is acquired by running
   things and being wrong.
2. **PyTorch as a fluent language.** Shapes, broadcasting, devices, `.detach()`, what
   `autograd` does and does not track, why your memory is full. CS336 assignment 1
   forbids `torch.nn.Linear`, which is the fastest way to acquire this and the reason that
   assignment exists.
3. **Systems.** GPU memory hierarchy, kernel launch overhead, what "memory-bound" means,
   why an operation that is 10× fewer FLOPs is not 10× faster. CS336 assignment 2 (Flash
   Attention 2 in Triton) requires this and does not teach it.
4. **The literature's conventions.** What "we train for 300B tokens" implies about compute,
   what a reported perplexity means and on what, why everyone reports MMLU and why that is
   a problem. This is cultural knowledge and it is acquired by reading papers, not courses.

Items 2 and 3 are the real prerequisites, and neither CS229 nor CS230 provides them.

## What to do about it

**For PyTorch fluency**, the fastest route is not a tutorial. It is `world-models/` and
`diffusion-models/` in this repository, which are template-based and will fail on you in
the specific ways that teach shape and device discipline. Two weekends.

**For systems**, `cuda-from-scratch/` in this repository — kernels from vector addition
through a complete neural network, plus `compiler-and-vgpu/` for the SIMT execution model
(warps, divergence, mask stacks). Do this **before** CS336, not during. Assignment 2 is
brutal without it and pleasant with it.

Confidence note: I am stating that CS229 and CS230 do not teach items 2 and 3 based on
their published syllabi, which are about ML content rather than GPU systems. If a recent
offering added a systems component, this section is out of date — check the syllabus.

## The maths that is genuinely new

Short list, because it is short:

- **Information theory as an engineering tool.** Cross-entropy as the training objective,
  perplexity as its exponential, KL as the regulariser in every alignment method. You know
  the definitions; what is new is that these are *quantities people tune*.
- **Rate–distortion**, if you go near quantization. Covered properly in the
  `compression-lower-bounds/` track on its own branch.
- **Scaling laws.** Empirical power laws in loss versus compute, and the IsoFLOP method for
  fitting them. CS336 assignment 3. The mathematics is elementary; the epistemology — what
  it means to fit a curve to seven runs and extrapolate three orders of magnitude — is not,
  and is worth your attention as someone trained to be careful about that.

## Build before block 1

```
cuda-from-scratch/        kernels, memory hierarchy, occupancy
compiler-and-vgpu/        SIMT: warps, divergence, mask stacks, barriers
```

Both have graded checks. If `compiler-and-vgpu/check.py` passes, you have the systems
model that CS336 assumes.
