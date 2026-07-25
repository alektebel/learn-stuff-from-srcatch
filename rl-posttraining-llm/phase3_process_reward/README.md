# Phase 3 — Process reward models (step-wise reasoning supervision)

**Goal:** move the reward from the *final answer* to the *reasoning trace*.
Reward-SQL trains a **Process Reward Model (PRM)** that scores each intermediate
step of the chain-of-thought that builds a query, giving a much denser signal
than any outcome reward — and one that is robust to the execution-accuracy false
positive you found in Phase 1 (the *reasoning* to a coincidentally-right answer
is still visibly wrong).

## Concepts

- **Outcome reward model (ORM)** scores the final query. **Process reward model
  (PRM)** scores each step: "given the steps so far, is this next step a correct
  move toward the goal?"
- **Where the labels come from.** Two standard recipes:
  1. *Monte-Carlo roll-outs* (Math-Shepherd style): a step is "good" if
     completing from it reaches the correct answer often. Fully automatic —
     you already have an executor to check the endpoint.
  2. *Human/LLM step annotations*: cheaper to bootstrap, noisier.
- **Using the PRM in RL.** Two options: (a) per-step reward shaping — add the
  PRM score of each step to the GRPO reward; (b) best-of-N / step-level beam
  search at inference. Start with (a).

## Exercises

1. **Define the step schema.** Decompose SQL construction into steps, e.g.
   `link_schema → choose_tables → build_joins → add_filters → add_aggregation →
   finalize`. Represent a trace as an ordered list of steps.
2. **Auto-label with MC roll-outs.** For each prefix of a trace, sample K
   completions with the tabular policy and use `execution_accuracy` at the end
   as the label. This reuses the Phase 1 executor — no new supervision needed.
3. **Train a PRM.** On CPU, a logistic-regression-style scorer over
   hand-features of `(prefix, step)` is enough to feel the mechanism. Features:
   does the step introduce a gold table? a spurious table? a needed filter?
4. **Shape the GRPO reward.** `reward = w_exec·exec + w_prm·mean(prm_step_scores)`.
   Show it distinguishes the `q5` candidates that execution reward could not,
   because the *reasoning* toward the wrong metric is visibly wrong.

## `template_prm.py`

Skeleton with the step schema, MC-labeling loop, a tiny feature-based scorer,
and the reward-shaping hook. Reuses `common/` throughout.

## Real-model track

Generate traces with the base model (`<think>` steps then `<sql>`), auto-label
step prefixes by MC roll-out + executor, fine-tune a small regression head on
the base model's hidden states (or a separate small model) as the PRM, then add
its score to the TRL reward. Compare PRM-shaped GRPO vs outcome-only GRPO on
multi-join queries.

## Done when

You have an auto-labeled PRM (no manual step labels) and PRM-shaped GRPO beats
outcome-only GRPO on the `q5`-style ambiguous cases.
