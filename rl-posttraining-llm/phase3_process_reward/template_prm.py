"""
Phase 3 template — a Process Reward Model, auto-labeled by Monte-Carlo roll-outs.

No new supervision: you reuse the Phase 1 executor to label whether a reasoning
*prefix* tends to lead to a correct query. Fill in the TODOs.
Run:  python template_prm.py
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from tiny_sql_env import SQLEnv, TASKS, SCHEMA_TABLES  # noqa: E402
from rewards import execution_accuracy, extract_sql, _identifiers  # noqa: E402


# A "trace" here is a list of step strings. In the real setting these are the
# model's <think> steps; on CPU you can synthesize them from candidate queries.
STEP_KINDS = ["link_schema", "choose_tables", "build_joins", "add_filters",
              "add_aggregation", "finalize"]


def step_features(prefix: list[str], step: str, task) -> list[float]:
    """Hand-features for a (prefix, step). Replace with model hidden states in
    the real-model track."""
    idents = _identifiers(" ".join(prefix + [step]))
    gold = {t.lower() for t in task.gold_tables} | {c.lower() for c in task.gold_columns}
    spurious = (idents & {t.lower() for t in SCHEMA_TABLES}) - gold
    # TODO: return a small feature vector, e.g.
    #   [fraction of gold identifiers introduced so far,
    #    number of spurious tables,
    #    1.0 if an aggregation keyword present else 0.0, ...]
    raise NotImplementedError


def mc_label(prefix_query: str, task, env: SQLEnv, k: int, rng) -> float:
    """Monte-Carlo: complete `prefix_query` k ways, return fraction correct.
    On CPU, 'completion' can be appending random plausible clauses; the point is
    the LABEL comes from the executor, not from a human."""
    # TODO: build k completions from prefix_query, score each with
    #       execution_accuracy, return the mean.
    raise NotImplementedError


class PRM:
    """A tiny linear scorer: score(prefix, step) in [0,1]."""
    def __init__(self, dim: int):
        self.w = [0.0] * dim
        self.b = 0.0

    def score(self, feats: list[float]) -> float:
        z = self.b + sum(wi * fi for wi, fi in zip(self.w, feats))
        return 1.0 / (1.0 + pow(2.718281828, -z))

    def fit_step(self, feats, label, lr=0.1):
        # TODO: one logistic-regression gradient step toward `label`
        raise NotImplementedError


def shaped_reward(completion: str, task, env, prm: PRM, w_exec=1.0, w_prm=0.5):
    exec_r = execution_accuracy(completion, task, env)
    # TODO: decompose completion into steps, average prm.score over them,
    #       return w_exec*exec_r + w_prm*mean_prm
    raise NotImplementedError


if __name__ == "__main__":
    print("Implement the TODOs, then plug shaped_reward into the Phase 2 GRPO loop.")
