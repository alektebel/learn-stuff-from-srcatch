"""
Phase 2 template — phased rewards + a reward curriculum.

Fill in the TODOs, then compare with
solutions/phase2_phased_rewards/partial_rewards.py.
Run:  python template_partial_rewards.py
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from tiny_sql_env import SQLEnv, TASKS  # noqa: E402
from rewards import combine, execution_accuracy  # noqa: E402
from grpo import TabularPolicy, grpo_step  # noqa: E402
from candidates import CANDIDATES, task_by_qid  # noqa: E402


def curriculum_weights(progress: float) -> dict[str, float]:
    # TODO: shaping = max(0, 1 - progress); exec_w = 0.5 + 1.5*progress
    # TODO: return the five component weights (see README)
    raise NotImplementedError


def greedy_exec_acc(env, qids, policy) -> float:
    # TODO: for each prompt, take argmax action, sum execution_accuracy, divide
    raise NotImplementedError


def train(steps: int = 200, seed: int = 0, use_curriculum: bool = True):
    rng = random.Random(seed)
    env = SQLEnv()
    qids = [t.qid for t in TASKS]
    n_actions = max(len(CANDIDATES[q]) for q in qids)
    policy = TabularPolicy(len(qids), n_actions, lr=0.7)

    for step in range(steps):
        progress = step / max(steps - 1, 1)
        # TODO: weights = curriculum_weights(progress) if use_curriculum
        #       else a fixed phased-weights dict
        # TODO: define reward_of(p, a) = combine(CANDIDATES[qids[p]][a], task, env, weights)[0]
        # TODO: grpo_step(...); periodically print mean reward + greedy_exec_acc
        raise NotImplementedError


if __name__ == "__main__":
    train(use_curriculum=False)
    train(use_curriculum=True)
