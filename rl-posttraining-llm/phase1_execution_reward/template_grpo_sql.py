"""
Phase 1 template — GRPO on text-to-SQL with execution-only reward.

Fill in the TODOs, then compare with solutions/phase1_execution_reward/grpo_sql.py.
Run:  python template_grpo_sql.py
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from tiny_sql_env import SQLEnv, TASKS  # noqa: E402
from rewards import combine  # noqa: E402
from grpo import TabularPolicy, grpo_step  # noqa: E402
from candidates import CANDIDATES, CORRECT_INDEX, task_by_qid  # noqa: E402


def reward_of_factory(env, qids):
    def reward_of(prompt: int, action: int) -> float:
        qid = qids[prompt]
        completion = CANDIDATES[qid][action]
        # TODO: call combine(...) with weights={"exec": 1.0} and return the total
        raise NotImplementedError
    return reward_of


def train(steps: int = 200, seed: int = 0):
    rng = random.Random(seed)
    env = SQLEnv()
    qids = [t.qid for t in TASKS]
    n_actions = max(len(CANDIDATES[q]) for q in qids)
    policy = TabularPolicy(len(qids), n_actions, lr=0.7)
    reward_of = reward_of_factory(env, qids)

    for step in range(steps):
        # TODO: call grpo_step(policy, reward_of, None, prompts, group_size, rng)
        # TODO: every ~40 steps, print step, mean reward, and
        #       policy.probs(p)[CORRECT_INDEX[qids[p]]] for each prompt p.
        #       Watch q5 stall -> that's the execution-accuracy false positive.
        raise NotImplementedError


if __name__ == "__main__":
    train()
