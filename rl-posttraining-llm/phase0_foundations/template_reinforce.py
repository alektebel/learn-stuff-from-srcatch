"""
Phase 0 template — implement REINFORCE, then add the GRPO baseline.

Fill in the TODOs. Then compare against solutions/phase0_foundations/.
Pure stdlib; run with `python template_reinforce.py`.
"""
from __future__ import annotations

import math
import random


class SoftmaxPolicy:
    def __init__(self, n_prompts: int, n_actions: int, lr: float = 0.5):
        self.n_prompts, self.n_actions, self.lr = n_prompts, n_actions, lr
        self.logits = [[0.0] * n_actions for _ in range(n_prompts)]

    def probs(self, p: int) -> list[float]:
        # TODO: numerically-stable softmax over self.logits[p]
        raise NotImplementedError

    def sample(self, p: int, rng: random.Random) -> int:
        # TODO: sample an action from probs(p)
        raise NotImplementedError


def group_relative_advantages(rewards: list[float]) -> list[float]:
    # TODO: return (r - mean) / (std + 1e-6) for each reward
    raise NotImplementedError


def train(use_baseline: bool, steps: int = 150, seed: int = 0):
    rng = random.Random(seed)
    correct = [1, 3, 0]
    pol = SoftmaxPolicy(3, 4)
    reward_of = lambda p, a: 1.0 if a == correct[p] else 0.0  # noqa: E731

    for _ in range(steps):
        grad = [[0.0] * pol.n_actions for _ in range(pol.n_prompts)]
        for p in range(pol.n_prompts):
            acts = [pol.sample(p, rng) for _ in range(8)]
            rews = [reward_of(p, a) for a in acts]
            # TODO: advantages = group_relative_advantages(rews) if use_baseline
            #       else the raw rewards (plain REINFORCE)
            # TODO: accumulate grad[p][j] += adv * (1[j==a] - probs[j])
            raise NotImplementedError
        # TODO: gradient ascent step on pol.logits using grad and pol.lr
    return pol


if __name__ == "__main__":
    train(use_baseline=False)
    train(use_baseline=True)
