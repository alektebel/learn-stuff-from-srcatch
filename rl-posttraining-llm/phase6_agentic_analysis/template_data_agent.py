"""
Phase 6 template — a multi-step data-analysis agent with process rewards and a
self-improving (EvoDS-style) data loop.

Fill in the TODOs. Run:  python template_data_agent.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from tiny_sql_env import SQLEnv  # noqa: E402


class AnalysisEpisode:
    """One multi-step analysis: a sequence of (query, result, reasoning) steps
    ending in an answer. Unlike Phase 5 there is no single gold SQL."""

    def __init__(self, question: str, verifier):
        self.question = question
        self.verifier = verifier   # callable(answer) -> float in [0,1]
        self.steps: list[dict] = []

    def add_step(self, query: str, env: SQLEnv, reasoning: str) -> None:
        ok, res = env.execute(query)
        self.steps.append({"query": query, "result": res if ok else None,
                            "ok": ok, "reasoning": reasoning})

    def finalize(self, answer) -> float:
        return self.verifier(answer)


def process_reward(episode: AnalysisEpisode) -> float:
    """Score whether each step was justified by the data so far.

    TODO: reward a query that is motivated by the previous result; reward
    reasoning that references returned numbers; penalize errored/aimless steps.
    Reuse the Phase 3 PRM idea. Return mean step score.
    """
    raise NotImplementedError


def make_verifier(gold_answer):
    """For checkable questions. For open ones, swap in an LLM-judge rubric."""
    def verify(answer) -> float:
        # TODO: exact / tolerant match to gold_answer -> 1.0 else 0.0
        raise NotImplementedError
    return verify


# --- EvoDS-style self-improving data loop ---------------------------------- #
def generate_candidate_questions(env: SQLEnv, n: int) -> list[str]:
    """Agent proposes new questions over the schema.
    TODO: template or model-generate questions; keep them schema-grounded."""
    raise NotImplementedError


def self_improve_round(policy, env: SQLEnv):
    """One EvoDS round:
        1. generate candidate questions
        2. solve each with the current policy (produce an AnalysisEpisode)
        3. validate (executor / verifier); keep hard-but-solvable ones
        4. add validated traces to the training set
        5. run GRPO (process_reward + final verification) on the augmented set
    TODO: implement; measure held-out accuracy before vs after the round.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("Build the multi-step episode, process reward, verifier, and EvoDS loop.")
    print("Reward = w_proc * process_reward(episode) + w_ans * episode.finalize(answer)")
