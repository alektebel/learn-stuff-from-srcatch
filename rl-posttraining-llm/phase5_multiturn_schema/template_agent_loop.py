"""
Phase 5 template — a tool-integrated, multi-turn text-to-SQL agent with GRPO.

The schema is HIDDEN from the prompt. The agent must discover it via tools.
Reward lands on the final query; you optionally shape per turn.
Fill in the TODOs. Run:  python template_agent_loop.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from tiny_sql_env import SQLEnv, TASKS, SCHEMA_TABLES, SCHEMA_COLUMNS  # noqa: E402
from rewards import execution_accuracy  # noqa: E402


class ToolEnv:
    """Wrap SQLEnv so the schema is only reachable through tool calls."""

    def __init__(self):
        self.db = SQLEnv()

    def list_tables(self) -> str:
        return "tables: " + ", ".join(SCHEMA_TABLES)

    def describe(self, table: str) -> str:
        cols = SCHEMA_COLUMNS.get(table)
        return f"{table}({', '.join(cols)})" if cols else f"unknown table {table}"

    def sample(self, table: str, n: int = 2) -> str:
        ok, res = self.db.execute(f"SELECT * FROM {table} LIMIT {n}")
        return str(res) if ok else f"error: {res}"

    def run(self, sql: str) -> str:
        ok, res = self.db.execute(sql)
        return str(res) if ok else f"error: {res}"


# Tools = the action space. A trajectory is a sequence of these.
TOOLS = ["list_tables", "describe", "sample", "run", "final"]


def rollout(policy, task, env: ToolEnv, max_turns: int = 6, rng=None):
    """Produce one trajectory: list of (turn_state, action) plus final query.

    TODO:
      - maintain a context of observations
      - each turn, ask `policy` for a tool (+arg) conditioned on turn state
      - execute the tool, append the observation
      - stop on 'final' or max_turns; return trajectory + final SQL
    """
    raise NotImplementedError


def turn_shaping(trajectory, task) -> float:
    """Optional dense signal (MARSQL flavor).

    TODO: reward describing a gold table; reward a revision that fixes an error;
          small negative step penalty. Return the summed shaping bonus.
    """
    raise NotImplementedError


def trajectory_reward(trajectory, final_sql, task, env: ToolEnv) -> float:
    exec_r = execution_accuracy(f"<sql>{final_sql}</sql>", task, env.db)
    # TODO: return exec_r (+ turn_shaping) as the trajectory reward
    raise NotImplementedError


def train():
    """
    TODO: for each question, sample G trajectories, compute trajectory rewards,
    standardize within the group (group_relative_advantages from common/grpo.py),
    and update the turn-conditioned policy. This is Phase-0 GRPO where the
    'action' is an entire trajectory.
    """
    raise NotImplementedError


if __name__ == "__main__":
    env = ToolEnv()
    print(env.list_tables())
    print(env.describe("orders"))
    print(env.sample("orders"))
    print("Implement rollout + trajectory GRPO to train the schema-discovery agent.")
