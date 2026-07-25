"""
Phase 4 template — execution-free graph-matching reward.

Score a predicted query against the gold query using only their STRUCTURE, never
running them. Validate against the q5 candidates that execution reward cannot
distinguish. Fill in the TODOs.
Run:  python template_graph_reward.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from tiny_sql_env import TASKS  # noqa: E402
from rewards import extract_sql  # noqa: E402
from candidates import CANDIDATES  # noqa: E402


AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(([^)]*)\)", re.IGNORECASE)
TABLE_RE = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)


def to_graph(sql: str) -> dict[str, set]:
    """Extract a typed component graph from a SQL string.

    Returns e.g. {"tables": {...}, "aggs": {...}, "filters": {...}}.
    """
    sql = sql.lower()
    tables = set(TABLE_RE.findall(sql))
    aggs = {f"{fn.lower()}({arg.strip()})" for fn, arg in AGG_RE.findall(sql)}
    # TODO: extract filter predicates (WHERE ... comparisons) into a set
    # TODO: extract join edges (pairs of tables joined) into a set
    return {"tables": tables, "aggs": aggs}  # extend with filters/joins


def weighted_jaccard(a: dict[str, set], b: dict[str, set],
                     weights: dict[str, float]) -> float:
    # TODO: for each component type, compute Jaccard(a[t], b[t]); return the
    #       weight-normalized sum. Handle empty sets (define 0/0 = 1.0).
    raise NotImplementedError


def graph_reward(completion: str, gold_sql: str,
                 weights=None) -> float:
    weights = weights or {"tables": 0.3, "aggs": 0.5, "filters": 0.2}
    return weighted_jaccard(to_graph(extract_sql(completion)),
                            to_graph(gold_sql), weights)


if __name__ == "__main__":
    # Goal: rank q5's correct candidate (index 0) first WITHOUT executing.
    gold = next(t for t in TASKS if t.qid == "q5").gold_sql
    print("q5 candidates ranked by execution-free graph reward:")
    scored = [(graph_reward(c, gold), i) for i, c in enumerate(CANDIDATES["q5"])]
    for score, i in sorted(scored, reverse=True):
        print(f"  cand[{i}] graph_reward={score:.3f}")
    print("Success = cand[0] on top, achieved with zero data access.")
