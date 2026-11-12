"""Offline labels for observed rollouts, not a runtime safety controller.

One JSON object per episode; see docs/research/safety_research_plan.md for the contract.
Uses only the Python standard library. Future horizons are measured in actions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(episodes, horizons=(4, 8, 16, 32)):
    horizons = sorted(set(horizons))
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons):
        raise ValueError("horizons must be positive integers")
    seen, labels, summaries = set(), [], []
    for episode in episodes:
        identity = episode["episode_id"]
        task = episode["task"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("episode_id must be a unique nonempty string")
        if not isinstance(task, str) or not task:
            raise ValueError("task must be a nonempty string")
        seen.add(identity)
        complete, success = episode["complete"], episode["task_success"]
        if type(complete) is not bool or (success is not None and type(success) is not bool):
            raise ValueError("complete must be bool; task_success must be bool or null")
        if (complete and success is None) or (not complete and success is not None):
            raise ValueError("only complete episodes may have a success outcome")
        events = episode["events"]
        if not isinstance(events, list) or not events:
            raise ValueError("events must contain one list per observed action")
        for step in events:
            if (not isinstance(step, list)
                    or any(not isinstance(e, str) or not e.strip() for e in step)
                    or len(set(step)) != len(step)):
                raise ValueError("each step requires unique, nonempty violation names")
        violations = [bool(step) for step in events]
        # events[t] describes the interval AFTER decision t / execution of action t.
        for t in range(len(events)):
            future = {}
            for horizon in horizons:
                observed = violations[t:t + horizon]
                future[str(horizon)] = (
                    1 if any(observed) else 0 if len(observed) == horizon else None
                )
            labels.append({"episode_id": identity, "task": task,
                           "decision_step": t, "future_violation": future})
        summaries.append({
            "episode_id": identity, "task": task, "complete": complete,
            "task_success": success,
            "observed_steps": len(events), "violation_steps": sum(violations),
            "observed_any_violation": any(violations),
            "safe_success": bool(success and not any(violations)) if complete else None,
            "event_step_counts": dict(Counter(e for step in events for e in step)),
        })
    if not summaries:
        raise ValueError("no episodes")
    by_task = defaultdict(list)
    for summary in summaries:
        by_task[summary["task"]].append(summary)
    per_task = {}
    for task, rows in sorted(by_task.items()):
        completed = [row for row in rows if row["complete"]]
        n = len(completed)
        per_task[task] = {
            "episodes": len(rows), "complete_episodes": n,
            "completion_coverage": n / len(rows),
            "success_rate": sum(r["task_success"] for r in completed) / n if n else None,
            "safe_success_rate": sum(r["safe_success"] for r in completed) / n if n else None,
            "violation_episode_rate": sum(r["observed_any_violation"] for r in completed) / n if n else None,
            "observed_violation_step_rate": sum(r["violation_steps"] for r in rows) / sum(r["observed_steps"] for r in rows),
        }
    return {"schema_version": 1, "label_scope": "observed_executed_policy_only",
            "horizons_actions": horizons, "episodes": summaries,
            "per_task": per_task, "labels": labels,
            "warning": "Not official RoboDojo scoring. Null future labels are censored; never treat them as safe. Incomplete episodes can bias rates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizons", nargs="+", type=int, default=[4, 8, 16, 32])
    args = parser.parse_args()
    with args.input.open() as handle:
        episodes = [json.loads(line) for line in handle if line.strip()]
    report = analyze(episodes, args.horizons)
    # Refuse to overwrite an experiment's existing result.
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
