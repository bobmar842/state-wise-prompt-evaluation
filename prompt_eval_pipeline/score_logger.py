"""
Score Logger
============
Aggregates per-turn evaluations into per-step scores.
Saves local JSON report + optional Langfuse push.
"""

import json
import os
from datetime import datetime
from collections import defaultdict

from .config import (
    LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST, OUTPUT_DIR,
)


def aggregate_and_log(
    turn_results: list[dict],
    conversation: dict,
    persona: dict,
    run_id: str,
) -> dict:
    """
    Aggregate turn-by-turn evaluations into per-step scores.

    Args:
        turn_results: list of per-turn results from the evaluate loop
        conversation: the simulated conversation
        persona: the persona used
        run_id: run identifier

    Returns:
        Report dict with per-step and overall scores
    """
    # Group turns by step
    step_scores = defaultdict(lambda: defaultdict(list))
    step_turns = defaultdict(list)

    for tr in turn_results:
        step = tr["step_id"]
        step_turns[step].append(tr["turn_index"])
        for dim, score in tr["scores"].items():
            step_scores[step][dim].append(score)

    # Aggregate per step
    per_step = {}
    for step, dims in step_scores.items():
        step_report = {"turns": step_turns[step]}
        dim_means = {}
        for dim, scores in dims.items():
            dim_means[dim] = round(sum(scores) / len(scores), 3) if scores else 0.0
        step_report["scores"] = dim_means
        step_report["overall"] = round(sum(dim_means.values()) / len(dim_means), 3) if dim_means else 0.0
        per_step[step] = step_report

    # Grand overall
    all_scores = []
    for step_report in per_step.values():
        all_scores.append(step_report["overall"])
    grand_overall = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

    report = {
        "run_id": run_id,
        "persona_id": persona.get("persona_id", "unknown"),
        "persona_name": persona.get("name_en", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "conversation_turns": conversation.get("turn_count", 0),
        "stop_reason": conversation.get("stop_reason", ""),
        "per_step_scores": per_step,
        "grand_overall": grand_overall,
        "turn_details": turn_results,
    }

    # Save locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pid = persona.get("persona_id", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f"report_{pid}_{ts}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Langfuse
    if LANGFUSE_ENABLED:
        _push_langfuse(report, run_id)

    return report


def save_run_summary(all_reports: list[dict], run_id: str) -> str:
    """Save a summary across all persona evaluations."""
    # Aggregate across all reports
    step_totals = defaultdict(lambda: defaultdict(list))
    for report in all_reports:
        for step, data in report.get("per_step_scores", {}).items():
            for dim, score in data.get("scores", {}).items():
                step_totals[step][dim].append(score)

    summary_steps = {}
    for step, dims in step_totals.items():
        summary_steps[step] = {
            dim: {
                "mean": round(sum(scores) / len(scores), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
                "count": len(scores),
            }
            for dim, scores in dims.items()
        }

    all_overalls = [r["grand_overall"] for r in all_reports]
    grand_mean = round(sum(all_overalls) / len(all_overalls), 3) if all_overalls else 0.0

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "num_conversations": len(all_reports),
        "grand_overall_mean": grand_mean,
        "per_step_summary": summary_steps,
        "per_persona": [
            {
                "persona": r["persona_name"],
                "overall": r["grand_overall"],
                "steps": {s: d["overall"] for s, d in r["per_step_scores"].items()},
            }
            for r in all_reports
        ],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{run_id}_summary.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return filepath


def _push_langfuse(report, run_id):
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        trace = client.trace(
            name=f"eval_{report['persona_id']}",
            metadata={"run_id": run_id, "persona": report["persona_name"]},
            tags=["prompt_eval", run_id],
        )
        for step, data in report.get("per_step_scores", {}).items():
            for dim, score in data.get("scores", {}).items():
                client.score(trace_id=trace.id, name=f"{step}/{dim}", value=score)
            client.score(trace_id=trace.id, name=f"{step}/overall", value=data["overall"])
        client.score(trace_id=trace.id, name="grand_overall", value=report["grand_overall"])
        client.flush()
    except Exception as e:
        print(f"    [warn] Langfuse push failed: {e}")
