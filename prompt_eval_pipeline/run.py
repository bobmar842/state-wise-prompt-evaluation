"""
Main Orchestrator
=================
Step 1: Simulate full conversation (full prompt, full flow)
Step 2: Walk through turn-by-turn:
        a) CLASSIFY — which step of the prompt is this turn?
        b) EVALUATE — score this turn against that step's rules
        c) REMEMBER — accumulate under that step
Step 3: Aggregate per-step scores → report

Modes:
  Normal:    simulate + evaluate
  Eval-only: skip simulation, re-evaluate existing conversation files

Usage:
  python -m prompt_eval_pipeline.run --prompt prompt.txt --transcripts ./transcripts/
  python -m prompt_eval_pipeline.run --prompt prompt.txt --personas personas.json
  python -m prompt_eval_pipeline.run --prompt prompt.txt --eval-only output/conversations/
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .config import (
    DEFAULT_NUM_PERSONAS, NUM_SIMULATIONS_PER_PERSONA,
    PERSONAS_FILE, OUTPUT_DIR, MAX_EVAL_WORKERS,
)


# ─────────────────────────────────────────────────────────
# Core evaluation logic (shared by normal and eval-only)
# ─────────────────────────────────────────────────────────

def evaluate_conversation(agent_prompt, conversation, persona, run_id,
                          classify_turn, evaluate_turn, aggregate_and_log):
    """
    Classify and evaluate all agent turns in one conversation.
    Returns the aggregated report, or None on failure.
    """
    turns = conversation.get("turns", [])

    # Filter to non-empty agent turns
    non_empty_agent_indices = []
    for i, turn in enumerate(turns):
        if turn["role"] == "agent" and turn["content"].strip():
            non_empty_agent_indices.append(i)

    if not non_empty_agent_indices:
        print(f"  [ERROR] No non-empty agent turns")
        return None

    # ── Phase 1: Classify all turns in parallel ──
    print(f"\n  [Step 2] EVALUATE — {len(non_empty_agent_indices)} agent turns")
    print(f"    Phase 1: Classifying turns (parallel, {MAX_EVAL_WORKERS} workers)...")

    classifications = {}

    def _classify_one(turn_index):
        turn = turns[turn_index]
        context = turns[:turn_index]
        return turn_index, classify_turn(agent_prompt, context, turn["content"])

    with ThreadPoolExecutor(max_workers=MAX_EVAL_WORKERS) as pool:
        futures = {pool.submit(_classify_one, idx): idx for idx in non_empty_agent_indices}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, result = future.result()
                classifications[idx] = result
            except Exception as e:
                print(f"    [WARN] Classification failed for turn {idx}: {e}")
                classifications[idx] = {"step_id": "unknown", "reasoning": f"error: {e}"}

    # ── Phase 2: Evaluate all turns in parallel ──
    print(f"    Phase 2: Evaluating turns (parallel, {MAX_EVAL_WORKERS} workers)...")

    evaluations = {}

    def _evaluate_one(turn_index):
        turn = turns[turn_index]
        context = turns[:turn_index]
        step_id = classifications[turn_index]["step_id"]
        return turn_index, evaluate_turn(agent_prompt, step_id, context, turn["content"])

    with ThreadPoolExecutor(max_workers=MAX_EVAL_WORKERS) as pool:
        futures = {pool.submit(_evaluate_one, idx): idx for idx in non_empty_agent_indices}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, result = future.result()
                evaluations[idx] = result
            except Exception as e:
                print(f"    [WARN] Evaluation failed for turn {idx}: {e}")
                evaluations[idx] = {
                    "scores": {"instruction_following": 0, "conversation_coherence": 0,
                               "professionalism": 0, "hallucination_risk": 0},
                    "reasoning": {}, "notes": f"error: {e}",
                }

    # ── Assemble turn results (in order) ──
    turn_results = []
    for i in non_empty_agent_indices:
        agent_text = turns[i]["content"]
        classification = classifications[i]
        evaluation = evaluations[i]

        turn_result = {
            "turn_index": i,
            "agent_text": agent_text[:100] + ("..." if len(agent_text) > 100 else ""),
            "step_id": classification["step_id"],
            "classification_reasoning": classification["reasoning"],
            "scores": evaluation["scores"],
            "reasoning": evaluation["reasoning"],
            "notes": evaluation["notes"],
        }
        turn_results.append(turn_result)

        scores_str = " ".join(f"{k}={v:.2f}" for k, v in evaluation["scores"].items())
        print(f"    Turn {i}: [{classification['step_id']}]")
        print(f"      Agent: {agent_text[:80]}...")
        print(f"      Scores: {scores_str}")

    # ── Aggregate ──
    print(f"\n  [Step 3] AGGREGATE")

    report = aggregate_and_log(turn_results, conversation, persona, run_id)

    for step, data in report["per_step_scores"].items():
        turns_str = ",".join(str(t) for t in data["turns"])
        scores_str = " ".join(f"{k}={v:.2f}" for k, v in data["scores"].items())
        print(f"    {step} (turns {turns_str}): {scores_str} | overall={data['overall']:.2f}")
    print(f"    GRAND OVERALL: {report['grand_overall']:.3f}")

    return report


# ─────────────────────────────────────────────────────────
# Normal mode: simulate + evaluate
# ─────────────────────────────────────────────────────────

def run_pipeline(
    prompt_path: str,
    transcript_dir: str = None,
    personas_path: str = None,
    num_personas: int = DEFAULT_NUM_PERSONAS,
    num_sims: int = NUM_SIMULATIONS_PER_PERSONA,
):
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    print(f"\n{'=' * 60}")
    print(f"  VOICE AGENT PROMPT EVALUATION PIPELINE")
    print(f"  Turn-by-turn stage-wise evaluation")
    print(f"  Run: {run_id}")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load prompt ──
    print(f"\n[Step 0] Loading prompt: {prompt_path}")
    agent_prompt = Path(prompt_path).read_text(encoding="utf-8")
    print(f"  {len(agent_prompt.split()):,} words")

    # ── Get personas ──
    personas = []
    if personas_path and os.path.exists(personas_path):
        print(f"\n[Personas] Loading: {personas_path}")
        with open(personas_path, encoding="utf-8") as f:
            personas = json.load(f)
        print(f"  Loaded {len(personas)} personas")
    elif transcript_dir:
        print(f"\n[Personas] Extracting from transcripts...")
        from .persona_extractor import extract_personas
        personas = extract_personas(transcript_dir, num_personas=num_personas)
    else:
        print("\n[ERROR] Need --transcripts or --personas")
        sys.exit(1)

    if not personas:
        print("[ERROR] No personas available.")
        sys.exit(1)

    # ── Import modules ──
    from .simulator import simulate_conversation, save_conversation
    from .turn_classifier import classify_turn
    from .turn_evaluator import evaluate_turn
    from .score_logger import aggregate_and_log, save_run_summary

    all_reports = []

    # ── For each persona × sim ──
    total_sims = len(personas) * num_sims
    sim_count = 0

    for persona in personas:
        for sim_num in range(num_sims):
            sim_count += 1
            persona_name = persona.get("name_en", persona.get("persona_id", "?"))
            print(f"\n{'─' * 60}")
            print(f"  Conversation {sim_count}/{total_sims}: {persona_name}")
            print(f"{'─' * 60}")

            # ── Step 1: Simulate ──
            print(f"\n  [Step 1] SIMULATE — Full conversation")
            try:
                conversation = simulate_conversation(agent_prompt, persona)
                save_conversation(conversation)
            except Exception as e:
                print(f"  [ERROR] Simulation failed: {e}")
                continue

            # ── Step 2+3: Evaluate ──
            report = evaluate_conversation(
                agent_prompt, conversation, persona, run_id,
                classify_turn, evaluate_turn, aggregate_and_log,
            )
            if report:
                all_reports.append(report)

    # ── Final summary ──
    if all_reports:
        summary_path = save_run_summary(all_reports, run_id)
        _print_final_summary(all_reports, summary_path)

    return all_reports


# ─────────────────────────────────────────────────────────
# Eval-only mode: re-evaluate existing conversations
# ─────────────────────────────────────────────────────────

def run_eval_only(prompt_path: str, conversations_dir: str):
    """
    Load existing conversation JSON files and re-evaluate them
    without re-simulating. Useful for testing evaluator consistency.
    """
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    print(f"\n{'=' * 60}")
    print(f"  VOICE AGENT PROMPT EVALUATION PIPELINE")
    print(f"  EVAL-ONLY MODE (re-evaluating existing conversations)")
    print(f"  Run: {run_id}")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load prompt ──
    print(f"\n[Step 0] Loading prompt: {prompt_path}")
    agent_prompt = Path(prompt_path).read_text(encoding="utf-8")
    print(f"  {len(agent_prompt.split()):,} words")

    # ── Load existing conversations ──
    conv_files = sorted(glob.glob(os.path.join(conversations_dir, "conv_*.json")))
    if not conv_files:
        print(f"\n[ERROR] No conversation files found in {conversations_dir}")
        sys.exit(1)

    print(f"\n[Conversations] Found {len(conv_files)} existing conversations")

    # ── Import modules ──
    from .turn_classifier import classify_turn
    from .turn_evaluator import evaluate_turn
    from .score_logger import aggregate_and_log, save_run_summary

    all_reports = []

    for idx, conv_file in enumerate(conv_files, 1):
        # Load conversation
        with open(conv_file, encoding="utf-8") as f:
            conversation = json.load(f)

        persona_name = conversation.get("persona_name", conversation.get("persona_id", "unknown"))
        persona_id = conversation.get("persona_id", "unknown")

        print(f"\n{'─' * 60}")
        print(f"  Re-evaluating {idx}/{len(conv_files)}: {persona_name}")
        print(f"  Source: {os.path.basename(conv_file)}")
        print(f"{'─' * 60}")

        # Build a minimal persona dict from the conversation metadata
        persona = {
            "persona_id": persona_id,
            "name_en": persona_name,
        }

        # Skip simulation (Step 1), go straight to evaluation
        print(f"\n  [Step 1] SIMULATE — SKIPPED (eval-only mode)")

        report = evaluate_conversation(
            agent_prompt, conversation, persona, run_id,
            classify_turn, evaluate_turn, aggregate_and_log,
        )
        if report:
            all_reports.append(report)

    # ── Final summary ──
    if all_reports:
        summary_path = save_run_summary(all_reports, run_id)
        _print_final_summary(all_reports, summary_path)
    else:
        print("\n[ERROR] No conversations were successfully evaluated.")

    return all_reports


# ─────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────

def _print_final_summary(reports, summary_path):
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Conversations evaluated: {len(reports)}\n")

    from collections import defaultdict
    step_scores = defaultdict(list)
    for r in reports:
        for step, data in r["per_step_scores"].items():
            step_scores[step].append(data["overall"])

    print(f"  {'Step':<45} {'Mean':>6} {'Min':>6} {'Max':>6} {'N':>4}")
    print(f"  {'─' * 70}")
    for step, scores in sorted(step_scores.items()):
        mean = sum(scores) / len(scores)
        print(f"  {step:<45} {mean:>6.3f} {min(scores):>6.3f} {max(scores):>6.3f} {len(scores):>4}")

    all_overalls = [r["grand_overall"] for r in reports]
    grand = sum(all_overalls) / len(all_overalls)
    print(f"  {'─' * 70}")
    print(f"  {'GRAND OVERALL':<45} {grand:>6.3f}")
    print(f"\n  Full report: {summary_path}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Prompt Evaluation Pipeline")
    parser.add_argument("--prompt", "-p", required=True, help="Path to full prompt file")
    parser.add_argument("--transcripts", "-t", default=None, help="Transcript directory")
    parser.add_argument("--personas", default=None, help="Pre-extracted personas JSON")
    parser.add_argument("--num-personas", type=int, default=DEFAULT_NUM_PERSONAS)
    parser.add_argument("--num-sims", type=int, default=NUM_SIMULATIONS_PER_PERSONA)
    parser.add_argument("--eval-only", default=None, metavar="CONV_DIR",
                        help="Re-evaluate existing conversations (skip simulation). "
                             "Pass the path to conversations directory.")
    args = parser.parse_args()

    if not os.path.exists(args.prompt):
        print(f"Error: {args.prompt} not found")
        sys.exit(1)

    if args.eval_only:
        # Eval-only mode
        if not os.path.isdir(args.eval_only):
            print(f"Error: {args.eval_only} is not a directory")
            sys.exit(1)
        run_eval_only(
            prompt_path=args.prompt,
            conversations_dir=args.eval_only,
        )
    else:
        # Normal mode
        run_pipeline(
            prompt_path=args.prompt,
            transcript_dir=args.transcripts,
            personas_path=args.personas,
            num_personas=args.num_personas,
            num_sims=args.num_sims,
        )


if __name__ == "__main__":
    main()
