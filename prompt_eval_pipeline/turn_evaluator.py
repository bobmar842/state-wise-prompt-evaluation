"""
Turn Evaluator — Step 2b (EVALUATE)
====================================
For each agent turn, given which step it belongs to,
evaluates it against the prompt's instructions for that step.
Scores: instruction_following, coherence, professionalism, hallucination_risk.

Fixes applied:
- Only send the relevant STEP section of the prompt (not the full 30KB)
- Truncate context to recent turns
- Better score extraction
"""

import json
import re

from .config import EVALUATOR_MODEL, EVALUATOR_TEMPERATURE, PROMPTS_DIR
from .llm_service import call_llm


EVAL_DIMENSIONS = ["instruction_following", "conversation_coherence", "professionalism", "hallucination_risk"]

# Max context turns to include
MAX_CONTEXT_TURNS = 6


def _extract_relevant_section(agent_prompt: str, step_id: str) -> str:
    """
    Extract just the section of the prompt relevant to the given step_id.
    Falls back to full prompt if step can't be found.
    """
    if not step_id or step_id == "unknown":
        # Can't narrow down — send the full prompt but capped
        return agent_prompt[:8000]

    lines = agent_prompt.split("\n")
    # Try to find the section header matching step_id
    step_start = None
    step_end = None

    # Normalize step_id for matching
    step_key = step_id.lower().strip()

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        # Match by header
        if step_key in line_lower or (
            re.search(r"step\s*\d", step_key) and re.search(r"step\s*\d", line_lower)
            and re.search(r"step\s*(\d)", step_key).group(1) == (re.search(r"step\s*(\d)", line_lower) or type('', (), {'group': lambda s, x: None})()).group(1)
        ):
            step_start = max(0, i - 2)
            continue

        # If we found the start, find the next major section
        if step_start is not None and step_end is None:
            if re.match(r"^#{1,3}\s+▼", line.strip()) and i > step_start + 3:
                step_end = i
                break

    if step_start is not None:
        section = "\n".join(lines[step_start:(step_end or step_start + 80)])
        # Also include global rules section (first ~50 lines usually)
        global_section = "\n".join(lines[:min(50, len(lines))])
        return global_section + "\n\n---\n\n[Relevant Step Section]\n\n" + section

    # Fallback: return capped prompt
    return agent_prompt[:8000]


def _truncate_context(turns: list[dict], max_turns: int = MAX_CONTEXT_TURNS) -> list[dict]:
    """Keep only the most recent N turns."""
    if len(turns) <= max_turns:
        return turns
    return turns[-max_turns:]


def evaluate_turn(
    agent_prompt: str,
    step_id: str,
    conversation_so_far: list[dict],
    current_turn: str,
) -> dict:
    """
    Evaluate one agent turn against its identified step.

    Args:
        agent_prompt: the full system prompt
        step_id: which step this turn belongs to (from classifier)
        conversation_so_far: all turns up to current turn
        current_turn: the agent utterance to evaluate

    Returns:
        {
            "scores": {"instruction_following": 0.9, ...},
            "reasoning": {"instruction_following": "...", ...},
            "notes": "..."
        }
    """
    template = (PROMPTS_DIR / "turn_evaluator.txt").read_text(encoding="utf-8")

    # Only send the relevant section, not the entire 30KB prompt
    relevant_prompt = _extract_relevant_section(agent_prompt, step_id)

    # Truncate context
    recent_context = _truncate_context(conversation_so_far)
    conv_text = _format_turns(recent_context)

    prompt = template.replace("{agent_prompt}", relevant_prompt)
    prompt = prompt.replace("{step_id}", step_id)
    prompt = prompt.replace("{conversation_so_far}", conv_text or "[First turn]")
    prompt = prompt.replace("{current_turn}", current_turn)

    result = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=EVALUATOR_MODEL,
        temperature=EVALUATOR_TEMPERATURE,
        max_tokens=1000,
    )

    return _parse_response(result["text"])


def _format_turns(turns: list[dict]) -> str:
    if not turns:
        return ""
    lines = []
    for t in turns:
        role = "エージェント" if t["role"] == "agent" else "顧客"
        content = t["content"][:300] + ("..." if len(t["content"]) > 300 else "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return _empty_eval("parse error")
        else:
            return _empty_eval("parse error")

    # Normalise scores
    raw_scores = data.get("scores", {})
    scores = {}
    for dim in EVAL_DIMENSIONS:
        scores[dim] = _extract_score(raw_scores.get(dim, 0))

    return {
        "scores": scores,
        "reasoning": data.get("reasoning", {}),
        "notes": data.get("notes", ""),
    }


def _extract_score(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "value", "rating"):
            if key in value:
                return _extract_score(value[key])
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _empty_eval(reason: str) -> dict:
    return {
        "scores": {dim: 0.0 for dim in EVAL_DIMENSIONS},
        "reasoning": {dim: reason for dim in EVAL_DIMENSIONS},
        "notes": reason,
    }
