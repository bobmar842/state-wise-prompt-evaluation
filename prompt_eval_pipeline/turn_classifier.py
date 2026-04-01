"""
Turn Classifier — Step 2a (IDENTIFY)
=====================================
For each agent turn in the conversation, sends it to the LLM
along with the full prompt and conversation context.
The LLM reads the prompt's step definitions and says
"this turn is executing STEP 2" or "this is an FAQ response."

The steps come from the prompt itself — we don't impose fixed stages.

Fixes applied:
- Extract only STEP/flow structure from prompt (saves ~70% tokens)
- Only include last N turns of context (not entire history)
- Better JSON fallback parsing
"""

import json
import re

from .config import CLASSIFIER_MODEL, CLASSIFIER_TEMPERATURE, PROMPTS_DIR
from .llm_service import call_llm

# Max context turns to include (most recent)
MAX_CONTEXT_TURNS = 6


def _extract_prompt_structure(agent_prompt: str) -> str:
    """
    Extract just the STEP definitions, flow names, and section headers
    from the full prompt. This dramatically reduces token usage while
    keeping what the classifier actually needs.
    """
    lines = agent_prompt.split("\n")
    structure_lines = []
    in_step = False
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        # Always include section headers and step definitions
        if re.match(r"^#{1,4}\s", stripped):
            structure_lines.append(line)
            in_step = True
            blank_count = 0
            continue

        # Include lines with STEP references, flow names, key markers
        if re.search(r"STEP\s*\d|▼|Global|Exit Flow|Transfer|Fallback|FAQ|Rebuttal", stripped, re.IGNORECASE):
            structure_lines.append(line)
            in_step = True
            blank_count = 0
            continue

        # Include a few lines after headers for context
        if in_step and blank_count < 2:
            if stripped:
                structure_lines.append(line)
                blank_count = 0
            else:
                blank_count += 1
                if blank_count >= 2:
                    in_step = False
                    structure_lines.append("")

    result = "\n".join(structure_lines)

    # If extraction is too aggressive (less than 20% of original), use full prompt
    if len(result) < len(agent_prompt) * 0.2:
        return agent_prompt

    return result


def _truncate_context(turns: list[dict], max_turns: int = MAX_CONTEXT_TURNS) -> list[dict]:
    """Keep only the most recent N turns for context."""
    if len(turns) <= max_turns:
        return turns
    return turns[-max_turns:]


def classify_turn(
    agent_prompt: str,
    conversation_so_far: list[dict],
    current_turn: str,
) -> dict:
    """
    Classify which step of the prompt an agent turn belongs to.

    Args:
        agent_prompt: the full system prompt
        conversation_so_far: all turns up to (but not including) current_turn
        current_turn: the agent utterance to classify

    Returns:
        {"step_id": "STEP 2: Assessment Purpose", "reasoning": "..."}
    """
    template = (PROMPTS_DIR / "turn_classifier.txt").read_text(encoding="utf-8")

    # Use extracted structure instead of full prompt
    prompt_structure = _extract_prompt_structure(agent_prompt)

    # Truncate context to recent turns
    recent_context = _truncate_context(conversation_so_far)
    conv_text = _format_turns(recent_context)

    prompt = template.replace("{agent_prompt}", prompt_structure)
    prompt = prompt.replace("{conversation_so_far}", conv_text or "[No previous turns — this is the first turn]")
    prompt = prompt.replace("{current_turn}", current_turn)

    result = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model=CLASSIFIER_MODEL,
        temperature=CLASSIFIER_TEMPERATURE,
        max_tokens=300,
    )

    return _parse_response(result["text"])


def _format_turns(turns: list[dict]) -> str:
    """Format turns into readable text."""
    if not turns:
        return ""
    lines = []
    for t in turns:
        role = "エージェント" if t["role"] == "agent" else "顧客"
        # Truncate long turns
        content = t["content"][:300] + ("..." if len(t["content"]) > 300 else "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _normalize_step_name(step_id: str) -> str:
    """
    Normalize step names for consistent aggregation.
    Fixes: "STEP2:" vs "STEP 2:", extra whitespace, etc.
    """
    if not step_id or step_id == "unknown":
        return step_id
    # Add space after STEP if missing: "STEP2:" → "STEP 2:"
    step_id = re.sub(r'STEP(\d)', r'STEP \1', step_id)
    # Normalize multiple spaces
    step_id = re.sub(r'\s+', ' ', step_id).strip()
    return step_id


def _parse_response(text: str) -> dict:
    """Parse LLM JSON response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to find JSON in the text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {"step_id": "unknown", "reasoning": f"Could not parse: {text[:200]}"}
        else:
            return {"step_id": "unknown", "reasoning": f"Could not parse: {text[:200]}"}

    return {
        "step_id": _normalize_step_name(data.get("step_id", "unknown")),
        "reasoning": data.get("reasoning", ""),
    }
