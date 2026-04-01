"""
Persona Extractor
=================
Reads raw transcripts and uses an LLM to extract distinct customer persona
profiles. These personas drive the user simulator in conversation testing.

Merged from eval_framework's persona_extractor.py:
- Better transcript sampling (fills remaining space after even spread)
- Persona field validation with warnings
"""

import json
import os
import re
from pathlib import Path

from .config import (
    PERSONA_EXTRACTOR_MODEL,
    PERSONA_EXTRACTOR_TEMPERATURE,
    DEFAULT_NUM_PERSONAS,
    PERSONAS_FILE,
    OUTPUT_DIR,
    PROMPTS_DIR,
)
from .llm_service import call_llm
from .transcript_loader import load_all_transcripts


# Required fields for a valid persona
REQUIRED_PERSONA_FIELDS = [
    "persona_id", "name_en", "selling_intent", "emotional_tone",
    "difficulty_level", "simulator_prompt_ja",
]


def extract_personas(
    transcript_dir: str,
    num_personas: int = DEFAULT_NUM_PERSONAS,
) -> list[dict]:
    """
    Main entry point: load transcripts, extract personas, save to file.

    Args:
        transcript_dir: path to directory containing transcript .txt files
        num_personas:   how many distinct personas to extract

    Returns:
        List of persona dicts
    """
    print(f"  Loading transcripts from: {transcript_dir}")
    transcripts = load_all_transcripts(transcript_dir)
    print(f"  Loaded {len(transcripts)} transcripts")

    if not transcripts:
        raise ValueError(f"No transcripts found in {transcript_dir}")

    # Prepare the prompt
    template = (PROMPTS_DIR / "persona_extraction.txt").read_text(encoding="utf-8")
    system_prompt = template.replace("{num_personas}", str(num_personas))
    transcript_block = _prepare_transcript_batch(transcripts)

    print(f"  Sending {len(transcript_block)} chars to LLM for persona extraction...")

    result = call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Here are {len(transcripts)} real call transcripts. "
                    f"Extract {num_personas} distinct customer personas.\n\n"
                    f"{transcript_block}"
                ),
            },
        ],
        model=PERSONA_EXTRACTOR_MODEL,
        temperature=PERSONA_EXTRACTOR_TEMPERATURE,
        max_tokens=4000,
    )

    personas = _parse_json_response(result["text"])
    print(f"  Extracted {len(personas)} personas")

    # Validate required fields
    for p in personas:
        missing = [f for f in REQUIRED_PERSONA_FIELDS if f not in p]
        if missing:
            print(f"  [warn] persona '{p.get('persona_id', '?')}' missing: {missing}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PERSONAS_FILE, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {PERSONAS_FILE}")

    return personas


def _prepare_transcript_batch(transcripts: list[dict], max_chars: int = 60000) -> str:
    """
    Combine transcripts into a single block for the LLM, staying under
    the character limit. We sample evenly across the set, then fill
    remaining space with any leftover transcripts.
    """
    # Sort by length to get variety — mix short and long calls
    by_length = sorted(transcripts, key=lambda t: len(t["text"]))
    selected = []
    total_chars = 0

    # Take every Nth to spread across the set
    step = max(1, len(by_length) // 20)
    for i in range(0, len(by_length), step):
        t = by_length[i]
        if total_chars + len(t["text"]) > max_chars:
            break
        selected.append(t)
        total_chars += len(t["text"])

    # If we have room, fill with remaining
    for t in by_length:
        if t in selected:
            continue
        if total_chars + len(t["text"]) > max_chars:
            break
        selected.append(t)
        total_chars += len(t["text"])

    parts = []
    for i, t in enumerate(selected, 1):
        parts.append(f"--- Transcript {i}: {t['filename']} ---\n{t['text']}\n")

    return "\n".join(parts)


def _parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try regex extraction
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return json.loads(match.group())
        raise ValueError("Could not parse LLM response as JSON array")
