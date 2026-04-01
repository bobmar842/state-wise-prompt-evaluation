"""
Simulator — Step 1
==================
Runs a full LLM-vs-LLM conversation using the FULL agent prompt.
Nothing fancy — same as the original eval framework.
The conversation is later evaluated turn-by-turn.

Fixes applied:
- Template variable resolution ({{name}} etc.)
- Repetition/stuck-loop detection
- Empty content detection & retry
- Extended end-of-conversation detection
"""

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher

from .config import (
    SIMULATOR_CALLER_MODEL, SIMULATOR_USER_MODEL,
    SIMULATOR_CALLER_TEMPERATURE, SIMULATOR_USER_TEMPERATURE,
    MAX_CONVERSATION_TURNS, MIN_CONVERSATION_TURNS,
    MAX_AGENT_REPETITIONS,
    CONVERSATIONS_DIR, PROMPTS_DIR,
)
from .llm_service import call_llm


# ─── Default values for unresolved template variables ───
DEFAULT_TEMPLATE_VARS = {
    "agent_name": "佐藤",
    "service_name": "不動産査定ナビ",
    "property_type": "不動産",
    "name": "お客様",
    "company_name": "不動産査定ナビ株式会社",
    "variable": "",
}


def _resolve_template_vars(text: str, persona: dict = None) -> str:
    """
    Replace {{var}} placeholders with sensible defaults.
    This prevents the LLM from echoing raw template variables
    or returning empty content due to confusion about placeholders.
    """
    # Use persona overrides if provided
    overrides = {}
    if persona:
        if persona.get("name_ja"):
            overrides["name"] = persona["name_ja"]

    for var, default in DEFAULT_TEMPLATE_VARS.items():
        value = overrides.get(var, default)
        text = text.replace("{{" + var + "}}", value)

    # Catch any remaining {{...}} patterns
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: m.group(1), text)
    return text


def strip_markup_tags(text: str) -> str:
    """
    Remove <fixed>, </fixed>, <flush />, and similar markup tags
    that should not appear in spoken phone conversation output.
    """
    # Remove <fixed>...</fixed> wrapper but keep inner text
    text = re.sub(r'`?</?fixed\s*/?>`?', '', text)
    # Remove <flush /> tags
    text = re.sub(r'`?<flush\s*/?>`?', '', text)
    # Remove any other XML-like tags that slipped through
    text = re.sub(r'`?</?[a-zA-Z][^>]*/?>`?', '', text)
    # Clean up extra whitespace/newlines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _is_transfer_wait(msg: str) -> bool:
    """Check if a message is a transfer/wait pattern (tighter repetition threshold)."""
    return bool(re.search(r'お繋ぎ|お待ちください|担当者に.*繋|転送', msg))


def _similarity(a: str, b: str) -> float:
    """Quick similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    # Normalize: strip whitespace and flush tags
    a_clean = re.sub(r"`<[^>]*>`|`</[^>]*>`|\s+", " ", a).strip()
    b_clean = re.sub(r"`<[^>]*>`|`</[^>]*>`|\s+", " ", b).strip()
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def _is_repetitive(new_msg: str, turns: list[dict], role: str, threshold: float = 0.75) -> int:
    """
    Count how many recent same-role turns are highly similar to new_msg.
    Returns the count of consecutive similar messages.
    """
    count = 0
    for t in reversed(turns):
        if t["role"] != role:
            continue
        if _similarity(new_msg, t["content"]) >= threshold:
            count += 1
        else:
            break
    return count


def simulate_conversation(agent_prompt: str, persona: dict) -> dict:
    """
    Run a full simulated conversation.

    Args:
        agent_prompt: the FULL system prompt (not split, not staged)
        persona: persona dict with simulator_prompt_ja

    Returns:
        Conversation dict with turns, metadata
    """
    persona_id = persona.get("persona_id", "unknown")
    persona_name = persona.get("name_en", persona_id)
    print(f"  [Sim] {persona_name}...", end=" ", flush=True)

    # ── Resolve template variables before sending to LLM ──
    resolved_prompt = _resolve_template_vars(agent_prompt, persona)

    caller_system = (
        "あなたはこれからシミュレーション電話会話を行います。\n"
        "以下のシステムプロンプトに完全に従って、AI音声アシスタントとして応答してください。\n"
        "一回の発話は短く自然にしてください（電話会話として）。\n"
        "会話は日本語で行ってください。\n"
        "注意: `<flush />`や`<fixed>`などのマークアップタグは出力に含めないでください。自然な日本語のみで応答してください。\n\n"
        "--- ここからシステムプロンプト ---\n\n"
        + resolved_prompt
    )

    user_template = (PROMPTS_DIR / "user_simulator.txt").read_text(encoding="utf-8")
    user_system = user_template.replace(
        "{persona_prompt}",
        persona.get("simulator_prompt_ja", "普通の顧客として応答してください。"),
    )

    turns = []
    stop_reason = "max_turns"
    caller_tokens = 0
    user_tokens = 0
    empty_count = 0
    start = datetime.now()

    for turn_num in range(MAX_CONVERSATION_TURNS):
        # ── Agent turn ──
        caller_msgs = [{"role": "system", "content": caller_system}]
        for t in turns:
            role = "assistant" if t["role"] == "agent" else "user"
            caller_msgs.append({"role": role, "content": t["content"]})

        if turn_num == 0:
            caller_msgs.append({"role": "user", "content": "（電話が繋がりました。顧客に最初の挨拶をしてください。）"})

        r = call_llm(caller_msgs, SIMULATOR_CALLER_MODEL, SIMULATOR_CALLER_TEMPERATURE, 500)
        agent_msg = r["text"].strip()
        caller_tokens += r["usage"].get("total_tokens", 0)

        # ── Strip markup tags from output ──
        agent_msg = strip_markup_tags(agent_msg)

        # ── Empty content check ──
        if not agent_msg:
            empty_count += 1
            if empty_count >= 3:
                stop_reason = "empty_responses"
                print(f"[WARN] {empty_count} empty agent responses, stopping")
                break
            # Skip this turn pair entirely
            continue

        # ── Repetition check ──
        # Tighter threshold for transfer-wait patterns (1 repeat = stop)
        rep_count = _is_repetitive(agent_msg, turns, "agent")
        if _is_transfer_wait(agent_msg) and rep_count >= 1:
            turns.append({"role": "agent", "content": agent_msg})
            stop_reason = "agent_repetition"
            print(f"[WARN] Agent stuck in transfer-wait loop, stopping")
            break
        if rep_count >= MAX_AGENT_REPETITIONS:
            turns.append({"role": "agent", "content": agent_msg})
            stop_reason = "agent_repetition"
            print(f"[WARN] Agent stuck repeating ({rep_count}x), stopping")
            break

        turns.append({"role": "agent", "content": agent_msg})

        # ── End detection (agent) ──
        if turn_num >= MIN_CONVERSATION_TURNS and _detect_end(agent_msg):
            stop_reason = "agent_ended"
            break

        # ── User turn ──
        user_msgs = [{"role": "system", "content": user_system}]
        for t in turns:
            role = "assistant" if t["role"] == "user" else "user"
            user_msgs.append({"role": role, "content": t["content"]})

        r = call_llm(user_msgs, SIMULATOR_USER_MODEL, SIMULATOR_USER_TEMPERATURE, 300)
        user_msg = r["text"].strip()
        user_tokens += r["usage"].get("total_tokens", 0)

        if not user_msg:
            empty_count += 1
            if empty_count >= 3:
                stop_reason = "empty_responses"
                break
            continue

        turns.append({"role": "user", "content": user_msg})

        if turn_num >= MIN_CONVERSATION_TURNS and _detect_end(user_msg):
            stop_reason = "user_ended"
            break

    elapsed = (datetime.now() - start).total_seconds()
    agent_turns = len([t for t in turns if t["role"] == "agent"])
    print(f"{agent_turns} turns, {stop_reason}, {elapsed:.1f}s")

    return {
        "persona_id": persona_id,
        "persona_name": persona_name,
        "timestamp": datetime.now().isoformat(),
        "turns": turns,
        "turn_count": agent_turns,
        "stop_reason": stop_reason,
        "metadata": {
            "total_time_seconds": round(elapsed, 2),
            "caller_total_tokens": caller_tokens,
            "user_total_tokens": user_tokens,
        },
    }


def save_conversation(conversation: dict) -> str:
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    pid = conversation["persona_id"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(CONVERSATIONS_DIR, f"conv_{pid}_{ts}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    return filepath


def _detect_end(msg):
    """Detect conversation-ending patterns — expanded set."""
    patterns = [
        r"失礼いたします", r"失礼します", r"さようなら",
        r"ありがとうございました.*失礼", r"→\s*END", r"電話を切", r"ガチャ",
        # Transfer patterns (conversation is done from AI side)
        r"オペレーターにお繋ぎ",
        r"お繋ぎいたします",
        r"担当者に(お|)繋ぎ",
        r"転送いたします",
        # Goodbye patterns
        r"お電話ありがとうございました",
        r"またのお電話",
        r"結構です.*切り",
        r"もういいです",
    ]
    return any(re.search(p, msg) for p in patterns)
