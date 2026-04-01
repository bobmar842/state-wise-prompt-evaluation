"""
Transcript loader — reads raw transcript files from the real estate company.

Handles two formats found in the data:
  1. Plain text block followed by "Timestamps:" section
  2. Python dict string with 'text' and 'chunks' keys
"""

import ast
import os
from pathlib import Path


def load_single_transcript(filepath: str) -> dict:
    """
    Load one transcript file and return a normalised dict.

    Returns:
        {
            "filename": str,
            "text":     str,       # full raw transcript text
            "chunks":   list[dict] # [{start, end, text}, ...] or empty
        }
    """
    raw = Path(filepath).read_text(encoding="utf-8")
    filename = os.path.basename(filepath)

    # --- Format 2: Python dict string (starts with { or {') ---
    stripped = raw.strip()
    if stripped.startswith("{") and "'text'" in stripped[:50]:
        try:
            data = ast.literal_eval(stripped)
            chunks = []
            for c in data.get("chunks", []):
                ts = c.get("timestamp", (0, 0))
                chunks.append({
                    "start": ts[0],
                    "end": ts[1],
                    "text": c.get("text", ""),
                })
            return {
                "filename": filename,
                "text": data.get("text", ""),
                "chunks": chunks,
            }
        except (ValueError, SyntaxError):
            pass  # fall through to format 1

    # --- Format 1: Plain text + Timestamps section ---
    text_part = raw
    chunks = []

    if "\nTimestamps:" in raw:
        parts = raw.split("\nTimestamps:", 1)
        text_part = parts[0].strip()
        ts_block = parts[1].strip()
        for line in ts_block.splitlines():
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            try:
                # Parse "(start, end): text"
                paren_end = line.index(")")
                times_str = line[1:paren_end]
                start_s, end_s = times_str.split(",")
                start = float(start_s.strip())
                end = float(end_s.strip())
                text = line[paren_end + 1:].lstrip(":").strip()
                chunks.append({"start": start, "end": end, "text": text})
            except (ValueError, IndexError):
                continue

    return {
        "filename": filename,
        "text": text_part.strip(),
        "chunks": chunks,
    }


def load_all_transcripts(directory: str) -> list[dict]:
    """Load all .txt transcript files from a directory."""
    transcripts = []
    dir_path = Path(directory)

    # Handle nested folder ("transcripts without speaker separation")
    txt_files = list(dir_path.rglob("*.txt"))

    for fp in sorted(txt_files):
        try:
            t = load_single_transcript(str(fp))
            if t["text"]:  # skip empty
                transcripts.append(t)
        except Exception as e:
            print(f"  [warn] skipping {fp.name}: {e}")

    return transcripts
