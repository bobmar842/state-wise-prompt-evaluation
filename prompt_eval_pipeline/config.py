"""
Configuration — all settings in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ─── LLM ───
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Simulation: gpt-4o-mini (fast, cheap, high volume)
SIMULATOR_CALLER_MODEL = os.getenv("SIMULATOR_CALLER_MODEL", "openai/gpt-4o-mini")
SIMULATOR_USER_MODEL = os.getenv("SIMULATOR_USER_MODEL", "openai/gpt-4o-mini")

# Persona extraction: gpt-5.4-mini (needs good Japanese understanding)
PERSONA_EXTRACTOR_MODEL = os.getenv("PERSONA_EXTRACTOR_MODEL", "openai/gpt-5.4-mini")

# Classification & Evaluation: gpt-5.4-mini (stricter grading, better JSON)
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "openai/gpt-5.4-mini")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "openai/gpt-5.4-mini")

PERSONA_EXTRACTOR_TEMPERATURE = 0.4
SIMULATOR_CALLER_TEMPERATURE = 0.7
SIMULATOR_USER_TEMPERATURE = 0.8
CLASSIFIER_TEMPERATURE = 0
EVALUATOR_TEMPERATURE = 0.2

# ─── Simulation ───
MAX_CONVERSATION_TURNS = 14
MIN_CONVERSATION_TURNS = 4
DEFAULT_NUM_PERSONAS = int(os.getenv("DEFAULT_NUM_PERSONAS", "10"))
NUM_SIMULATIONS_PER_PERSONA = int(os.getenv("NUM_SIMULATIONS_PER_PERSONA", "1"))

# ─── Repetition detection ───
# If agent repeats the same message (by similarity) N times, force-stop
MAX_AGENT_REPETITIONS = 2

# ─── Parallelism ───
# Max concurrent LLM calls for classify+evaluate
MAX_EVAL_WORKERS = int(os.getenv("MAX_EVAL_WORKERS", "4"))

# ─── Langfuse ───
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# ─── Paths ───
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
PERSONAS_FILE = os.path.join(OUTPUT_DIR, "personas.json")
CONVERSATIONS_DIR = os.path.join(OUTPUT_DIR, "conversations")
SCORES_DIR = os.path.join(OUTPUT_DIR, "scores")
PROMPTS_DIR = Path(__file__).parent / "prompts"
