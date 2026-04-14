# Voice Agent Prompt Evaluation Pipeline

Simulates conversations using the full prompt, then evaluates **turn-by-turn** — identifying which step of the prompt each agent response belongs to and scoring it against that step's rules.

## How it works

```
Full Prompt + Personas
       │
  ┌────▼────┐
  │ SIMULATE │  Full LLM-vs-LLM conversation (10-14 turns)
  │          │  Uses the complete prompt, nothing split
  └────┬────┘
       │
       ▼  For each agent turn:
  ┌─────────┐
  │ CLASSIFY │  "Which step of the prompt is this turn executing?"
  │          │  LLM reads the prompt's own step definitions
  │          │  Returns: "STEP 2: Assessment Purpose" or "FAQ: Fees"
  └────┬────┘
       │
  ┌────▼────┐
  │ EVALUATE │  Score this turn against that step's instructions
  │          │  instruction_following, coherence, professionalism,
  │          │  hallucination_risk
  └────┬────┘
       │
  ┌────▼────┐
  │ REMEMBER │  Accumulate scores under that step
  └────┬────┘
       │
       ▼  After all turns:
  ┌──────────┐
  │ AGGREGATE│  Per-step scores + grand overall
  │          │  STEP 1 (turns 0,2): 0.91
  │          │  STEP 2 (turns 4):   0.81
  │          │  FAQ (turns 6):      0.78
  │          │  GRAND OVERALL:      0.83
  └──────────┘
```

## Quick Start

```bash
pip install -r requirements.txt

# Copy and fill in your API key(s)
cp .env.example .env

# With transcripts (extracts personas first)
python -m prompt_eval_pipeline --prompt examples/input/real_estate_prompt.txt --transcripts "transcripts/transcripts without speaker separation/"

# With pre-extracted personas
python -m prompt_eval_pipeline --prompt examples/input/real_estate_prompt.txt --personas output/personas.json
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```env
# Required: your LiteLLM-compatible API key
LLM_API_KEY=sk-...
OPENAI_API_KEY=sk-...         # if using OpenAI models

# Optional: override model choices (defaults shown)
LLM_MODEL=openai/gpt-4o-mini
SIMULATOR_CALLER_MODEL=openai/gpt-4o-mini
SIMULATOR_USER_MODEL=openai/gpt-4o-mini
PERSONA_EXTRACTOR_MODEL=openai/gpt-5.4-mini
CLASSIFIER_MODEL=openai/gpt-5.4-mini
EVALUATOR_MODEL=openai/gpt-5.4-mini

# Optional: run settings
DEFAULT_NUM_PERSONAS=10
NUM_SIMULATIONS_PER_PERSONA=1
MAX_EVAL_WORKERS=4            # parallel classify+evaluate threads
OUTPUT_DIR=output

# Optional: Langfuse observability
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## CLI Reference

```
python -m prompt_eval_pipeline --prompt PROMPT [options]

Required:
  --prompt, -p PROMPT         Path to the full agent prompt file

Persona source (one required):
  --transcripts, -t DIR       Extract personas from transcript directory
  --personas PATH             Use pre-extracted personas JSON file

Options:
  --num-personas N            How many personas to extract (default: 10)
  --num-sims N                Simulations per persona (default: 1)
  --eval-only CONV_DIR        Skip simulation; re-evaluate existing
                              conversation JSON files in CONV_DIR
```

### Examples

```bash
# Standard run: extract personas from transcripts, simulate + evaluate
python -m prompt_eval_pipeline \
  --prompt examples/input/real_estate_prompt.txt \
  --transcripts "transcripts/transcripts without speaker separation/" \
  --num-personas 5

# Use saved personas (skip extraction)
python -m prompt_eval_pipeline \
  --prompt examples/input/real_estate_prompt.txt \
  --personas output/personas.json \
  --num-sims 2

# Re-evaluate existing conversations with a modified prompt (no re-simulation)
python -m prompt_eval_pipeline \
  --prompt examples/input/real_estate_prompt.txt \
  --eval-only output/conversations/
```

## What you get

```
output/
├── personas.json                      # extracted customer personas
├── conversations/
│   └── conv_<persona>_<ts>.json       # full simulated conversations
├── report_<persona>_<ts>.json         # per-turn + per-step scores
└── run_<ts>_summary.json              # aggregated summary across all personas
```

The summary shows scores grouped by prompt step:
```
  Step                                          Mean    Min    Max
  ──────────────────────────────────────────────────────────────
  STEP 1: Opening Greeting                     0.920  0.880  0.960
  STEP 2: Assessment Purpose                   0.810  0.750  0.870
  FAQ: Fees                                    0.780  0.780  0.780
  STEP 3: Desired Direction                    0.850  0.850  0.850
  STEP 4: Priority Needs                       0.900  0.900  0.900
  ──────────────────────────────────────────────────────────────
  GRAND OVERALL                                0.852
```

Each report also includes per-turn feedback text (reasoning from the evaluator LLM) grouped by step, useful for debugging which specific turns are dragging down a score.

## Comparing Two Runs

`compare.py` is a quick utility to diff grand-overall scores between two runs:

```bash
# Move run1 reports to output/run1_reports/ first, then:
python compare.py
```

Output:
```
Persona                    Run1   Run2   Diff
---------------------------------------------
persona_a                 0.852  0.871  0.019
persona_b                 0.801  0.760  0.041 ⚠️
```

Differences above 0.1 are flagged with ⚠️.

## Project Structure

```
prompt_eval_pipeline/
├── __main__.py             # Entry point (python -m prompt_eval_pipeline)
├── run.py                  # Orchestrator — ties everything together
├── config.py               # All settings (reads .env)
├── llm_service.py          # LiteLLM wrapper
├── persona_extractor.py    # Extracts personas from transcripts
├── transcript_loader.py    # Reads transcript files
├── simulator.py            # Runs full conversation (full prompt)
├── turn_classifier.py      # "Which step is this turn?" (LLM call)
├── turn_evaluator.py       # "How well did this turn follow that step?" (LLM call)
├── score_logger.py         # Aggregates per-step scores, saves report, optional Langfuse push
└── prompts/
    ├── turn_classifier.txt     # Prompt for classifying turns
    ├── turn_evaluator.txt      # Prompt for evaluating turns
    ├── persona_extraction.txt  # Prompt for persona extraction
    └── user_simulator.txt      # Prompt for user simulator

examples/
└── input/
    └── real_estate_prompt.txt  # Example agent prompt

compare.py                  # Utility to diff scores across two runs
```

## Key Design Decisions

**Stages come from the prompt, not from us.** The classifier reads the prompt's own step definitions (STEP 1, STEP 2, FAQ, Supplementary Flow, Exit Flow — whatever the prompt defines) and maps turns to those. We don't impose fixed categories.

**Full prompt for simulation, full prompt for evaluation.** The simulator uses the complete prompt. The classifier and evaluator also see the complete prompt so they can match turns to any step or flow defined in it.

**Four dimensions per turn.** Every turn is scored on instruction_following, conversation_coherence, professionalism, and hallucination_risk. These are universal — they apply to any step.

**Parallel classify + evaluate.** All turns in a conversation are classified in parallel, then evaluated in parallel (controlled by `MAX_EVAL_WORKERS`). This makes runs significantly faster on long conversations.

**Langfuse observability (optional).** If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, every per-step and grand-overall score is pushed to Langfuse automatically after each conversation.
