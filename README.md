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
cp .env.example .env        # add your API key

# With transcripts (extracts personas first)
python -m prompt_eval_pipeline.run --prompt examples/input/real_estate_prompt.txt --transcripts "transcripts/transcripts without speaker separation/"

# With pre-extracted personas
python -m prompt_eval_pipeline.run --prompt examples/input/real_estate_prompt.txt --personas output/personas.json
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

## Project Structure

```
prompt_eval_pipeline/
├── run.py                  # Orchestrator — ties everything together
├── config.py               # All settings
├── llm_service.py          # LiteLLM wrapper
├── persona_extractor.py    # Extracts personas from transcripts
├── transcript_loader.py    # Reads transcript files
├── simulator.py            # Runs full conversation (full prompt)
├── turn_classifier.py      # "Which step is this turn?" (LLM call)
├── turn_evaluator.py       # "How well did this turn follow that step?" (LLM call)
├── score_logger.py         # Aggregates per-step scores, saves report
└── prompts/
    ├── turn_classifier.txt     # Prompt for classifying turns
    ├── turn_evaluator.txt      # Prompt for evaluating turns
    ├── persona_extraction.txt  # Prompt for persona extraction
    └── user_simulator.txt      # Prompt for user simulator
```

## Key Design Decisions

**Stages come from the prompt, not from us.** The classifier reads the prompt's own step definitions (STEP 1, STEP 2, FAQ, Supplementary Flow, Exit Flow — whatever the prompt defines) and maps turns to those. We don't impose fixed categories.

**Full prompt for simulation, full prompt for evaluation.** The simulator uses the complete prompt. The classifier and evaluator also see the complete prompt so they can match turns to any step or flow defined in it.

**Four dimensions per turn.** Every turn is scored on instruction_following, conversation_coherence, professionalism, and hallucination_risk. These are universal — they apply to any step.
