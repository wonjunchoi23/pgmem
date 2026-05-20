# CLAUDE.md

## Project Overview
Memory-augmented LLM agent experiments on the ImplexConv dataset (QA-only variant).
Each module (amem, gmem3, ldagent, theanine, dense, memorybank) is a separate memory system.
- Experiment spec: `global_readme.md`
- Logging spec: `additional_log.md`

## Before Writing Code
1. State the plan clearly.
2. Ask about anything ambiguous or requiring a decision — before touching code.
3. Wait for confirmation before proceeding.

## Code Style
- Concise. No unnecessary abstractions.
- All comments in English, brief unless detail is explicitly requested.
- No docstrings unless asked. No refactoring of untouched code.

## Conventions
- Each module is self-contained. Config files (`config_N.py`) define hyperparameters.
- Output dirs: `config_N_outputs_<model>_<subset>/`
- `merge_results.py` aggregates parallel run outputs.
- Parallel runs use nohup. Check running jobs before modifying shared files.

## Models
- Default: claude-sonnet-4-6, effort auto(default: medium)
