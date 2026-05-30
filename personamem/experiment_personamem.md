# Experiment: Memory-augmented LLM Agents on PersonaMem

Evaluate a memory-augmented agent on the **PersonaMem** benchmark (Jiang et al., 2025).
Phase 1 builds memory from ground-truth conversation turns only; the LLM is called
**only for QA answering** (Phase 2). No conversational responses are generated.

## Dataset

Three context-size variants, selected with `--benchmark-size`:

| Size | Shared contexts | Total QA | Avg msgs/context |
|---|---|---|---|
| 32k  | 37  | 589   | 169   |
| 128k | 110 | 2,727 | 736   |
| 1M   | 33  | 2,674 | 3,548 |

Files live under `dataset/`: `questions_{size}.csv` (one row per 4-choice QA) and
`shared_contexts_{size}.jsonl` (`{<context_id>: [{role, content}, ...]}`).

- A `system` message marks the start of a **block** (one session at a time period);
  following `user`/`assistant` messages belong to that block.
- **7 question types**: `recall_user_shared_facts`, `suggest_new_ideas`,
  `acknowledge_latest_preferences`, `track_full_preference_evolution`,
  `revisit_reasons_behind_preference_updates`,
  `provide_preference_aligned_recommendations`, `generalize_to_new_scenarios`.
- **Metric**: accuracy (predicted letter == `correct_answer`).

## Flow

Unit of processing is one shared context `C_i`. Contexts are sorted by descending QA
count; `--start-session`/`--end-session` index into that sorted list.

For each context:

1. **Phase 1 — Build memory.** Walk `user`/`assistant` messages in order (skip
   `system`, used only as block boundaries). For each: retrieve → store
   (timestamp `{ctx:04d}_{block:04d}_{msg:04d}`, content stored as-is with speaker
   prefix) → update module state. Then finalize and snapshot memory.
2. **Phase 2 — QA.** Memory is frozen. For each QA (ordered by
   `end_index_in_shared_context`): retrieve with the question → build prompt
   (memory + question + 4 options) → generate `{"answer": "a|b|c|d"}` (`"unknown"`
   on parse failure).
3. **Phase 3 — Cleanup.** Aggregate token stats, save results (atomic write),
   clear all memory state, update checkpoint.

## Key rules

- Only `user`/`assistant` messages are stored; never `system`, never QA pairs.
- Memory is fully cleared between contexts; snapshot is saved before cleanup.
- `end_index_in_shared_context` is recorded as metadata only — Phase 1 uses the full
  context and does not truncate.
- Virtual time: `CONV_IDS_PER_DAY = 1` (1 block = 1 day), `TIME_PER_TURN_MINUTES = 10`.
- Checkpoint after each context: `{"completed_session_ids": [...]}`; restart resumes
  from the next unprocessed context.

## Run

```bash
python pgmem/run_experiment.py \
    --start-session 0 --end-session 36 \
    --benchmark-size 32k \
    --model Qwen/Qwen3-1.7B \
    --batch-size 4
```

| Arg | Description |
|---|---|
| `--start-session` / `--end-session` | (required) First / last context index, inclusive |
| `--benchmark-size` | (required) `32k`, `128k`, or `1M` |
| `--model` | HF model path (default from config) |
| `--tensor-parallel` / `--gpu-memory` / `--max-model-len` | vLLM knobs |
| `--batch-size` | Contexts processed in parallel (default: `BATCH_SIZE`) |
| `--config` | Config filename without `.py` (default: `config_0`) |
| `--qa-only` + `--results-suffix` | Skip Phase 1, load snapshots, run QA only |

## Output

Written under `{module}/{config}_outputs_{model}_{size}/session_{start}_{end}/`:
`results_*.json`, `checkpoint_*.json`, `retrieval_logs/`, `memory_snapshots/`,
`prompt_log/`, `logs/`. Each result records `qa_results` (generated vs. ground-truth
letter, retrieved-memory metadata), `token_statistics`, and `memory_at_qa_start`.
