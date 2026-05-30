# Experiment: Memory-augmented LLM Agents on ImplexConv (QA-Only)

QA-only protocol for the ImplexConv dataset. Phase 1 builds memory from ground-truth
(GT) conversation turns; the LLM is called **only for QA** (Phase 2). No conversational
response is generated — GT agent responses are used for memory construction.

## Dataset

- **ImplexConv**, two subsets via `--subset`: `opposed` / `supportive`
  (`dataset/implexconv/ImplexConv_{subset}_processed.json`).
- Each file is a list of session objects: `metadata`, `conversations` (flattened turns:
  session_id, conv_id, turn_id, global_turn_id, speaker, utterance), and `qa`
  (question, answer, opposed_implicit_reasoning, retrieved_conv_ids).
- **Metric**: opposed → free-form answer (accuracy, F1); supportive → `{yes, no}` (accuracy).
- QA exchanges are never added to dialogue history or stored in memory.

## Flow

Unit = one session `S_i` (i = start_session..end_session).

1. **Phase 1 — Build memory.** For each (user, assistant) turn: retrieve → store
   (GT agent response; QA excluded) → update module state → log retrieval. After all
   turns: finalize, record `memory_at_qa_start`, save snapshot.
2. **Phase 2 — QA.** Memory frozen (no new memory). For each QA: retrieve with the
   question as query → answer (`return_usage=True`) → log retrieval + record results.
   Supportive questions are 3rd-person; the prompt must frame the task as QA, not as a
   user utterance. Supportive answer ∈ `{yes, no}`.
3. **Phase 3 — Cleanup.** Aggregate token stats, save results (atomic write), clear all
   memory state, update checkpoint.

## Key rules

- **GT Agent Response Rule**: wherever an agent response is reused (storage, context,
  summarization), use the dataset GT response — never a generated one.
- QA question/answer pairs never enter dialogue history or memory.
- No LLM call for response generation in Phase 1.
- Memory fully cleared between sessions; snapshot saved before cleanup.
- **Virtual time**: `CONV_IDS_PER_DAY = 2` (2 conv_ids = 1 day), `TIME_PER_TURN_MINUTES = 10`
  (local turn_id). Reset per-day state when `conv_id // CONV_IDS_PER_DAY` changes.
- **Checkpoint** (atomic) after each session; supports `{"completed_session_ids": [...]}`
  (current) and `{"last_completed_session_index": int}` (legacy).
- **Token tracking**: every LLM call (QA + module-internal) uses `return_usage=True`,
  counted per call type as `{input, output, llm_calls}`; totals are the sums. Use the key
  `llm_calls` (not `api_calls`).
- Shared protocol knobs: `CHECKPOINT_INTERVAL=1`, `TEMPERATURE=0.7`, `MAX_TOKENS=750`.

## Memory module interface

Any module must implement: **Retrieve** (query → memories), **Store** (a turn; GT rule),
**Finalize** (end-of-phase synthesis), **get_memory_stats**
(`{num_memories, total_content_tokens}`), **Clear**, and **Retrieve-with-metadata**
(returns session_id/conv_id/turn_id for QA tracking). Internal architecture is free
(vector, graph, hybrid, …).

All modules share `{project_root}/llm_module/llm_client.py`:
`create_llm_client(engine=vllm|together|openai, ...)`, called via
`client.generate(prompt, system_prompt, guided_json=?, temperature, max_tokens,
json_retry=?, return_usage=True)`. Thinking mode is **off by default**
(`enable_thinking=False`); pass `enable_thinking=True` to enable (ignored if unsupported).

## Output

All modules emit the **same schema**: `results_*.json` = list of session results, each
with `session_id`, `config_metadata`, `memory_at_qa_start`
(`{num_memories, total_content_tokens}`), `qa_results` (question, generated_answer,
ground_truth_answer, module-specific `retrieved_memories`, `qa_tokens`),
`token_statistics` (per-call-type + totals), an optional module-specific Phase 1 stats
field, and `memory_snapshot_path`.

Per-module layout:

```text
{module}/{config}_outputs_{model}_{subset}/session_{start}_{end}/
├── results_{model}_{subset}_session_{start}_{end}.json
├── checkpoint_*.json
├── retrieval_logs/session_{id}_retrieval_log.jsonl
├── memory_snapshots/session_{id}/
├── prompt_log/session_{id}/{call_type}/calls.jsonl
└── logs/
```

### Run

```bash
python run_experiment.py \
    --start-session 0 --end-session 4 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 --gpu-memory 0.5 \
    --config config
```

`--config` is the config filename (without `.py`), loaded dynamically and used as the
output dir prefix `{config}_outputs_{model}_{subset}/`.

## Logging

- **LLM call logs**: `prompt_log/session_{id}/{call_type}/calls.jsonl`, one JSON/line
  (`timestamp, call_type, system_prompt, user_prompt, output`; `output` excludes
  `_usage`). Call types are module-specific. Toggle with `ENABLE_LLM_CALL_LOGGING`.
- **Retrieval logs**: `retrieval_logs/session_{id}_retrieval_log.jsonl`, one op/line
  (`timestamp, phase, session_id, conv_id, turn_id, query, num_retrieved,
  retrieved_items[{score, source_turn}], module_specific{}`). Exact format is
  module-specific.
