# Dense Retrieval Baseline — LoCoMo, Batched, QA-Only

Dense retrieval baseline for the LoCoMo QA-only experiment protocol.

## Overview

- Phase 1 stores every LoCoMo turn as a dense memory.
- Each memory is a single turn formatted with speaker prefix:
  `Speaker {speaker} says: {text}`
- No LLM response generation happens during memory construction.
- Phase 2 retrieves top-k memories for each QA and answers with one LLM call
  (`call_1_qa`).
- Retrieved QA context includes session date/time:
  `[Memory N] Session {session_id} | {date_time}`

## Files

- `load_dataset.py`: LoCoMo dataset loader
- `dense_store.py`: per-sample dense memory store
- `run_experiment.py`: batched LoCoMo runner
- `config_0.py`: configuration and output path helpers
- `merge_results.py`: merge `sample_*` result folders

## Dataset Assumptions

- Dataset file: `dataset/locomo10.json`
- Unit of processing: `sample`
- Each sample contains multiple sessions and sample-level QA
- Image turns are converted at load time to:
  `[Image: {blip_caption}] {text}`

## Memory Format

Each memory snapshot entry stores:

- `content`
- `embedding`
- `sample_id`
- `session_id`
- `dia_id`
- `speaker`
- `date_time`

Snapshots are saved as JSON to:

- `memory_snapshots/sample_{sample_id}/memories.json`

## Logging

- Retrieval logs:
  `retrieval_logs/sample_{sample_id}_retrieval_log.jsonl`
- LLM call logs:
  `prompt_log/sample_{sample_id}/call_1_qa/calls.jsonl`

Terminology uses `llm_calls` throughout, not `api_calls`.

## Result Schema Highlights

Each sample result includes:

- `sample_id`
- `config_metadata`
- `memory_at_qa_start`
- `qa_results`
- `token_statistics`
- `memory_snapshot_path`

`token_statistics` uses the LoCoMo-style schema:

```json
{
  "call_1_qa": {
    "input": 0,
    "output": 0,
    "llm_calls": 0
  },
  "total_input": 0,
  "total_output": 0,
  "total_llm_calls": 0
}
```

## Example Run

```bash
python dense/run_experiment.py \
  --start-sample 0 \
  --end-sample 99 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel 1 \
  --gpu-memory 0.5 \
  --batch-size 4 \
  --config config_0
```

## Merge Results

```bash
python dense/merge_results.py dense/config_0_outputs_Llama-3.1-8B-Instruct
```
