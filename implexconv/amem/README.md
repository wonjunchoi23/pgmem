# A-MEM Memory Module (ImplexConv) — Batched Main Variant, QA-Only

A-MEM (Xu et al., 2025) memory module adapted for the ImplexConv QA-only
experiment protocol used in this repository.

`amem/` is the current main implementation. It keeps session memory states
independent, but batches internal LLM calls across multiple sessions to improve
GPU utilization. The archived sequential reference is kept under
`amem/amem_sequential/`.

Phase 1 stores memories only. No LLM response is generated during memory
construction — only QA answering is executed with the LLM in Phase 2.

---

## Architecture

Core files:

| File | Role |
|---|---|
| `memory_layer.py` | Agentic memory system, note construction, evolution, retrieval |
| `agent.py` | A-MEM experiment wrapper, QA answering, stats delegation |
| `run_experiment.py` | Batched multi-session experiment runner |
| `config_0.py` | Hyperparameters, output paths, batch settings |
| `load_dataset.py` | ImplexConv dataset loader |
| `batch_difference.md` | Notes on how the current batched main layout differs from the old sequential flow |

Main behavior:

- Each session keeps its own memory state.
- Multiple sessions are processed together in one batch.
- Internal A-MEM calls are grouped by step and sent through batched generation.
- A shared embedding model instance is reused across session agents.
- Memory snapshots are saved as JSON (`memories.json`, `retriever.json`).
- Stored memory content keeps speaker labels in the original A-MEM style:
  `Speaker user says: ...`, `Speaker assistant says: ...`.

---

## Experiment Flow

### Phase 1 — Memory Construction

For each turn position across the active session batch:

1. Batch note-construction prompts (`call_2_note_construction`).
2. Apply note-construction outputs.
3. Store no-evolve notes immediately.
4. Batch evolution prompts when needed (`call_3_evolution`).
5. Apply evolution outputs and store evolved notes.

User and assistant turns are both stored as separate ground-truth dialogue
turns.

After Phase 1: memory state stats and evolution/fallback stats are collected.

### Phase 2 — QA Answering

After memory is frozen:

1. Retrieve memory for every QA item.
2. Build QA prompts.
3. Batch QA generation in `QA_BATCH_SIZE` chunks (`call_4_qa`).
4. Write results back to the correct session.

### Phase 3 — Cleanup

Save memory snapshots, persist results, update checkpoint, and clear memory.

---

## Logging and Checkpointing

- Retrieval logs are written per session as JSONL for QA retrieval only.
- Prompt logs are grouped by call type:
  `call_2_note_construction`, `call_3_evolution`, `call_4_qa`
- Checkpoints use a set-based format:

```json
{"completed_session_ids": [0, 1, 2, 3]}
```

Older checkpoints using `last_completed_session_index` are still readable.

---

## Key Config Values

From [config_0.py](/NAS/wonjun/exp_implexconv_no_response/amem/config_0.py):

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence embedding model |
| `RETRIEVE_K` | `10` | Memories retrieved per query |
| `EVOLUTION_THRESHOLD` | `100` | Evolution threshold |
| `BATCH_SIZE` | `4` | Sessions processed together |
| `QA_BATCH_SIZE` | `64` | QA prompts per batch-generate call |
| `CONV_IDS_PER_DAY` | `2` | Virtual day granularity |
| `MINUTES_PER_TURN` | `10` | Virtual minutes per turn |
| `MAX_TOKENS` | `750` | LLM max output tokens |
| `JSON_RETRY` | `3` | Retry count for structured output |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint during runs |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save per-session memory snapshots |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Save prompt/output logs |

Dataset paths:

```text
dataset/implexconv/ImplexConv_opposed_processed.json
dataset/implexconv/ImplexConv_supportive_processed.json
```

---

## Usage

Run from the repository root.

### Basic run

```bash
python amem/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config_0
```

### With `max_model_len` override

```bash
python amem/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config_0
```

### Supportive subset

```bash
python amem/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset supportive \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config_0
```

### Background run with `nohup`

```bash
CUDA_VISIBLE_DEVICES=0 nohup python amem/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config_0 \
    > amem/nohup/nohup_opp_session_0_99.out 2>&1 &
```

### Merge results from multiple runs

Use the root-level `merge_results.py`.

```bash
python merge_results.py amem config_0_outputs_Qwen3-1.7B_opposed

python merge_results.py \
    amem \
    config_0_outputs_Qwen3-1.7B_opposed \
    --dry-run
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start-session` | int | required | First session to process |
| `--end-session` | int | required | Last session to process |
| `--subset` | str | required | `opposed` or `supportive` |
| `--model` | str | config default | Model path |
| `--tensor-parallel` | int | config default | Tensor parallelism |
| `--gpu-memory` | float | config default | GPU memory utilization fraction |
| `--max-model-len` | int | `None` | Optional vLLM max model length |
| `--batch-size` | int | `BATCH_SIZE` | Sessions processed in parallel |
| `--config` | str | `config_0` | Config filename without `.py` |

---

## Output Structure

```text
amem/
├── nohup/
├── {config}_outputs_{model}_{subset}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{subset}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{id}_retrieval_log.jsonl  # QA retrieval only
│   │   ├── memory_snapshots/
│   │   │   └── session_{id}/
│   │   │       ├── memories.json
│   │   │       ├── retriever.json
│   │   │       ├── embeddings.npy
│   │   │       └── metadata.json
│   │   ├── prompt_log/
│   │   │   └── session_{id}/
│   │   │       ├── call_2_note_construction/calls.jsonl
│   │   │       ├── call_3_evolution/calls.jsonl
│   │   │       └── call_4_qa/calls.jsonl
│   │   └── logs/
│   └── results_{model}_{subset}_merged.json
└── logs/
```

---

## Result Schema

Each element of the top-level results JSON array:

```json
{
    "session_id": <int>,
    "config_metadata": {
        "config_name": <str>,
        "model": <str>,
        "subset": <str>,
        "embedding_model": <str>,
        "retrieve_k": <int>,
        "evolution_threshold": <int>,
        "temperature": <float>,
        "max_tokens": <int>,
        "session_range": [<int>, <int>]
    },
    "memory_at_qa_start": {
        "num_memories": <int>,
        "total_content_tokens": <int>
    },
    "qa_results": [...],
    "token_statistics": {
        "call_2_note_construction": {
            "input": <int>, "output": <int>, "llm_calls": <int>,
            "parse_fallback_count": <int>
        },
        "call_3_evolution": {"input": <int>, "output": <int>, "llm_calls": <int>},
        "call_4_qa":        {"input": <int>, "output": <int>, "llm_calls": <int>},
        "total_input": <int>,
        "total_output": <int>,
        "total_llm_calls": <int>
    },
    "evolution_statistics": {
        "evo_triggered_count": <int>,
        "actions_taken": {"strengthen": <int>, "update_neighbor": <int>}
    },
    "memory_snapshot_path": <str or null>
}
```

---

## Notes

- Retrieval log entries use `"module": "amem"`.
- Supportive QA answers are normalized to `yes`, `no`, or fallback `unknown`.
- Memory snapshots are stored as JSON (not pickle). Embeddings remain in `.npy` format.
- Token count keys use `llm_calls` throughout (not `api_calls`).
- The archived `amem/amem_sequential/` directory is kept for reference, not as the primary execution path.
