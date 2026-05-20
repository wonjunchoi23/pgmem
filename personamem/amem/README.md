# A-MEM Memory Module (PersonaMem) — Batched Variant, QA-Only

A-MEM (Xu et al., 2025) memory module adapted for the PersonaMem QA experiment
protocol used in this repository.

For the full experiment specification, see [`personamem_experiment.md`](../personamem_experiment.md).  
For differences from the ImplexConv protocol, see [`difference.md`](../difference.md).

`amem/` processes sessions (shared contexts) independently, batching internal LLM
calls across multiple sessions to improve GPU utilization.

Phase 1 stores memories only — no LLM response is generated during memory
construction. Only QA answering calls the LLM in Phase 2.

---

## Architecture

Core files:

| File | Role |
|---|---|
| `memory_layer.py` | Agentic memory system, note construction, evolution, retrieval |
| `agent.py` | A-MEM experiment wrapper, QA answering, stats delegation |
| `run_experiment.py` | Batched multi-session experiment runner |
| `config_0.py` | Hyperparameters, output paths, batch settings |
| `load_dataset.py` | PersonaMem dataset loader (CSV + JSONL) |

Main behavior:

- Each shared context keeps its own memory state.
- Multiple contexts are processed together in one batch.
- Internal A-MEM calls are grouped by step and sent through batched generation.
- A shared embedding model instance is reused across session agents.
- Memory snapshots are saved as JSON (`memories.json`, `retriever.json`).
- Stored memory content is the raw message content as-is (`"User: ..."` / `"Assistant: ..."`).
- `system` messages are used only as block boundary markers and are never stored.

---

## Experiment Flow

### Phase 1 — Memory Construction

For each message position across the active batch:

1. Skip `system` messages; record `block_idx` increment.
2. Batch note-construction prompts (`call_2_note_construction`).
3. Apply note-construction outputs.
4. Store no-evolve notes immediately.
5. Batch evolution prompts when needed (`call_3_evolution`).
6. Apply evolution outputs and store evolved notes.

Both `user` and `assistant` messages are stored as separate memory notes.

After Phase 1: memory state stats and evolution/fallback stats are collected.

### Phase 2 — QA Answering

After memory is frozen:

1. Retrieve memory for every QA item using the question as query.
2. Build QA prompt: retrieved memory + question + all 4 options.
3. Batch QA generation in `QA_BATCH_SIZE` chunks (`call_4_qa`).
4. Normalize answer to `a`, `b`, `c`, or `d` (fallback: `unknown`).
5. Write results back to the correct session.

### Phase 3 — Cleanup

Save memory snapshots, persist results, update checkpoint, clear memory.

---

## Logging and Checkpointing

- Retrieval logs are written per context as JSONL for QA retrieval only.
- Prompt logs are grouped by call type:
  `call_2_note_construction`, `call_3_evolution`, `call_4_qa`
- Checkpoints use a set-based format:

```json
{"completed_session_ids": [0, 1, 2, 3]}
```

---

## Key Config Values

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence embedding model |
| `RETRIEVE_K` | `5` | Memories retrieved per query |
| `EVOLUTION_THRESHOLD` | `100` | Evolution trigger threshold |
| `BATCH_SIZE` | `4` | Contexts processed together |
| `QA_BATCH_SIZE` | `64` | QA prompts per batch-generate call |
| `CONV_IDS_PER_DAY` | `1` | 1 block = 1 virtual day |
| `MINUTES_PER_TURN` | `10` | Virtual minutes per message |
| `MAX_TOKENS` | `750` | LLM max output tokens |
| `JSON_RETRY` | `5` | Retry count for structured output |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint during runs |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save per-context memory snapshots |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Save prompt/output logs |

Dataset paths (configured per benchmark size in `config_0.py`):

```text
dataset/questions_32k.csv           dataset/shared_contexts_32k.jsonl
dataset/questions_128k.csv          dataset/shared_contexts_128k.jsonl
dataset/questions_1M.csv            dataset/shared_contexts_1M.jsonl
```

---

## Usage

Run from the `amem/` directory or the repository root.

### Basic run (32k, all contexts)

```bash
python amem/run_experiment.py \
    --start-session 0 --end-session 36 \
    --benchmark-size 32k \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config_0
```

### With `max_model_len` override

```bash
python amem/run_experiment.py \
    --start-session 0 --end-session 36 \
    --benchmark-size 32k \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config_0
```

### 128k contexts

```bash
python amem/run_experiment.py \
    --start-session 0 --end-session 109 \
    --benchmark-size 128k \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 2 \
    --config config_0
```

### Background run with `nohup`

```bash
CUDA_VISIBLE_DEVICES=0 nohup python amem/run_experiment.py \
    --start-session 0 --end-session 36 \
    --benchmark-size 32k \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config_0 \
    > amem/logs/nohup_32k_session_0_36.out 2>&1 &
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start-session` | int | required | First context index (QA-count-sorted) |
| `--end-session` | int | required | Last context index (inclusive) |
| `--benchmark-size` | str | required | `32k`, `128k`, or `1M` |
| `--model` | str | config default | Model path |
| `--tensor-parallel` | int | config default | Tensor parallelism |
| `--gpu-memory` | float | config default | GPU memory utilization fraction |
| `--max-model-len` | int | `None` | Optional vLLM max model length |
| `--batch-size` | int | `BATCH_SIZE` | Contexts processed in parallel |
| `--config` | str | `config_0` | Config filename without `.py` |

---

## Output Structure

```text
amem/
├── logs/
├── {config}_outputs_{model}_{benchmark_size}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{benchmark_size}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{benchmark_size}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{context_index}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{context_index}/
│   │   │       ├── memories.json
│   │   │       ├── retriever.json
│   │   │       ├── embeddings.npy
│   │   │       └── metadata.json
│   │   ├── prompt_log/
│   │   │   └── session_{context_index}/
│   │   │       ├── call_2_note_construction/calls.jsonl
│   │   │       ├── call_3_evolution/calls.jsonl
│   │   │       └── call_4_qa/calls.jsonl
│   │   └── logs/
│   └── results_{model}_{benchmark_size}_merged.json
```

---

## Result Schema

Each element of the top-level results JSON array:

```json
{
    "context_index": <int>,
    "persona_id": <int>,
    "shared_context_id": <str>,
    "config_metadata": {
        "config_name":         <str>,
        "model":               <str>,
        "benchmark_size":      <str>,
        "embedding_model":     <str>,
        "retrieve_k":          <int>,
        "evolution_threshold": <int>,
        "temperature":         <float>,
        "max_tokens":          <int>,
        "session_range":       [<int>, <int>]
    },
    "memory_at_qa_start": {
        "num_memories":         <int>,
        "total_content_tokens": <int>
    },
    "qa_results": [
        {
            "question":                    <str>,
            "question_type":               <str>,
            "topic":                       <str>,
            "all_options":                 [<str>, <str>, <str>, <str>],
            "generated_answer":            <str>,
            "ground_truth_answer":         <str>,
            "end_index_in_shared_context": <int>,
            "retrieved_memories": [
                {"context_index": <int>, "block_idx": <int>, "local_msg_idx": <int>}
            ],
            "qa_tokens": {"input": <int>, "output": <int>, "model": <str>}
        }
    ],
    "token_statistics": {
        "call_2_note_construction": {
            "input": <int>, "output": <int>, "llm_calls": <int>,
            "parse_fallback_count": <int>
        },
        "call_3_evolution": {"input": <int>, "output": <int>, "llm_calls": <int>},
        "call_4_qa":        {"input": <int>, "output": <int>, "llm_calls": <int>},
        "total_input":     <int>,
        "total_output":    <int>,
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
- Multiple-choice answers are normalized to `a`, `b`, `c`, `d`, or fallback `unknown`.
- Memory snapshots are stored as JSON (not pickle). Embeddings remain in `.npy` format.
- Token count keys use `llm_calls` throughout (not `api_calls`).
- Session indices (`--start-session` / `--end-session`) index into the dataset
  sorted by QA count descending. Session 0 is the context with the most QA questions.
