# Dense Retrieval Baseline — PersonaMem

Dense retrieval baseline for the PersonaMem QA experiment. Stores all conversation turn pairs as dense embeddings and retrieves them by cosine similarity. **No LLM calls during Phase 1** — only embedding + cosine similarity.

For the full experiment specification, see [`personamem_experiment.md`](../personamem_experiment.md).

---

## Overview

- **Phase 1 (Memory Construction)**: PersonaMem messages are paired into `(user, assistant)` turns within each block. Each pair is embedded with SentenceTransformer and stored in a flat vector store. No LLM calls.
- **Phase 2 (QA Answering)**: QA questions are embedded, top-K most similar memories are retrieved, and the LLM answers the multiple-choice question (`call_1_qa`).
- **Phase 3 (Cleanup)**: Memory snapshots saved, stores cleared, results written.
- Multiple contexts processed concurrently; embedding is batched across contexts.

---

## Architecture

| File | Role |
|---|---|
| `run_experiment.py` | Batched multi-context experiment runner |
| `dense_store.py` | `DenseMemoryStore` — flat vector store with cosine retrieval |
| `config_0.py` | Hyperparameters and output path helpers |
| `load_dataset.py` | PersonaMem dataset loader (CSV + JSONL) |

---

## PersonaMem Adaptation

### Message Pairing

PersonaMem stores individual user and assistant messages. The runner pairs them within each block before storing:

```
block 0:  user[0] + assistant[1]  → pair (block_idx=0, pair_idx=0)
          user[2] + assistant[3]  → pair (block_idx=0, pair_idx=1)
          ...
block 1:  user[0] + assistant[1]  → pair (block_idx=1, pair_idx=0)
          ...
```

Content stored per pair:
```
"User: {stripped_user_utterance}\nAssistant: {stripped_assistant_utterance}"
```

`"User: "` / `"Assistant: "` prefixes already embedded in PersonaMem content are stripped before re-adding, to avoid double-prefixing.

### Concept Mapping

| PersonaMem | dense_store internal | Description |
|---|---|---|
| `context_index` | `session_id` | Experiment unit identifier |
| `block_idx` | `conv_id` | Block within a context |
| `pair_idx_in_block` | `turn_id` | Turn pair order within a block |

---

## Retrieval

Phase 1 retrieval (before storing each turn, for logging):
- Query: stripped user utterance
- Retrieves from memories stored so far in this context

Phase 2 retrieval (QA):
- Query: QA question text
- Retrieves top-`RETRIEVE_K` memories by cosine similarity

---

## Usage

### Basic run (32k, all contexts)

```bash
python dense/run_experiment.py \
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
python dense/run_experiment.py \
    --start-session 0 --end-session 36 \
    --benchmark-size 32k \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config_0
```

### Background run with `nohup`

```bash
CUDA_VISIBLE_DEVICES=0 nohup python dense/run_experiment.py \
    --start-session 0 --end-session 36 \
    --benchmark-size 32k \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config_0 \
    > dense/nohup/nohup_32k_session_0_36.out 2>&1 &
```

### CLI Arguments

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
| `--engine` | str | `None` | Override LLM engine (vllm/together/openai) |

---

## Configuration (`config_0.py`)

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | SentenceTransformer model for encoding |
| `RETRIEVE_K` | `20` | Top-K memories retrieved per QA question |
| `TEMPERATURE` | `0.7` | LLM generation temperature |
| `MAX_TOKENS` | `750` | Max output tokens for QA |
| `JSON_RETRY` | `5` | JSON structured output retry count |
| `BATCH_SIZE` | `4` | Contexts processed per batch |
| `QA_BATCH_SIZE` | `64` | QA questions per batched LLM call |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint after each context |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save embedding store after each context |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Log QA prompts/outputs |

---

## Output Structure

```
dense/
├── logs/
└── {config}_outputs_{model}_{benchmark_size}/
    ├── session_{start}_{end}/
    │   ├── results_{model}_{benchmark_size}_session_{start}_{end}.json
    │   ├── checkpoint_{model}_{benchmark_size}_session_{start}_{end}.json
    │   ├── retrieval_logs/
    │   │   └── session_{context_index}_retrieval_log.jsonl
    │   ├── memory_snapshots/
    │   │   └── session_{context_index}/
    │   │       └── memories.json
    │   ├── prompt_log/
    │   │   └── session_{context_index}/
    │   │       └── call_1_qa/calls.jsonl
    │   └── logs/
    └── results_{model}_{benchmark_size}_merged.json
```

---

## Result Schema

```json
{
  "context_index":     0,
  "persona_id":        42,
  "shared_context_id": "<sha256>",

  "config_metadata": {
    "config_name":    "config_0",
    "model":          "Qwen/Qwen3-1.7B",
    "benchmark_size": "32k",
    "embedding_model": "all-MiniLM-L6-v2",
    "retrieve_k":     20,
    "temperature":    0.7,
    "max_tokens":     750,
    "session_range":  [0, 36]
  },

  "memory_at_qa_start": {
    "num_memories":         169,
    "total_content_tokens": 4230
  },

  "qa_results": [
    {
      "question":                    "Which option best describes the user's music preference?",
      "question_type":               "acknowledge_latest_preferences",
      "topic":                       "musicRecommendation",
      "all_options":                 ["(a) classical", "(b) jazz", "(c) indie rock", "(d) pop"],
      "generated_answer":            "c",
      "ground_truth_answer":         "c",
      "end_index_in_shared_context": 87,
      "retrieved_memories": [
        {"context_index": 0, "block_idx": 2, "pair_idx": 1},
        {"context_index": 0, "block_idx": 0, "pair_idx": 3}
      ],
      "qa_tokens": {"input": 620, "output": 12, "model": "Qwen/Qwen3-1.7B"}
    }
  ],

  "token_statistics": {
    "call_1_qa": {
      "input":               6400,
      "output":              280,
      "llm_calls":           16,
      "parse_fallback_count": 0
    },
    "total_input":     6400,
    "total_output":    280,
    "total_llm_calls": 16
  },

  "memory_snapshot_path": "memory_snapshots/session_0/"
}
```

**Notes:**
- `generated_answer` is one of `a`, `b`, `c`, `d`, or `"unknown"`.
- `retrieved_memories[*].pair_idx` is the turn-pair index within the block.
- `memory_at_qa_start.num_memories` = total pairs stored = number of turn pairs in the context.
- No Phase 1 LLM calls → `token_statistics` only contains `call_1_qa`.

---

## Memory Snapshot Format

Each context's snapshot is saved to `memory_snapshots/session_{context_index}/memories.json`:

```json
[
  {
    "content":    "User: Hi there! I've been diving into music...\nAssistant: That sounds great!...",
    "embedding":  [0.123, -0.045, ...],
    "session_id": 0,
    "conv_id":    0,
    "turn_id":    0
  }
]
```

Note: internally `session_id`/`conv_id`/`turn_id` store `context_index`/`block_idx`/`pair_idx`.

---

## Notes

- `dense_store.py` is unchanged — dataset-agnostic. Values passed as `session_id`/`conv_id`/`turn_id` internally map to `context_index`/`block_idx`/`pair_idx`.
- Session indices (`--start-session` / `--end-session`) index into the dataset sorted by QA count descending. Session 0 is the context with the most QA questions.
- Phase 1 has **no LLM calls**. GPU is only used during Phase 2 QA generation.
