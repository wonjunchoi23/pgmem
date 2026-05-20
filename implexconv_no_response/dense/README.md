# Dense Retrieval Baseline (ImplexConv) — Batched, QA-Only

Dense retrieval baseline for the ImplexConv QA-only experiment protocol.

All conversation turns are stored as dense vector embeddings during Phase 1.
No LLM calls are made during memory construction — only QA answering (Phase 2)
uses the LLM. For each QA question, the top-k most similar turns are retrieved
by cosine similarity and provided as context.

Multiple sessions are processed in parallel. Embedding computations and QA
generation are both batched across sessions.

---

## Architecture

| File | Role |
|---|---|
| `dense_store.py` | Per-session vector memory store (`MemoryUnit`, `DenseMemoryStore`) |
| `run_experiment.py` | Batched experiment runner (Phase 1/2/3, checkpointing, logging) |
| `config.py` | Hyperparameters and output path helpers |
| `load_dataset.py` | ImplexConv dataset loader |

Main behavior:

- Each session has its own `DenseMemoryStore` (independent state).
- A single `SentenceTransformer` instance is shared across all sessions in a batch.
- Phase 1 produces no LLM calls — embeddings only.
- Phase 2 batches all QA prompts across all sessions and sends them in
  `QA_BATCH_SIZE` chunks via `generate_batch_raw()`.
- Memory snapshots are saved as `memories.json` (content + embeddings as float lists).

---

## Memory Unit

Each stored memory corresponds to one `(user_turn, assistant_turn)` pair:

```
User: {user utterance}
Assistant: {assistant utterance}
```

If the assistant turn is absent (last turn), only the user line is stored.
The memory is identified by the user turn's `(session_id, conv_id, turn_id)`.

---

## Experiment Flow

### Phase 1 — Memory Construction

For each turn position across the active session batch (interleaved):

1. Collect user utterances as retrieval queries for all active sessions.
2. **[batch encode]** queries → query embeddings.
3. For each session: retrieve top-k from its current store (**before** storing this turn).
4. Write retrieval log (`phase="prompt_construction"`).
5. Build pair contents `"User: ...\nAssistant: ..."` for each session.
6. **[batch encode]** pair contents → pair embeddings.
7. For each session: store `(content, embedding, session_id, conv_id, turn_id)`.

No LLM calls occur in Phase 1. Steps 2 and 6 use `SentenceTransformer.encode()` with `batch_size=32`.

After Phase 1: memory stats (`num_memories`, `total_content_tokens`) are captured.

### Phase 2 — QA Answering

After memory is frozen:

1. Collect all QA questions across all sessions.
2. **[batch encode]** questions → QA embeddings.
3. For each session/QA: retrieve top-k, write retrieval log (`phase="qa"`), build QA prompt.
4. **[batch generate]** all QA prompts in `QA_BATCH_SIZE` chunks (`call_1_qa`).
5. Distribute answers back to each session; track token usage.

### Phase 3 — Cleanup

Save memory snapshot → clear memory store → save results (atomic write) → update checkpoint.

---

## Retrieval Mechanism

Cosine similarity via `sklearn.metrics.pairwise.cosine_similarity`.

- Query: user utterance (Phase 1) or QA question (Phase 2)
- Index: all stored `(user, assistant)` pair embeddings in the session
- Returns: top-k `(MemoryUnit, score)` pairs sorted descending by score

---

## QA Prompt Structure

Retrieved memories are formatted as a numbered list:

```
[Memory 1] User: {utt}
Assistant: {utt}

[Memory 2] User: {utt}
...
```

If no memories are retrieved: `"No relevant memories found."`

Full prompt template:

```
You are a helpful assistant. Answer the question based only on the
retrieved memories provided. Be concise (max 100 words).

Retrieved memories:
{context}

Question: {question}

Answer in JSON format:
{"answer": "<your answer here>"}
```

- **opposed**: free-form answer, schema `{"answer": str}`
- **supportive**: constrained to `"yes"` or `"no"`, schema `{"answer": enum["yes","no"]}`

Answers for the supportive subset are normalized: if the raw answer is neither
`"yes"` nor `"no"` after lowercasing, it is recorded as `"unknown"`.

---

## Key Config Values

From `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence embedding model |
| `RETRIEVE_K` | `30` | Memories retrieved per query |
| `BATCH_SIZE` | `4` | Sessions processed together |
| `QA_BATCH_SIZE` | `64` | QA prompts per `generate_batch_raw()` call |
| `TEMPERATURE` | `0.7` | LLM generation temperature |
| `MAX_TOKENS` | `250` | LLM max output tokens |
| `JSON_RETRY` | `3` | Structured output retry count |
| `CONV_IDS_PER_DAY` | `2` | Virtual day granularity |
| `MINUTES_PER_TURN` | `10` | Virtual minutes per turn |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint during runs |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save per-session memory to disk |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Log all LLM calls to `prompt_log/` |

---

## LLM Call Logging

When `ENABLE_LLM_CALL_LOGGING = True`, QA calls are logged per session:

| Folder | Call |
|---|---|
| `call_1_qa/calls.jsonl` | QA answer generation |

Each line is a JSON object with fields `timestamp`, `call_type`, `system_prompt`,
`user_prompt`, `output` (excluding `_usage`).

Dense has no Phase 1 LLM calls, so this is the only call type.

---

## Token Tracking

| Key | Description |
|---|---|
| `call_1_qa` | QA answer generation — one call per QA question |
| `total_input` | Equal to `call_1_qa.input` |
| `total_output` | Equal to `call_1_qa.output` |
| `total_llm_calls` | Equal to `call_1_qa.llm_calls` |

---

## Usage

### Basic run

```bash
python dense/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config
```

### With `max_model_len` override

```bash
python dense/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config
```

### Supportive subset

```bash
python dense/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset supportive \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config
```

### Background run with `nohup`

```bash
CUDA_VISIBLE_DEVICES=0 nohup python dense/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config \
    > dense/nohup/nohup_opp_session_0_99.out 2>&1 &
```

### Merge results from multiple runs

```bash
python merge_results.py dense config_outputs_Qwen3-1.7B_opposed

# Dry run (show what would be merged)
python merge_results.py dense config_outputs_Qwen3-1.7B_opposed --dry-run
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start-session` | int | required | First session to process (inclusive) |
| `--end-session` | int | required | Last session to process (inclusive) |
| `--subset` | str | required | `opposed` or `supportive` |
| `--model` | str | config default | Model path (HF format or local) |
| `--tensor-parallel` | int | config default | Tensor parallelism degree |
| `--gpu-memory` | float | config default | GPU memory utilization (0.0–1.0) |
| `--max-model-len` | int | `None` | vLLM max context length (optional) |
| `--batch-size` | int | `BATCH_SIZE` | Sessions processed in parallel |
| `--config` | str | `"config"` | Config filename without `.py` |
| `--engine` | str | config default | Override LLM engine (`vllm`, `together`, `openai`) |

---

## Output Structure

```text
dense/
├── {config}_outputs_{model}_{subset}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{subset}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{id}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{id}/
│   │   │       └── memories.json
│   │   ├── prompt_log/
│   │   │   └── session_{id}/
│   │   │       └── call_1_qa/calls.jsonl
│   │   └── logs/
│   └── results_{model}_{subset}_merged.json
└── logs/
```

Results are saved atomically (temp file + rename) after each session.

---

## Result Schema

Each element of the top-level results JSON array:

```json
{
  "session_id": 1,
  "config_metadata": {
    "config_name": "config",
    "model": "Qwen/Qwen3-1.7B",
    "subset": "opposed",
    "embedding_model": "all-MiniLM-L6-v2",
    "retrieve_k": 30,
    "temperature": 0.7,
    "max_tokens": 250,
    "session_range": [0, 99]
  },
  "memory_at_qa_start": {
    "num_memories": 82,
    "total_content_tokens": 1640
  },
  "qa_results": [
    {
      "question": "Does the user prefer remote work?",
      "generated_answer": "yes",
      "ground_truth_answer": "yes",
      "retrieved_memories": [
        {"session_id": 1, "conv_id": 2, "turn_id": 3},
        {"session_id": 1, "conv_id": 0, "turn_id": 1}
      ],
      "qa_tokens": {"input": 820, "output": 12, "model": "Qwen/Qwen3-1.7B"}
    }
  ],
  "token_statistics": {
    "call_1_qa": {
      "input": 4100, "output": 60, "llm_calls": 5,
      "parse_fallback_count": 0
    },
    "total_input": 4100,
    "total_output": 60,
    "total_llm_calls": 5
  },
  "memory_snapshot_path": "memory_snapshots/session_1/"
}
```

`memory_snapshot_path` is `null` when `SAVE_MEMORY_SNAPSHOTS = False`.

---

## Retrieval Log

`retrieval_logs/session_{id}_retrieval_log.jsonl` — one JSON object per line.
One entry per turn (Phase 1) and per QA question (Phase 2).

```json
{
  "timestamp": "2026-04-09T12:00:00.000000",
  "phase": "prompt_construction",
  "session_id": 1,
  "conv_id": 2,
  "turn_id": 3,
  "query": "I actually prefer working from home.",
  "memory_type": "dense_vector",
  "num_retrieved": 5,
  "retrieved_items": [
    {
      "content_preview": "User: I like the flexibility of remote...",
      "score": 0.91,
      "source_turn": {"session_id": 1, "conv_id": 1, "turn_id": 2}
    }
  ],
  "module_specific": {
    "store_size_at_retrieval": 10
  }
}
```

For Phase 2 QA entries, `conv_id` and `turn_id` are `-1` (no associated turn).
Phase 1 entries at `turn_idx=0` will have `num_retrieved=0` (store is empty).
