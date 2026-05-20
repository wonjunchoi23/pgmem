# Dense Retrieval Memory Module — PrefEval Port

Dense retrieval baseline adapted for the PrefEval implicit-persona
**chained-cumulative** experiment. See `../experiment_prefeval.md` for the
shared experiment spec and `../difference.md` for the per-aspect comparison
with the ImplexConv variant.

The simplest module: stores all `(user, assistant)` turn pairs as dense
embeddings, retrieves by cosine similarity for QA. **Zero LLM calls during
ingestion** — the only LLM call type is `call_1_qa`.

---

## Files

| File | Role |
|---|---|
| `dense_store.py`     | `MemoryUnit` + `DenseMemoryStore`. Copied from ImplexConv with `load_snapshot()` added |
| `load_dataset.py`    | PrefEval loader (shared shape with `../amem/load_dataset.py`) |
| `run_experiment.py`  | Chained-cumulative runner with batch encode + batched QA + resume |
| `config_0.py`        | Hyperparameters, output paths |

---

## Experiment flow per checkpoint k

`samples[0..K]` is fixed (no sampling). For each `k = 0..K`:

1. **Ingest session k** (Q1 = b: no Phase 1 retrieval):
   - Build `(user, assistant)` pair contents for all turns of session k.
   - Batch encode via `SentenceTransformer.encode(...)`.
   - Store each `(content, embedding, session_id, conv_id, turn_id)` into
     the shared store.
   - **Zero LLM calls.**
2. **Batched QA over q_0..q_k**:
   - Batch encode all `(k+1)` questions.
   - For each j ∈ [0, k]: retrieve top-`RETRIEVE_K` by cosine, build prompt
     `[Memory 1] ... [Memory RETRIEVE_K] ...` + question.
   - Batch generate (chunked by `QA_BATCH_SIZE`).
3. **Snapshot `m_k`** — saved after QA succeeds. Snapshot existence ⇔ QA done.
4. **Stats** — update cumulative `stats.json`.

### Q1 (b) policy: no Phase 1 retrieval

ImplexConv runner did per-turn retrieval and wrote `phase="prompt_construction"`
log entries during Phase 1. These were originally for response-generation
context, but in the QA-only protocol they are unused. The PrefEval port
**drops these entirely** — Phase 1 is pure embed + store, much faster, and
`retrieval_log.jsonl` only contains Phase 2 QA-time retrievals.

---

## Resume behavior

`dense_store.save_snapshot()` writes `memories.json` (content + float-list
embeddings + source-turn metadata). `load_snapshot()` (added in this port)
restores the full state. No additional runner-side reconstruction needed.

On startup the runner scans `memory_snapshots/m_*/`:

- No snapshots → start fresh from `k = 0`.
- `K_target ≤ k_existing` → no-op, log "already complete".
- `K_target > k_existing` → `load_snapshot(m_{k_existing})`, continue from
  `k_existing + 1`.

---

## Key config values (`config_0.py`)

| Parameter | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL`    | `"all-MiniLM-L6-v2"` | |
| `RETRIEVE_K`         | 20 | Top-k retrieved per QA query |
| `ENCODE_BATCH_SIZE`  | 32 | SentenceTransformer internal batch size |
| `QA_BATCH_SIZE`      | 64 | QA prompts per batch generate |
| `MAX_TOKENS`         | **1500** | Raised from 750 — PrefEval answers advisory |
| `JSON_RETRY`         | **10** | Raised from 5 |
| QA word cap          | **200 words** | In `QA_PROMPT` template |

Dataset path: `dataset/implicit_persona.json` (single file; no subset split).

---

## Usage

```bash
python run_experiment.py \
    --end-session 10 \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --config config_0
```

Resume:

```bash
python run_experiment.py --end-session 10 --model <M> ...
# Later — picks up from m_10, ingests 11..90, appends new rows
python run_experiment.py --end-session 90 --model <M> ...
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--end-session`      | int   | required        | Last sample index in chain |
| `--model`            | str   | config default  | Model path |
| `--tensor-parallel`  | int   | config default  | Tensor parallelism |
| `--gpu-memory`       | float | config default  | GPU memory utilization |
| `--max-model-len`    | int   | None            | Optional vLLM max model length |
| `--config`           | str   | `config_0`      | Config filename without `.py` |

---

## Output structure

```
exp_prefeval/dense/
├── config_0_outputs_<model>/
│   ├── results.jsonl          # one row per (k, question_session)
│   ├── retrieval_log.jsonl    # one row per QA-time retrieval (no Phase 1 entries)
│   ├── stats.json             # cumulative across all runs
│   ├── meta.json              # config metadata, written once
│   ├── memory_snapshots/
│   │   ├── m_0/memories.json
│   │   ├── m_1/memories.json
│   │   └── m_K/memories.json
│   ├── prompt_log/
│   │   └── call_1_qa/calls.jsonl
│   └── logs/
└── logs/
```

### `results.jsonl` row schema

```json
{
  "k":                3,
  "question_session": 1,
  "question":         "...",
  "model_answer":     "...",
  "topic":            "...",
  "persona":          "...",
  "preference":       "...",
  "explanation":      "...",
  "retrieved_memories": [
    {"session_id": 0, "conv_id": 1, "turn_id": 4, "score": 0.81}
  ],
  "qa_tokens": {"input": 0, "output": 0, "model": "..."}
}
```

### `stats.json` schema

```json
{
  "call_1_qa": {"input": 0, "output": 0, "llm_calls": 0},
  "checkpoints_completed": [0, 1, 2, ...]
}
```

Only one call type because dense ingestion is LLM-free.

---

## Notes on differences vs ImplexConv Dense

| Aspect | ImplexConv | PrefEval port |
|---|---|---|
| QA subset | opposed/supportive | **opposed-style only**, 200-word cap |
| Phase 1 retrieval | Per-turn retrieve + log `phase="prompt_construction"` | **Removed** (Q1=b). Phase 1 is embed+store only |
| Cross-session batching | Phase 1 turns batched across N sessions | N/A (chain count = 1). Per-session encode batched internally by SentenceTransformer |
| Memory between sessions | Cleared between sessions | **Persisted across whole chain** |
| Memory state | `MemoryUnit` keyed by `(session_id, conv_id, turn_id)`; `session_id` varies | `session_id` always 0; `conv_id` = chain position |

---

## Notes

- Snapshot saved after QA succeeds. Mid-QA crash → resume re-encodes session
  k's turns from `m_{k-1}` (cheap — embedding only, no LLM).
- The store has no forgetting / clearing — it monotonically grows. By
  checkpoint K the store contains all turn pairs from sessions 0..K.
- The `Turn` dataclass populates `global_turn_id` cumulatively across the
  chain; dense does not consume it but other modules (Theanine) do.
