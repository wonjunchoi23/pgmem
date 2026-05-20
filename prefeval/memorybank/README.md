# MemoryBank Memory Module — PrefEval Port

MemoryBank (Zhong et al., AAAI 2024) memory module adapted for the PrefEval
implicit-persona **chained-cumulative** experiment. See
`../experiment_prefeval.md` for the shared experiment spec and
`../difference.md` for the per-aspect comparison with the ImplexConv variant.

---

## Files

| File | Role |
|---|---|
| `memory_bank.py`   | MemoryBankSystem: forgetting curve, hierarchical summarization, retrieval. **Verbatim copy** from the ImplexConv variant |
| `retriever.py`     | EmbeddingRetriever (cosine similarity). **Verbatim copy** |
| `agent.py`         | MemoryBankAgent + LLMCallLogger. Trimmed: no subset branching, QA cap 200 words. Adds `load_memory_snapshot` |
| `load_dataset.py`  | PrefEval loader (shared shape with `../amem/load_dataset.py`) |
| `run_experiment.py` | Chained-cumulative runner with resume |
| `config_0.py`      | Hyperparameters, output paths, batch settings |

---

## Experiment flow per checkpoint k

`samples[0..K]` is fixed (no sampling). For each `k = 0..K` (incremental):

1. **Forgetting** — `apply_forgetting(k)` if `k > 0` (boundary `k-1 → k`).
2. **Ingest session k** — for each `(user, assistant)` pair (sequential):
   - `add_memory(snippet, conv_id=k, timestamp="0000_{k:04d}_{turn_id:04d}")`
   - append pair to `history_buffer`
3. **Daily summary for session k** — `on_conv_boundary(k, dialogue_text)`
   - 2 LLM calls: `call_2_daily_event` + `call_3_daily_personality`
   - **One daily summary per session** (matches original MemoryBank's
     "1 date = 1 summary"; differs from ImplexConv adapter which batched
     `CONVS_PER_DAY=2` together)
4. **Global synthesis** — `on_session_end()`
   - 2 LLM calls: `call_4_global_event` + `call_5_global_personality`
   - Forced at every checkpoint so `user_portrait` reflects all sessions
     `0..k` before QA. Matches the original's "always re-synthesize after
     dailies".
5. **QA over `q_0..q_k`** — batched (chunks of `QA_BATCH_SIZE`):
   - For each `j ∈ [0, k]`: retrieve with `current_conv_id=k`,
     `update_strength=False` (memory is frozen during QA).
   - Build prompt: `[user_portrait]` + `[memory + memo_dates]` + `[example]`
     + `[history (last HISTORY_CONV_WINDOW conv_ids)]` + question.
   - Append row to `results.jsonl`; append retrieval entry to `retrieval_log.jsonl`.
6. **Snapshot `m_k`** — saved **after** QA succeeds. Snapshot existence ⇔ QA done.
7. **Stats** — update cumulative `stats.json`.

---

## Resume behavior

On startup the runner scans `memory_snapshots/m_*/`:

- No snapshots → start fresh from `k = 0`, empty `history_buffer`.
- `K_target ≤ k_existing` → no-op, log "already complete".
- `K_target > k_existing`:
  - `load_memory_snapshot(m_{k_existing})` restores the memory_system
    (entries, embeddings, daily/global summaries, daily-summary index).
  - Reconstruct `history_buffer` by iterating `sessions[0..k_existing]`
    (no LLM calls — pure data iteration).
  - Continue from `k_existing + 1`.

`day_batch_turns` / `conv_turn_pairs` are not maintained as runner state
because each session is summarized in one shot at its own checkpoint.

Resume only works if config (model, retrieve_k, forgetting_divisor, prompt
templates, etc.) is unchanged between runs.

---

## Key config values (`config_0.py`)

| Parameter | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL`        | `"all-MiniLM-L6-v2"` | |
| `RETRIEVE_K`             | 6  | Original `VECTOR_SEARCH_TOP_K` |
| `FORGETTING_DIVISOR`     | 5  | `retention = exp(-day_gap / (DIVISOR * S))` |
| `CONVS_PER_DAY`          | 2  | Same as ImplexConv (used for time-formatting only; daily-summary granularity is now per-session) |
| `MINUTES_PER_TURN`       | 10 | Same as ImplexConv |
| `HISTORY_CONV_WINDOW`    | 2  | Recent conv_ids shown in QA prompt |
| `SUMMARIZE_TEMPERATURE`  | 0.7 | |
| `SUMMARIZE_MAX_TOKENS`   | 400 | |
| `QA_BATCH_SIZE`          | 64 | Chunk size for batched QA at a checkpoint |
| `MAX_TOKENS`             | **1500** | Raised from 750 — PrefEval answers advisory |
| QA word cap              | **200 words** | In `agent.QA_PROMPT_SUFFIX` |

Dataset path: `dataset/implicit_persona.json` (single file; no subset split).

`GLOBAL_SUMMARY_INTERVAL` was removed: at every checkpoint we re-synthesize
global summaries unconditionally.

---

## Usage

Run from the `exp_prefeval/memorybank/` directory.

```bash
python run_experiment.py \
    --end-session 10 \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --config config_0
```

With `max_model_len`:

```bash
python run_experiment.py \
    --end-session 10 \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --max-model-len 8000 \
    --config config_0
```

Extending an existing run (resume):

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
exp_prefeval/memorybank/
├── config_0_outputs_<model>/
│   ├── results.jsonl            # one row per (k, question_session)
│   ├── retrieval_log.jsonl      # one row per QA-time retrieval
│   ├── stats.json               # cumulative across all runs
│   ├── meta.json                # config metadata, written once
│   ├── memory_snapshots/
│   │   ├── m_0/{entries.json, embeddings.npy, corpus.json, summaries.json, metadata.json}
│   │   ├── m_1/...
│   │   └── m_K/...
│   ├── prompt_log/
│   │   ├── call_2_daily_event/calls.jsonl
│   │   ├── call_3_daily_personality/calls.jsonl
│   │   ├── call_4_global_event/calls.jsonl
│   │   ├── call_5_global_personality/calls.jsonl
│   │   └── call_6_qa/calls.jsonl
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
    {"memory_subtype": "dialogue_snippet", "source_label": "Day 1, 00:00",
     "score": 0.72, "session_id": 0, "conv_id": 1, "turn_id": 4}
  ],
  "qa_tokens": {"input": 0, "output": 0, "model": "..."}
}
```

`retrieved_memories[].memory_subtype` is `"dialogue_snippet"` or
`"daily_summary"` (MemoryBank retrieves from both).

### `stats.json` schema

```json
{
  "call_2_daily_event":        {"input": 0, "output": 0, "llm_calls": 0},
  "call_3_daily_personality":  {"input": 0, "output": 0, "llm_calls": 0},
  "call_4_global_event":       {"input": 0, "output": 0, "llm_calls": 0},
  "call_5_global_personality": {"input": 0, "output": 0, "llm_calls": 0},
  "call_6_qa":                 {"input": 0, "output": 0, "llm_calls": 0},
  "forgetting": {
    "forgetting_events_count": 0,
    "memories_forgotten":      0
  },
  "checkpoints_completed": [0, 1, 2, ...]
}
```

---

## Notes on differences vs ImplexConv MemoryBank

| Aspect | ImplexConv | PrefEval port |
|---|---|---|
| Daily summary granularity | 1 per `CONVS_PER_DAY` virtual day (= 2 conv_ids batched) | **1 per session** (matches original MemoryBank) |
| Global synthesis trigger | Every `GLOBAL_SUMMARY_INTERVAL=10` conv_ids + Phase 1 end | **Every checkpoint, unconditional** |
| LLM calls per checkpoint (excl. QA) | Variable | **Fixed: 4** (call_2, call_3, call_4, call_5) |
| Memory between sessions | Cleared between sessions | **Persisted across the whole chain** |
| QA history | Last 2 conv_ids of session | Last 2 conv_ids of chain (sessions `[k-1, k]`) |
| QA subset | opposed/supportive | **opposed-style only**, 200-word cap |
| Cross-session batching | Phase 1 prompts batched across N sessions | N/A (chain count = 1) |
- Memory snapshot saved after QA succeeds. Mid-QA crash → resume re-ingests
  session k from `m_{k-1}` (small redundancy, cleaner invariant).
- `Turn` dataclass populates `global_turn_id` cumulatively across the chain;
  MemoryBank does not consume it but other modules (Theanine) do.
