# MemoryBank — Sequential Archived Reference

Archived sequential MemoryBank reference.
The current batched main implementation lives in `memorybank/`.

MemoryBank (Zhong et al., AAAI 2024) memory module for the ImplexConv experiment
(**QA-only variant** — response generation LLM calls are removed).
Stores dialogue turns as dense embeddings, retrieves via cosine similarity,
and builds hierarchical summaries (event + personality) that are injected into prompts.

Paper: *MemoryBank: Enhancing Large Language Models with Long-Term Memory*

---

## Overview

- **Phase 1 (Memory Construction)**: For each (user\_turn, assistant\_turn) pair in
  a session, MemoryBank retrieves relevant memories (top-k cosine similarity), builds
  the response generation prompt (for logging — **no LLM call is made**), then stores
  both the user utterance and the **GT assistant response** as new memory entries
  (embedding only — no LLM call at storage time).
  At each **conv\_id boundary** (= "new day"), daily event and personality summaries
  are generated. Every `GLOBAL_SUMMARY_INTERVAL` conv\_ids, an intermediate global
  synthesis is also run.
- **Phase 2 (QA Answering)**: After all turns are processed, QA pairs are answered
  using memory retrieval. The question text is used directly as the retrieval query
  (no keyword generation), following the original MemoryBank approach. No new memories
  are written during this phase.
- **Phase 1 End / Phase 2 Transition**: Global summaries are always synthesized
  unconditionally from all accumulated daily summaries before QA begins.
- Memory is **cleared between sessions**.
- **GT agent responses are always used** — no generated responses are fed back into
  memory or dialogue history.
- **QA exchanges are never stored in memory** and do not appear in dialogue history.

---

## Key Design: MemoryBank vs Other Modules

### Forgetting Curve (Ebbinghaus-Inspired)

Each memory entry tracks `strength` (S, starts at 1) and `last_recall_conv_id`.
At each conv\_id boundary, probabilistic deletion is applied:

```
retention = exp( -day_gap / (FORGETTING_DIVISOR * S) )
day_gap   = conv_gap / CONVS_PER_DAY

if random() > retention → permanently delete
```

When a memory is recalled (retrieved in top-k during Phase 1), its strength is
incremented (`S += 1`) and `last_recall_conv_id` is reset. During Phase 2 (QA),
`update_strength=False` — memory is frozen.

### Virtual Time Model

```
CONVS_PER_DAY   = 2   → every 2 consecutive conv_ids = 1 virtual day
MINUTES_PER_TURN = 10  → each local turn_id = 10 virtual minutes
```

These parameters are shared across all modules for consistency. `MINUTES_PER_TURN`
is declared in config for uniformity but is not directly used in MemoryBank logic.

### Hierarchical Summarization

At each conv\_id boundary ("end of day"), two LLM summaries are generated:

1. **Daily Event Summary** (`call_2_daily_event`) — concise digest of the day's conversations
2. **Daily Personality Summary** (`call_3_daily_personality`) — user traits and response strategies

Every `GLOBAL_SUMMARY_INTERVAL` conv\_ids (and unconditionally at Phase 1 end), these
are synthesized into:

3. **Global Event Summary** (`call_4_global_event`) — bird's-eye view of all events
4. **Global User Portrait** (`call_5_global_personality`) — overall personality understanding

### Prompt Structure

Response prompt (Phase 1, not executed) and QA prompt (Phase 2) both use the same layout:

```
[Event Summary]   Global event summary (or daily concatenation if global not yet available)
[User Portrait]   Global user portrait (or daily concatenation)
[Memory]          Top-k retrieved memories (formatted as bullet list)
[History]         Recent conversation turns (last HISTORY_CONV_WINDOW conv_ids)
[User]            Current utterance / QA question
[AI]              (response prompt: empty suffix for logging; QA: LLM generates answer)
```

### No LLM at Storage Time

Unlike A-MEM (which extracts keywords/tags/context via LLM per note), MemoryBank
stores memories using only embedding — no LLM call at write time. Internal LLM
calls are limited to summarization (at conv\_id boundaries and session end).

---

## Configuration (`config_0.py`)

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence embedding model |
| `RETRIEVE_K` | `6` | Memories retrieved per query (aligned with original paper's `VECTOR_SEARCH_TOP_K`) |
| `FORGETTING_DIVISOR` | `5` | Divisor in forgetting curve: `exp(-day_gap / (DIVISOR * S))` |
| `CONVS_PER_DAY` | `2` | Number of conv\_ids that constitute one virtual day |
| `MINUTES_PER_TURN` | `10` | Virtual minutes per local turn\_id (declared for cross-module consistency) |
| `GLOBAL_SUMMARY_INTERVAL` | `10` | Run global synthesis every N conv\_ids (= 5 virtual days); always runs at Phase 1 end |
| `HISTORY_CONV_WINDOW` | `2` | Number of conv\_ids to look back for history in the prompt |
| `TEMPERATURE` | `0.7` | LLM generation temperature |
| `MAX_TOKENS` | `750` | Max output tokens per QA generation |
| `JSON_RETRY` | `3` | Retry count on JSON parse failure |
| `SUMMARIZE_TEMPERATURE` | `0.7` | LLM temperature for summarization calls |
| `SUMMARIZE_MAX_TOKENS` | `400` | Max tokens for summarization outputs |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint after each session |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save memory snapshots after each session |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Log all LLM calls (prompts + outputs) per session |
| `LLM_ENGINE` | `"vllm"` | Engine: `"vllm"`, `"together"`, or `"openai"` |

Dataset paths (auto-selected by `--subset`):

```
dataset/implexconv/ImplexConv_opposed_processed.json
dataset/implexconv/ImplexConv_supportive_processed.json
```

---

## Usage

```bash
python run_experiment.py \
    --start-session 0 --end-session 4 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --config config_0

# Merge results from multiple runs
python merge_results.py \
    --model Qwen/Qwen3-1.7B \
    --subset opposed \
    --config config_0
```

`--config` specifies the config file name (without `.py`) in the
`memorybank/memorybank_sequential/` directory. The config name is used as the output directory prefix:
`{config}_outputs_{model}_{subset}/`.

---

## Module Structure

```
memorybank/
├── config_0.py            # Configuration & path helpers
├── load_dataset.py        # ImplexConv dataset parser (shared schema)
├── retriever.py           # EmbeddingRetriever — sentence-transformers cosine similarity
├── memory_bank.py         # MemoryBankSystem — forgetting curve + hierarchical summarizer
├── agent.py               # MemoryBankAgent + LLMCallLogger — experiment-facing wrapper
├── run_experiment.py      # Experiment orchestrator (Phase 1/2 loop, QA-only)
├── merge_results.py       # Result aggregation across session ranges
├── README.md
└── nohup/
```

### Module Responsibilities

**`retriever.py`** — Embedding retrieval
- `EmbeddingRetriever`: sentence-transformers based cosine similarity search
  - `add_document(text)` — embed and append to corpus
  - `search_with_scores(query, k)` — return top-k (index, score) pairs
  - `remove_by_indices(indices)` — delete entries (used by forgetting curve)
  - `save(directory)` / `load(directory)` — persist embeddings and corpus

**`memory_bank.py`** — Core MemoryBank system
- `MemoryEntry`: dataclass with `content`, `timestamp`, `conv_id`, `strength`, `last_recall_conv_id`
- `MemoryBankSystem`: vector store with forgetting curve and summarization
  - `add_memory(content, conv_id, timestamp)` — embed and store (no LLM call)
  - `retrieve(query, k, current_conv_id, update_strength)` — cosine top-k → optional strength update
  - `apply_forgetting(current_conv_id)` — probabilistic deletion at conv\_id boundaries
  - `summarize_daily(conv_id, dialogue_text)` — 2 LLM calls (`call_2_daily_event`, `call_3_daily_personality`)
  - `synthesize_global()` — 2 LLM calls (`call_4_global_event`, `call_5_global_personality`)
  - `get_event_summary()` / `get_user_portrait()` — global > daily concatenation fallback
  - `set_llm_logger(logger)` — inject per-session `LLMCallLogger`
  - `clear()` / `save_snapshot()` / `load_snapshot()`

**`agent.py`** — Experiment interface
- `LLMCallLogger`: per-session JSONL logger for all LLM calls (6 call types)
- `MemoryBankAgent`: wraps `MemoryBankSystem` + LLM client
  - `add_memory(content, conv_id, timestamp)` — store memory entry
  - `retrieve_memory(query, current_conv_id, k, update_strength)` — returns `RetrievalResult`
  - `build_response_prompt(user_utterance, retrieved_memory, history)` — construct prompt, log to `call_1_response` (output=null), return prompt string for token estimation
  - `answer_qa(question, retrieved_memory, subset, history)` — LLM call, log to `call_6_qa`
  - `on_conv_boundary(conv_id, dialogue_text)` — trigger daily summarization (2 LLM calls)
  - `on_session_end()` — trigger global summary synthesis (2 LLM calls)
  - `set_llm_logger(logger)` — inject logger, propagates to `memory_system`
  - Token tracking: `get_and_reset_summary_tokens()`

**`run_experiment.py`** — Orchestrator
- Phase 1 loop with conv\_id boundary detection for forgetting + summarization triggers
- Response prompt construction and logging per turn (no LLM call)
- Phase 2 QA with direct question-as-query retrieval (timed)
- Per-session `LLMCallLogger` initialization and injection

---

## LLM Call Logging

All LLM calls are logged per session in `prompt_log/session_{id}/`:

```
prompt_log/session_{id}/
├── call_1_response/calls.jsonl          # Response prompt, output=null (Phase 1, per turn)
├── call_2_daily_event/calls.jsonl       # Daily event summary (at conv_id boundary)
├── call_3_daily_personality/calls.jsonl # Daily personality summary (at conv_id boundary)
├── call_4_global_event/calls.jsonl      # Global event summary (every GLOBAL_SUMMARY_INTERVAL + Phase 1 end)
├── call_5_global_personality/calls.jsonl # Global personality summary (same triggers)
└── call_6_qa/calls.jsonl                # QA answering (Phase 2, per question)
```

Each line is a JSON object:

```json
{
  "timestamp":     "2026-03-27T11:23:45.123456",
  "call_type":     "call_2_daily_event",
  "system_prompt": "/no_think",
  "user_prompt":   "(full prompt string)",
  "output":        "(summary text)"
}
```

- `call_1_response`: `output` is always `null` (no LLM call)
- All other calls: `output` is the generated text or structured dict (without `_usage`)
- Logging can be disabled via `ENABLE_LLM_CALL_LOGGING = False` in config

---

## Output Structure

```
memorybank/
└── {config}_outputs_{model}_{subset}/
    ├── session_{start}_{end}/
    │   ├── results_{model}_{subset}_session_{start}_{end}.json
    │   ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
    │   ├── retrieval_logs/
    │   │   └── session_{id}_retrieval_log.jsonl
    │   ├── memory_snapshots/
    │   │   └── session_{id}/
    │   │       ├── entries.json
    │   │       ├── embeddings.npy
    │   │       ├── corpus.pkl
    │   │       ├── summaries.json
    │   │       └── metadata.json
    │   ├── prompt_log/
    │   │   └── session_{id}/
    │   │       ├── call_1_response/calls.jsonl
    │   │       ├── call_2_daily_event/calls.jsonl
    │   │       ├── call_3_daily_personality/calls.jsonl
    │   │       ├── call_4_global_event/calls.jsonl
    │   │       ├── call_5_global_personality/calls.jsonl
    │   │       └── call_6_qa/calls.jsonl
    │   └── logs/
    └── results_{model}_{subset}_merged.json
```

### `results_*.json`

List of session result objects (QA-only schema):

```json
[
  {
    "session_id": 0,

    "qa_results": [
      {
        "question": "...",
        "generated_answer": "...",
        "ground_truth_answer": "...",
        "retrieved_memories": [
          {"session_id": 0, "conv_id": 0, "turn_id": 2, "score": 0.72}
        ],
        "qa_tokens": {"input": 300, "output": 20, "model": "..."}
      }
    ],

    "timing_statistics": {
      "phase2_qa_avg": {"retrieval_time": 0.03, "inference_time": 0.14, "total_time": 0.17}
    },

    "token_statistics": {
      "total_response_input": 9800,
      "total_qa_input": 4000,
      "total_qa_output": 800,
      "total_input": 16000,
      "total_output": 2200,
      "num_qa_api_calls": 5,
      "num_total_api_calls": 17
    },

    "memory_snapshot_path": "memory_snapshots/session_0/"
  }
]
```

**Notes:**
- `total_response_input`: estimated via `len(prompt) // 4` (no actual LLM call)
- `total_input` = `total_response_input + total_qa_input + summarization_input`
- `total_output` = `total_qa_output + summarization_output` (no response output)
- `num_qa_api_calls`: successful QA calls only (failed retries excluded)
- `num_total_api_calls` = QA + summarization calls (response prompt not counted)
- `generated_answer` for `supportive` subset: one of `"yes"`, `"no"` (or `"unknown"` on error fallback). For `opposed`: free-form text.

### `retrieval_logs/session_{id}_retrieval_log.jsonl`

One JSON object per retrieval operation (one per turn in Phase 1, one per QA in Phase 2).
Prompt snapshots are **not** stored here — they are in `prompt_log/` instead.

| Field | Description |
|---|---|
| `phase` | `"prompt_construction"` (Phase 1) or `"qa"` (Phase 2) |
| `retrieved_items` | Top-k memories: content preview, cosine score, source turn, strength |
| `module_specific.current_conv_id` | conv\_id at time of retrieval (forgetting curve reference) |
| `module_specific.total_memories` | Total memory entries in the store at retrieval time |
| `module_specific.event_summary_length` | Character count of current event summary |
| `module_specific.user_portrait_length` | Character count of current user portrait |

---

## Token Tracking

MemoryBank makes internal LLM calls for **summarization only**:

1. **Daily event summary** (`call_2`) — at each conv\_id boundary (every `CONVS_PER_DAY`)
2. **Daily personality summary** (`call_3`) — same trigger
3. **Global event synthesis** (`call_4`) — every `GLOBAL_SUMMARY_INTERVAL` conv\_ids + Phase 1 end
4. **Global personality synthesis** (`call_5`) — same trigger

No LLM calls at memory storage time. No keyword generation for QA retrieval.

```
total_input  = total_response_input (estimated) + total_qa_input + summarization_input
total_output = total_qa_output + summarization_output
num_total_api_calls = num_qa_api_calls + num_summarization_calls
```

---

## Data Flow

### Phase 1: Memory Construction (QA-Only Variant)

```
For each (user_turn, assistant_turn) pair:
  1. Retrieve: top-k cosine similarity (update_strength=True)
  2. Build response prompt → log to call_1_response (output=null)
     → estimate input tokens via len(prompt) // 4
  3. Log retrieval entry (phase="prompt_construction")
  4. Store user utterance in memory (embedding only)
  5. Store GT assistant response in memory (embedding only)

  On conv_id boundary (new conv_id detected):
  6. apply_forgetting(current_conv_id) → probabilistic deletion
  7. summarize_daily → call_2_daily_event + call_3_daily_personality
  8. Every GLOBAL_SUMMARY_INTERVAL conv_ids:
     synthesize_global → call_4_global_event + call_5_global_personality

Phase 1 End:
  9. Flush remaining turns → summarize_daily (final batch)
  10. synthesize_global (unconditional) → call_4 + call_5
```

### Phase 2: QA Answering

```
For each QA question:
  1. Retrieve: top-k cosine similarity (question as query, update_strength=False)
  2. Build QA prompt: event_summary + user_portrait + retrieved_mem + history + question
  3. Generate answer via LLM → log to call_6_qa
     (opposed: free-form; supportive: yes/no, fallback=unknown on error)
  4. Log retrieval entry (phase="qa")
  5. Measure timing: retrieval_time + inference_time
```

### Phase 3: Cleanup

```
  1. Save memory snapshot (entries.json, embeddings.npy, corpus.pkl, summaries.json)
  2. Clear memory for next session (also clears llm_logger reference)
  3. Save results + update checkpoint
```

---

## Citation

```bibtex
@inproceedings{zhong2024memorybank,
  title={MemoryBank: Enhancing Large Language Models with Long-Term Memory},
  author={Zhong, Wanjun and Guo, Lianghong and Gao, Qiqi and Ye, He and Wang, Yanlin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  pages={19724--19731},
  year={2024}
}
```
