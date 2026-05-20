# MemoryBank — Long-Term Memory with Ebbinghaus Forgetting Curve (Batched Main Variant)

MemoryBank (Zhong et al., AAAI 2024) memory module for the ImplexConv experiment
(**batched main variant** — no response-generation LLM calls; memory storage is
embedding-only, while boundary summarization and Phase 2 QA still use the LLM).
Stores dialogue turns as dense embeddings, retrieves via cosine similarity,
and builds hierarchical summaries (event + personality) that are injected into prompts.

`memorybank/` is the current batched main implementation.
The archived sequential reference is kept under `memorybank/memorybank_sequential/`.

Paper: *MemoryBank: Enhancing Large Language Models with Long-Term Memory*

---

## Overview

- **Phase 1 (Memory Construction)**: For each (user\_turn, assistant\_turn) pair in
  a session, MemoryBank retrieves relevant memories (top-k cosine similarity), logs
  the retrieval, then stores both the user utterance and the **GT assistant response**
  as new memory entries (embedding only — no LLM call at storage time).
  At each **conv\_id boundary** (= "new day"), daily event and personality summaries
  are generated. Every `GLOBAL_SUMMARY_INTERVAL` conv\_ids, an intermediate global
  synthesis is also run.
- **Phase 2 (QA Answering)**: After all turns are processed, QA pairs are answered
  using memory retrieval. The question text is used directly as the retrieval query
  (no keyword generation), following the original MemoryBank approach. No new memories
  are written during this phase. QA prompts are executed in `QA_BATCH_SIZE` chunks.
- **Phase 1 End / Phase 2 Transition**: Global summaries are always synthesized
  unconditionally from all accumulated daily summaries before QA begins. Memory state
  and Phase 1 internal statistics are recorded at this boundary.
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

At each virtual day boundary (`CONVS_PER_DAY` conv\_ids batched together), two
LLM summaries are generated:

1. **Daily Event Summary** (`call_2_daily_event`) — concise digest of the day's conversations
2. **Daily Personality Summary** (`call_3_daily_personality`) — user traits and response strategies

Every `GLOBAL_SUMMARY_INTERVAL` conv\_ids (and unconditionally at Phase 1 end), these
are synthesized into:

3. **Global Event Summary** (`call_4_global_event`) — bird's-eye view of all events
4. **Global User Portrait** (`call_5_global_personality`) — overall personality understanding

### QA Prompt Structure

```
[User Portrait]   Global user portrait
[Memory]          Top-k QA retrieval hits (dialogue snippets + daily event summaries)
[Memory Dates]    Source virtual-day labels for the retrieved memory block
[Example]         Original MemoryBank eval-style one-shot example
[History]         Recent conversation turns (last HISTORY_CONV_WINDOW conv_ids)
[User]            QA question
[AI]              LLM generates answer
```

Additional prompt-level answer constraints:

- `opposed`: concise English answer, maximum 100 words
- `supportive`: answer with exactly `yes` or `no`

### No LLM at Storage Time

Unlike A-MEM (which extracts keywords/tags/context via LLM per note), MemoryBank
stores memories using only embedding — no LLM call at write time. Internal LLM
calls are limited to summarization (at conv\_id boundaries and session end) and QA answering.

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
| `MAX_TOKENS` | `750` | Max output tokens per QA generation (in addition to subset-specific prompt instructions) |
| `JSON_RETRY` | `3` | Retry count on JSON parse failure |
| `BATCH_SIZE` | `4` | Number of sessions processed together |
| `QA_BATCH_SIZE` | `64` | QA prompts per batched generation call |
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
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --batch-size 4 \
    --config config_0

# Merge results from multiple runs
python merge_results.py memorybank config_0_outputs_Qwen3-1.7B_opposed
```

`--config` specifies the config file name (without `.py`) in the `memorybank/`
directory. The config name is used as the output directory prefix:
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
├── run_experiment.py      # Batched multi-session experiment orchestrator
├── README.md
└── nohup/
```

### Module Responsibilities

**`retriever.py`** — Embedding retrieval
- `EmbeddingRetriever`: sentence-transformers based cosine similarity search
  - `add_document(text)` — embed and append to corpus
  - `search_with_scores(query, k)` — return top-k (index, score) pairs
  - `remove_by_indices(indices)` — delete entries (used by forgetting curve)
  - `save(directory)` / `load(directory)` — persist `embeddings.npy` and `corpus.json`

**`memory_bank.py`** — Core MemoryBank system
- `MemoryEntry`: dataclass with `content`, `timestamp`, `conv_id`, `strength`, `last_recall_conv_id`
- `MemoryBankSystem`: vector store with forgetting curve and summarization
  - `add_memory(content, conv_id, timestamp)` — embed and store (no LLM call)
  - `retrieve(query, k, current_conv_id, update_strength)` — cosine top-k → optional strength update
  - `apply_forgetting(current_conv_id)` — probabilistic deletion at conv\_id boundaries; increments `_forgetting_events` and `_memories_forgotten`
  - `summarize_daily(conv_id, dialogue_text)` — 2 LLM calls (`call_2_daily_event`, `call_3_daily_personality`)
  - `synthesize_global()` — 2 LLM calls (`call_4_global_event`, `call_5_global_personality`)
  - `get_event_summary()` / `get_user_portrait()` — global > daily concatenation fallback
  - `get_memory_stats()` — returns `{num_memories, total_content_tokens}`
  - `get_and_reset_token_counts_by_type()` — returns per-call-type token dict; resets counters
  - `get_and_reset_internal_stats()` — returns forgetting stats; resets counters
  - `accumulate_call_tokens(call_type, input, output)` — accumulate tokens into the correct per-call-type bucket
  - `set_llm_logger(logger)` — inject per-session `LLMCallLogger`
  - `clear()` / `save_snapshot()` / `load_snapshot()`

**`agent.py`** — Experiment interface
- `LLMCallLogger`: per-session JSONL logger for all LLM calls (5 call types)
- `MemoryBankAgent`: wraps `MemoryBankSystem` + LLM client
  - `add_memory(content, conv_id, timestamp)` — store memory entry
  - `retrieve_memory(query, current_conv_id, k, update_strength)` — returns `RetrievalResult`
  - `build_qa_prompt(question, retrieved_memory, subset, history)` — construct QA prompt string
  - `answer_qa(question, retrieved_memory, subset, history)` — LLM call, log to `call_6_qa`
  - `on_conv_boundary(conv_id, dialogue_text)` — trigger daily summarization (2 LLM calls)
  - `on_session_end()` — trigger global summary synthesis (2 LLM calls)
  - `set_llm_logger(logger)` — inject logger, propagates to `memory_system`
  - `get_memory_stats()` — delegates to `memory_system`
  - `get_and_reset_token_counts_by_type()` — delegates to `memory_system`
  - `get_and_reset_internal_stats()` — delegates to `memory_system`
  - `accumulate_call_tokens(call_type, input, output)` — delegates to `memory_system`

**`run_experiment.py`** — Orchestrator
- Phase 1 loop with conv\_id boundary detection for forgetting + summarization triggers
- Batched summarization (daily and global) across sessions
- Phase 1 end: records `memory_at_qa_start` and `phase1_statistics` before Phase 2
- Phase 2 QA with direct question-as-query retrieval
- Per-session `LLMCallLogger` initialization and injection

---

## LLM Call Logging

All LLM calls are logged per session in `prompt_log/session_{id}/`:

```
prompt_log/session_{id}/
├── call_2_daily_event/calls.jsonl        # Daily event summary (at conv_id boundary)
├── call_3_daily_personality/calls.jsonl  # Daily personality summary (at conv_id boundary)
├── call_4_global_event/calls.jsonl       # Global event summary (every GLOBAL_SUMMARY_INTERVAL + Phase 1 end)
├── call_5_global_personality/calls.jsonl # Global personality summary (same triggers)
└── call_6_qa/calls.jsonl                 # QA answering (Phase 2, per question)
```

Each line is a JSON object:

```json
{
  "timestamp":     "2026-03-27T11:23:45.123456",
  "call_type":     "call_2_daily_event",
  "system_prompt": null,
  "user_prompt":   "(full prompt string)",
  "output":        "(summary text)"
}
```

- All calls: `output` is the generated text or structured dict (without `_usage`)
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
    │   │       ├── corpus.json
    │   │       ├── summaries.json
    │   │       └── metadata.json
    │   ├── prompt_log/
    │   │   └── session_{id}/
    │   │       ├── call_2_daily_event/calls.jsonl
    │   │       ├── call_3_daily_personality/calls.jsonl
    │   │       ├── call_4_global_event/calls.jsonl
    │   │       ├── call_5_global_personality/calls.jsonl
    │   │       └── call_6_qa/calls.jsonl
    │   └── logs/
    └── results_{model}_{subset}_merged.json
```

### `results_*.json`

List of session result objects:

```json
[
  {
    "session_id": 0,

    "config_metadata": {
      "config_name":             "config_0",
      "model":                   "Qwen/Qwen3-1.7B",
      "subset":                  "opposed",
      "embedding_model":         "all-MiniLM-L6-v2",
      "temperature":             0.7,
      "max_tokens":              750,
      "session_range":           [0, 99],
      "retrieve_k":              6,
      "forgetting_divisor":      5,
      "convs_per_day":           2,
      "global_summary_interval": 10,
      "history_conv_window":     2,
      "summarize_temperature":   0.7,
      "summarize_max_tokens":    400
    },

    "memory_at_qa_start": {
      "num_memories":        356,
      "total_content_tokens": 18400
    },

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

    "token_statistics": {
      "call_2_daily_event":        {"input": 12000, "output": 1800, "llm_calls": 30},
      "call_3_daily_personality":  {"input": 12000, "output": 2400, "llm_calls": 30},
      "call_4_global_event":       {"input": 3000,  "output": 400,  "llm_calls": 4},
      "call_5_global_personality": {"input": 3000,  "output": 600,  "llm_calls": 4},
      "call_6_qa":                 {"input": 9900,  "output": 320,  "llm_calls": 5},
      "total_input":     39900,
      "total_output":    5520,
      "total_llm_calls": 73
    },

    "phase1_statistics": {
      "forgetting_events_count": 60,
      "memories_forgotten": 142
    },

    "memory_snapshot_path": "memory_snapshots/session_0/"
  }
]
```

**Notes:**
- `memory_at_qa_start.total_content_tokens`: estimated via `len(content) // 4`
- `token_statistics.total_input` = sum of input across all 5 call types
- `token_statistics.total_output` = sum of output across all 5 call types
- `token_statistics.total_llm_calls` = sum of llm\_calls across all 5 call types
- `generated_answer` for `supportive` subset: one of `"yes"`, `"no"` (or `"unknown"` on error fallback). For `opposed`: free-form text.

### `retrieval_logs/session_{id}_retrieval_log.jsonl`

One JSON object per QA retrieval operation.

| Field | Description |
|---|---|
| `phase` | Always `"qa"` |
| `retrieved_items` | Top-k memories with subtype (`dialogue_snippet` / `daily_summary`) |
| `module_specific.current_conv_id` | conv\_id at time of retrieval (forgetting curve reference) |
| `module_specific.total_memories` | Total memory entries in the store at retrieval time |
| `module_specific.total_daily_summaries` | Number of daily event summary retrieval docs available |
| `module_specific.user_portrait_length` | Character count of current user portrait |

---

## Token Tracking

MemoryBank makes internal LLM calls for **summarization only** (Phase 1) and **QA answering** (Phase 2).
No LLM calls at memory storage time. No keyword generation for QA retrieval.

Token counts are tracked separately per call type:

| Call type | When | LLM calls per session (approx.) |
|---|---|---|
| `call_2_daily_event` | Each virtual day boundary + final flush | `num_conv_ids / CONVS_PER_DAY` |
| `call_3_daily_personality` | Each virtual day boundary + final flush | same |
| `call_4_global_event` | Every `GLOBAL_SUMMARY_INTERVAL` + Phase 1 end | small |
| `call_5_global_personality` | Every `GLOBAL_SUMMARY_INTERVAL` + Phase 1 end | small |
| `call_6_qa` | Per QA question | `len(session.qa)` |

```
total_input     = sum of input  across call_2 … call_6
total_output    = sum of output across call_2 … call_6
total_llm_calls = sum of llm_calls across call_2 … call_6
```

---

## Data Flow

### Phase 1: Memory Construction

```
For each (user_turn, assistant_turn) pair:
  1. Store the dialogue snippet for later QA retrieval (embedding only)

  On conv_id boundary (new conv_id detected):
  2. apply_forgetting(current_conv_id) → probabilistic deletion
     (increments forgetting_events_count + memories_forgotten)
  3. If the new conv_id closes a virtual day batch:
     summarize_daily -> call_2_daily_event + call_3_daily_personality
  4. Every GLOBAL_SUMMARY_INTERVAL conv_ids:
     synthesize_global → call_4_global_event + call_5_global_personality

Phase 1 End:
  5. Flush remaining turns → summarize_daily (final batch)
  6. synthesize_global (unconditional) → call_4 + call_5
  7. Record memory_at_qa_start (num_memories, total_content_tokens)
  8. Record phase1_statistics (forgetting_events_count, memories_forgotten)
```

### Phase 2: QA Answering

```
For each QA question:
  1. Retrieve: top-k cosine similarity over dialogue snippets + daily event summaries
     (question as query, update_strength=False)
  2. Build QA prompt: user_portrait + retrieved_mem + memo_dates + example + history + question
  3. Generate answer via LLM → log to call_6_qa
     (opposed: concise English answer, maximum 100 words;
      supportive: exactly yes/no, fallback=unknown on error)
  4. Log retrieval entry (phase="qa")
```

### Phase 3: Cleanup

```
  1. Save memory snapshot (entries.json, embeddings.npy, corpus.json, summaries.json)
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
