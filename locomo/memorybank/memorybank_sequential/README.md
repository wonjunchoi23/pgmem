# MemoryBank — Long-Term Memory with Ebbinghaus Forgetting Curve

MemoryBank (Zhong et al., AAAI 2024) memory module for the LoComo experiment
(**QA-only variant** — response generation LLM calls are removed).
Stores dialogue turns as dense embeddings, retrieves via cosine similarity,
and builds hierarchical summaries (event + two-speaker personality) injected into prompts.

Paper: *MemoryBank: Enhancing Large Language Models with Long-Term Memory*

---

## Overview

This module runs the **QA-only variant** of the experiment: no response
generation LLM call is made.

- **Phase 1 (Memory Construction)**: For each sample, sessions are processed
  in order. Within each session, every turn is stored as a new memory entry
  (embedding only — no LLM call at storage time). After all turns in a session,
  two LLM summaries are generated (event + two-speaker personality). The event
  summary is also added to the FAISS embedding store as a searchable memory
  document (`is_summary=True`).
- **Phase 1 End**: After all sessions are processed, global summaries are
  synthesized from all per-session summaries (2 LLM calls). Then the Ebbinghaus
  forgetting curve is applied using the last session's date as "now".
- **Phase 2 (QA Answering)**: QA pairs are answered using memory retrieval.
  The question text is used directly as the retrieval query (no keyword
  generation). No new memories are written during this phase.
- Memory is **cleared between samples**.
- Turns from both speakers are stored without filtering.
- **QA exchanges are never stored in memory.**

### Speaker Prefix

Turns are stored with the format:

```
[{speaker_name}]: {text}
```

### Image Handling

If a turn contains image fields (`img_url`, `blip_caption`), the caption is
prepended to the turn text as `[Image: caption]` at dataset load time.
`img_url` is discarded.

---

## Key Design: MemoryBank on LoComo

### Forgetting Curve (Ebbinghaus-Inspired)

Each memory entry tracks `strength` (S, starts at 1) and `last_recall_date_str`.
At Phase 1 end (before Phase 2), probabilistic deletion is applied using actual
calendar dates. The last session's date is used as "now" (using the current date
would make all 2023 LoComo memories have near-zero retention).

```
day_gap   = (now_date - entry_date).days
retention = exp( -day_gap / (FORGETTING_DIVISOR * S) )

if random() > retention → permanently delete from FAISS
```

Strength update (`S += 1`) only happens during Phase 1 retrieval
(`update_strength=True`). During Phase 2 QA retrieval, `update_strength=False`.

### Real Calendar Dates

LoComo uses real timestamps like `"1:56 pm on 8 May, 2023"`. These are parsed
into `date` objects for the forgetting curve calculation.

```python
_DATE_PATTERN = re.compile(r'(\d{1,2})\s+(\w+),?\s+(\d{4})')
```

### Hierarchical Summarization

After each session, two LLM summaries are generated:

1. **Session Event Summary** (`call_1_session_event`) — concise digest of the session's dialogue
2. **Session Personality Summary** (`call_2_session_personality`) — personality traits, emotions,
   and communication patterns of **both speakers** (two-speaker adaptation)

The event summary is then added to the FAISS embedding store as a searchable
document (alongside raw turn memories). This makes session summaries retrievable
during Phase 2 QA.

After all sessions (Phase 1 end), these are synthesized into:

3. **Global Event Summary** (`call_3_global_event`) — bird's-eye view of all events across sessions
4. **Global User Portrait** (`call_4_global_personality`) — overall personality understanding

### QA Context

The QA prompt context combines:

```
{retrieved_memories}

[Summary of past conversations]: {global_event_summary}

[Speaker profiles]: {global_personality_portrait}
```

### No LLM at Storage Time

Unlike A-MEM (which extracts keywords/tags/context via LLM per note), MemoryBank
stores memories using only embedding — no LLM call at write time. Internal LLM
calls are limited to session-end summarization (2 calls) and Phase 1 end global
synthesis (2 calls).

---

## Configuration (`config_0.py`)

| Parameter | Default | Description |
|---|---|---|
| `DATASET_PATH` | `dataset/locomo10.json` | Path to the LoComo dataset |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence embedding model for FAISS retrieval |
| `RETRIEVE_K` | `6` | Memories retrieved per query (original paper `VECTOR_SEARCH_TOP_K = 6`) |
| `FORGETTING_DIVISOR` | `5` | Divisor in forgetting curve: `exp(-day_gap / (DIVISOR * S))` |
| `SUMMARIZE_TEMPERATURE` | `0.7` | LLM temperature for summarization calls |
| `SUMMARIZE_MAX_TOKENS` | `400` | Max tokens for summarization outputs |
| `TEMPERATURE` | `0.7` | LLM temperature for QA generation (categories 1–4) |
| `TEMPERATURE_C5` | `0.5` | LLM temperature for adversarial QA (category 5) |
| `MAX_TOKENS` | `750` | Max output tokens per QA generation |
| `JSON_RETRY` | `3` | Retry count on JSON parse failure |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint after each sample |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save memory snapshots after each sample |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Log all LLM prompts/outputs per sample |
| `LLM_ENGINE` | `"vllm"` | Engine: `"vllm"`, `"together"`, or `"openai"` |

---

## QA Prompts (Category-Aware)

| Category | Description | Prompt used |
|---|---|---|
| 1, 3, 4 | Single-hop, open-domain, multi-hop | Default: short phrase, exact words from context |
| 2 | Temporal | Temporal: use DATE OF CONVERSATION for approximate date |
| 5 | Adversarial | Binary choice: `adversarial_answer` vs `"Not mentioned in the conversation"` (randomised order) |

Category 5 uses `TEMPERATURE_C5`; all others use `TEMPERATURE`.

---

## Usage

```bash
python run_experiment.py \
    --start-sample 0 --end-sample 4 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --config config_0        # optional; defaults to "config_0"

# Merge results from multiple runs
python merge_results.py config_0_outputs_Llama-3.1-8B-Instruct
python merge_results.py config_0_outputs_Llama-3.1-8B-Instruct --dry-run
```

`--config` specifies the config file name (without `.py`) in the `memorybank/`
directory. The config name is also used as the output directory prefix:
`{config}_outputs_{model}/`.

---

## Module Structure

```
memorybank/
├── config_0.py            # Configuration & path helpers
├── load_dataset.py        # LoComo dataset parser (shared schema)
├── retriever.py           # EmbeddingRetriever — sentence-transformers cosine similarity
├── memory_bank.py         # MemoryBankSystem — forgetting curve + hierarchical summarizer
├── agent.py               # MemoryBankAgent + LLMCallLogger — experiment-facing wrapper
├── run_experiment.py      # Experiment orchestrator (Phase 1/2 loop, QA-only)
├── merge_results.py       # Result aggregation across sample ranges
└── README.md
```

### Module Responsibilities

**`retriever.py`** — Embedding retrieval
- `EmbeddingRetriever`: sentence-transformers based cosine similarity search
  - `add_document(text)` — embed and append to corpus
  - `search_with_scores(query, k)` — return top-k (index, score) pairs
  - `remove_by_indices(indices)` — delete entries (used by forgetting curve)
  - `save(directory)` / `load(directory)` — persist embeddings and corpus

**`memory_bank.py`** — Core MemoryBank system
- `MemoryEntry`: dataclass with `id`, `content`, `dia_id`, `session_id`, `date_str`, `strength`, `last_recall_date_str`, `is_summary`
- `MemoryBankSystem`: vector store with forgetting curve and summarization
  - `add_memory(content, dia_id, session_id, date_str, is_summary=False)` — embed and store
  - `retrieve(query, k, update_strength)` — cosine top-k → optional strength update
  - `apply_forgetting(now_date_str)` — probabilistic deletion using actual calendar dates
  - `summarize_session(session_id, date_str, dialogue_text, speaker_a, speaker_b)` — 2 LLM calls; adds event summary to FAISS
  - `synthesize_global()` — 2 LLM calls (`call_3_global_event`, `call_4_global_personality`)
  - `get_event_summary()` / `get_user_portrait()` — returns global summaries
  - `set_llm_logger(logger)` — inject per-sample `LLMCallLogger`
  - `clear()` / `save_snapshot()` / `get_memory_count()`

**`agent.py`** — Experiment interface
- `LLMCallLogger`: per-sample JSONL logger for all LLM calls (5 call types)
- `MemoryBankAgent`: wraps `MemoryBankSystem` + LLM client
  - `add_memory(content, dia_id, session_id, date_str)` — store memory entry
  - `retrieve_memory(query, k, update_strength)` — returns `RetrievalResult`
  - `on_session_end(session_id, date_str, dialogue_text, speaker_a, speaker_b)` — trigger session summarization
  - `on_phase1_end()` — trigger global summary synthesis
  - `apply_forgetting(now_date_str)` — apply forgetting curve before Phase 2
  - `answer_qa(question, retrieved_memory, category, adversarial_answer)` — category-aware QA with 1 LLM call
  - `set_llm_logger(logger)` — inject logger, propagates to `memory_system`
  - Token tracking: `get_and_reset_summary_tokens()`

**`run_experiment.py`** — Orchestrator
- Phase 1 loop: per-turn storage → per-session summarization → global synthesis + forgetting
- Phase 2 QA with direct question-as-query retrieval (timed, category-aware)
- Per-sample `LLMCallLogger` initialization and injection

---

## LLM Call Logging

All LLM calls are logged per sample in `prompt_log/sample_{id}/`:

```
prompt_log/sample_{id}/
├── call_1_session_event/calls.jsonl       # Session event summary (Phase 1, per session)
├── call_2_session_personality/calls.jsonl # Session personality summary (Phase 1, per session)
├── call_3_global_event/calls.jsonl        # Global event summary (Phase 1 end)
├── call_4_global_personality/calls.jsonl  # Global personality summary (Phase 1 end)
└── call_5_qa/calls.jsonl                  # QA answering (Phase 2, per question)
```

Each line is a JSON object:

```json
{
  "timestamp":     "2026-03-27T11:23:45.123456",
  "call_type":     "call_1_session_event",
  "system_prompt": "",
  "user_prompt":   "(full prompt string)",
  "output":        "(summary text)"
}
```

Logging can be disabled via `ENABLE_LLM_CALL_LOGGING = False` in config.

---

## Output Structure

```
memorybank/
└── {config}_outputs_{model}/
    ├── sample_{start}_{end}/
    │   ├── results_{model}_sample_{start}_{end}.json
    │   ├── checkpoint_{model}_sample_{start}_{end}.json
    │   ├── retrieval_logs/
    │   │   └── sample_{id}_retrieval_log.jsonl
    │   ├── memory_snapshots/
    │   │   └── sample_{id}/
    │   │       ├── entries.json
    │   │       ├── embeddings.npy
    │   │       ├── corpus.pkl
    │   │       ├── summaries.json
    │   │       └── metadata.json
    │   ├── prompt_log/
    │   │   └── sample_{id}/
    │   │       ├── call_1_session_event/calls.jsonl
    │   │       ├── call_2_session_personality/calls.jsonl
    │   │       ├── call_3_global_event/calls.jsonl
    │   │       ├── call_4_global_personality/calls.jsonl
    │   │       └── call_5_qa/calls.jsonl
    │   └── logs/
    └── sample_{min}_{max}/         ← created by merge_results.py
        ├── results_{model}_sample_{min}_{max}.json
        └── retrieval_logs/
```

### `results_*.json`

List of sample result objects:

```json
[
  {
    "sample_id": "0",

    "qa_results": [
      {
        "question": "...",
        "category": 1,
        "generated_answer": "...",
        "ground_truth_answer": "...",
        "evidence": ["D1:3", "D2:7"],
        "retrieved_memories": [
          {"dia_id": "D1:3", "score": 0.72}
        ],
        "qa_tokens": {"input": 300, "output": 20, "model": "..."}
      }
    ],

    "timing_statistics": {
      "phase2_qa_avg": {"retrieval_time": 0.03, "inference_time": 0.14, "total_time": 0.17}
    },

    "token_statistics": {
      "total_qa_input": 4000,
      "total_qa_output": 800,
      "total_internal_input": 3200,
      "total_internal_output": 1600,
      "total_input": 7200,
      "total_output": 2400,
      "num_qa_api_calls": 5,
      "num_internal_api_calls": 10,
      "num_total_api_calls": 15
    },

    "memory_snapshot_path": "memory_snapshots/sample_0/"
  }
]
```

**Notes:**
- `sample_id` is a string (matches the `sample_id` field in the LoComo dataset).
- `ground_truth_answer` is `adversarial_answer` for category 5, `answer` otherwise.
- `evidence` is the list of `dia_id`s from the dataset QA annotation.
- `retrieved_memories` contains items with `dia_id` (non-null only; session summary
  entries with `dia_id=None` are excluded from this field).
- `total_internal_input/output`: tokens from MemoryBank summarization calls (sessions + global).
- `num_qa_api_calls`: successful QA generation calls only (failed retries excluded).
- `num_internal_api_calls`: 2 per session (event + personality) + 2 at Phase 1 end (global).
- `memory_snapshot_path` is relative to the sample directory;
  `null` when `SAVE_MEMORY_SNAPSHOTS = False`.

### `retrieval_logs/sample_{id}_retrieval_log.jsonl`

One JSON object per line, one entry per QA question (Phase 2 only).

| Field | Description |
|---|---|
| `phase` | `"qa"` for all entries |
| `query` | The QA question text used as retrieval query |
| `memory_type` | Parallel labels for `num_retrieved` (`["dialogue_memory", "session_summary"]`) |
| `retrieved_items` | Top-k memories: content preview, cosine score, `dia_id`, `memory_type` |
| `num_retrieved` | Per-type counts of retrieved dialogue memories and retrieved session summaries |
| `module_specific.total_memories` | Total memory entries in store at retrieval time |
| `module_specific.event_summary_length` | Character count of global event summary |
| `module_specific.user_portrait_length` | Character count of global personality portrait |
| `module_specific.prompt_context_type` | Full QA-context labels including always-appended global summary / portrait |
| `module_specific.num_prompt_context` | Parallel counts for the final QA context composition |

---

## Token Tracking

MemoryBank makes internal LLM calls for **summarization only** — no keyword
generation for retrieval, no LLM at storage time.

Internal calls per sample:
1. **Session event summary** (`call_1`, once per session)
2. **Session personality summary** (`call_2`, once per session)
3. **Global event synthesis** (`call_3`, once at Phase 1 end)
4. **Global personality synthesis** (`call_4`, once at Phase 1 end)

For a sample with N sessions: `num_internal_api_calls = 2*N + 2`.

```
total_input  = total_qa_input + total_internal_input
total_output = total_qa_output + total_internal_output
num_total_api_calls = num_qa_api_calls + num_internal_api_calls
```

---

## Data Flow

### Phase 1: Memory Construction

```
For each session:
  For each turn:
    1. Store turn in memory (embedding only, no LLM)
       Format: "[{speaker}]: {text}"
       Metadata: dia_id, session_id, date_str (session date), is_summary=False

  After all turns in session:
    2. Format all session turns as dialogue text
    3. summarize_session() → call_1_session_event + call_2_session_personality
       - event summary added to FAISS (is_summary=True, dia_id=None)
    4. Collect summarization token counts

Phase 1 End:
  5. synthesize_global() → call_3_global_event + call_4_global_personality
  6. apply_forgetting(last_session_date) → probabilistic deletion from FAISS
```

### Phase 2: QA Answering

```
For each QA question:
  1. Retrieve: top-k cosine similarity (question as query, update_strength=False)
  2. Build context: retrieved_memories + global_event_summary + global_personality
  3. Select category-aware prompt (cat 1/3/4: default; cat 2: temporal; cat 5: binary choice)
  4. Generate answer via LLM → log to call_5_qa
  5. Log retrieval entry
  6. Measure timing: retrieval_time + inference_time
```

### Phase 3: Cleanup

```
  1. Save memory snapshot (entries.json, embeddings.npy, corpus.pkl, summaries.json, metadata.json)
  2. Clear memory for next sample (also clears llm_logger reference)
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
