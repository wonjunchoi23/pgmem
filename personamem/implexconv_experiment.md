# Experiment Specification: Memory-augmented LLM Agents on ImplexConv (QA-Only Variant)

This document defines the **QA-only experiment protocol** for evaluating memory-augmented LLM agents on the ImplexConv dataset.

> **Response generation LLM calls are removed.** Phase 1 retrieves memories and stores new memories, but does NOT construct or log a response prompt, and does NOT call the LLM to generate a response. Memory is constructed using GT agent responses. Only QA (Task 2) is evaluated.

This is possible because GT agent responses — not generated responses — are always used for memory construction and dialogue history. The generated response was only needed for Task 1 evaluation, which this variant skips.

---

## 1. Task Formulation

### Dataset

- **ImplexConv** dataset
- Two subsets: `opposed` and `supportive`
- Dataset files:
  - `dataset/implexconv/ImplexConv_opposed_processed.json`
  - `dataset/implexconv/ImplexConv_supportive_processed.json`
- Subset is selected via `--subset` argument in `run_experiment.py`, which routes to the corresponding file through `load_dataset`.

### Dataset Schema

Each JSON file is a list of session objects:

```json
[
  {
    "metadata": {
      "session_id": int,
      "total_conversations": int,
      "total_turns": int
    },
    "conversations": [
      {
        "session_id": int,
        "conv_id": int,
        "turn_id": int,
        "global_turn_id": int,
        "speaker": str,        // "user" or "assistant"
        "utterance": str
      },
      ...
    ],
    "qa": [
      {
        "question": str,
        "answer": str,
        "opposed_implicit_reasoning": str,
        "retrieved_conv_ids": [str, ...]
      },
      ...
    ]
  },
  ...
]
```

- `conversations`: Flattened list of all utterances in the session, ordered by `global_turn_id`.
- `qa`: List of persona-relevant QA pairs associated with the session.
  - `answer`: Ground truth answer.
  - `opposed_implicit_reasoning`: Reasoning chain for the opposed subset.
  - `retrieved_conv_ids`: Conversation IDs from which the answer can be derived.

### Task: Question Answering via Implicit Persona Reasoning

Answer persona-relevant questions using a memory-augmented LLM agent, where persona information is implicitly captured in memory. **QA exchanges (question and generated answer) are not included in dialogue history and are not stored in memory.**

- **Opposed subset**: Free-form answer generation. Evaluation via accuracy, F1 score.
- **Supportive subset**: The agent answers each question with one of `{yes, no}`. Evaluation via accuracy against ground truth labels.

---

## 2. Experiment Flow

Unit of processing: **Individual session S_i**

For each session S_i where i = start_session, ..., end_session:

### Phase 1: Memory Construction on S_i

Process **all turns** in session S_i. No LLM call is made for response generation.

```text
for each (user_turn, assistant_turn) in S_i:
    1. Retrieve relevant information from memory module

    2. Store turn information in memory module
       - Use GT agent response for storage (GT Agent Response Rule applies)
       - QA exchanges are excluded

    3. Update any module-specific components (e.g., summaries, graphs, evolution)

    4. Log retrieval details (see Section 6: Retrieval Logging)

After all turns in S_i:
    5. Finalize memory (e.g., flush pending summaries, run global synthesis)
    6. Record memory state at QA start (memory_at_qa_start)
    7. Save final memory snapshot for session S_i
```

### Phase 2: QA Answering on S_i

QA is performed **after** all turns have been processed and memory is fully constructed. **No new memory is constructed during this phase.**

```text
for each QA in session S_i:
    8. Retrieve relevant memories for the question
    9. Generate answer (return_usage=True)
    10. Log retrieval details (see Section 6: Retrieval Logging)
    11. Record: QA results with retrieved_memories metadata and qa_tokens
        - For supportive subset: record answer as one of {yes, no}
```

> **QA question framing for memory modules:** For modules with memory, the QA question should be used as a retrieval query to search memory (same as a user utterance in Phase 1). However, the supportive subset questions are written in **3rd-person perspective** (e.g., "Does the user like soccer?"), not as natural conversational turns. As a result:
> - **Retrieval**: Use the question string directly as the retrieval query.
> - **LLM prompt**: The prompt must explicitly frame the task as **QA** (not response generation). Do not present the question to the LLM as if it were a user utterance in a conversation. The LLM should be told it is answering a question about the user based on memory/context.

### Phase 3: Cleanup

```text
12. Aggregate statistics:
    - token_statistics: per-call-type input/output/llm_calls, plus totals
    - memory_at_qa_start: memory state captured at end of Phase 1
    - module-specific Phase 1 internal statistics (optional)
13. Save results (atomic write)
14. Clear ALL memory module state for next session
15. Update checkpoint
```

---

## 3. Implementation Constraints

### GT Agent Response Rule

Wherever agent responses are re-used as input to the system, **GT agent responses must be used**:

1. **Memory storage**
   - If the module stores or processes agent responses (e.g., as turn context, summarization source, or paired input), pass GT agent response
   - Module-specific: modules that store only user utterances are unaffected

2. **QA exchanges are excluded from all of the above**
   - QA question/answer pairs must NOT appear in dialogue history
   - QA exchanges must NOT be stored in memory

> **Note:** "GT agent response" refers to the ground truth assistant response from the dataset, not a generated response. No response is generated in this variant.

### Time Model (All Modules)

For all modules in this experiment, virtual time is modeled as:
- **CONV_IDS_PER_DAY = 2**: Every 2 consecutive conv_ids constitute one virtual day
- **MINUTES_PER_TURN = 10**: Each local turn_id within a conv represents 10 minutes
  (Note: local turn_id is per-conv, not global turn)

**How this affects your module:**
- If your module maintains dialogue context or accumulated state: you should reset
  it at each day boundary (when `conv_id // CONV_IDS_PER_DAY` changes)
- If your module needs temporal decay or time-based weighting: you should use this
  model for consistency across modules
- These parameters are configurable in each module's `config.py` so implementations
  can adjust the virtual time scale

### Checkpoint & Resumption

- Checkpoint saved after each session completion (atomic write)
- On restart: loads checkpoint and resumes from the next unprocessed session
  - Legacy format: `{"last_completed_session_index": <int>}`
  - Current format (amem and newer modules): `{"completed_session_ids": [<int>, ...]}`
  - Both formats must be supported for backward compatibility
- Results file also uses atomic write to prevent corruption

### Token Tracking

All LLM calls across the entire pipeline must track token usage via `return_usage=True`. This includes QA generation and **all module-internal LLM calls** (e.g., memory note construction, keyword generation, event summarization).

Token counts are tracked **separately per call type** (not as a flat total). Each call type records:

```json
{
  "input":     <int>,
  "output":    <int>,
  "llm_calls": <int>
}
```

Aggregation in `token_statistics`:

```
total_input     = sum of input     across all call types
total_output    = sum of output    across all call types
total_llm_calls = sum of llm_calls across all call types
```

Use the key name **`llm_calls`** (not `api_calls`) everywhere.

### Configuration Parameters

```text
# Experiment protocol (DO NOT change across modules)
CHECKPOINT_INTERVAL = 1         # Save every session
TEMPERATURE = 0.7               # LLM generation temperature
MAX_TOKENS = 750                # LLM max output tokens (for QA and module-internal calls)
JSON_RETRY = ...                 # Structured output retry count (module-specific; e.g. amem=5, ldagent=3)

# Batch settings (module-specific; used where batched generation is implemented)
BATCH_SIZE = ...                # Sessions processed in parallel within one GPU
QA_BATCH_SIZE = ...             # QA prompts per batched generate call

# Module-specific (may vary)
EMBEDDING_MODEL = "..."         # Embedding model for memory
RETRIEVE_K = ...                # Number of memories to retrieve for QA
SAVE_MEMORY_SNAPSHOTS = True    # Memory snapshot saved after each session

# Virtual time model (all modules should follow — see "Time Model" section above)
CONV_IDS_PER_DAY = 2           # Number of consecutive conv_ids that constitute one virtual day
MINUTES_PER_TURN = 10          # Minutes elapsed per local turn_id within a conv (local, not global)
```

### Validation Checklist

**Invariants:**
1. GT Agent Response Rule & QA exclusion enforced (see above)
2. No LLM call made for response generation during Phase 1
3. Memory completely cleared between sessions
4. Memory snapshot saved at end of each session (before cleanup)
5. Retrieval logs recorded for QA retrieval operations (Phase 2); Phase 1 retrieval logging is module-specific (some modules omit it)
6. `total_input = sum(call_type["input"] for all call types)`
7. `total_output = sum(call_type["output"] for all call types)`
8. `total_llm_calls = sum(call_type["llm_calls"] for all call types)`

**Common Pitfalls:**
- Accidentally calling the LLM for response generation
- Extracting tokens from `llm_client.last_usage` instead of `return_usage=True` per call (unreliable)
- Forgetting to count tokens from module-internal LLM calls (memory construction, keyword generation, etc.)
- Constructing new memories during Phase 2 (QA phase is retrieval-only)
- Using `api_calls` key instead of `llm_calls` in token tracking

---

## 4. Memory Module Interface

Any memory module plugged into this experiment must implement the following behaviors. The internal architecture is free (vector store, graph, hybrid, etc.), but the experiment runner expects these capabilities:

### Required Capabilities

| Capability | Description | Used In |
|---|---|---|
| **Retrieve** | Given a query string, return relevant memories | Phase 1 memory update, Phase 2 QA |
| **Store** | Store a turn into memory (GT Agent Response Rule applies — see Section 3) | Phase 1 after each turn |
| **Finalize** | Flush pending state, run any end-of-phase synthesis (e.g., global summaries) | Phase 1 end |
| **get_memory_stats** | Return `{num_memories, total_content_tokens}` for the current memory state | Phase 1 end (before Phase 2) |
| **Clear** | Reset all memory state completely | Phase 3 between sessions |
| **Retrieve with metadata** | Return memories with (session_id, conv_id, turn_id) for QA tracking | Phase 2 QA |

### LLM Client Infrastructure

All modules share a unified LLM client located at `{project_root}/llm_module/llm_client.py`. Each module's `config.py` adds this to the Python path:

```python
LLM_MODULE_DIR = PROJECT_ROOT / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))
```

**Client creation** (in each module's `run_experiment.py`):

```python
from llm_client import create_llm_client as create_client

# vLLM (local GPU)
client = create_client(
    engine="vllm",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5
)

# Together AI / OpenAI (API)
client = create_client(engine="together", model_name="...", api_key="...")
client = create_client(engine="openai", model_name="...", api_key="...")
```

**Calling convention** — all LLM calls use the same interface:

```python
result = client.generate(
    prompt="...",
    system_prompt="...",
    guided_json=SOME_SCHEMA,      # optional, for JSON structured output
    temperature=0.7,
    max_tokens=150,
    json_retry=3,                  # optional, retry on JSON parse failure
    return_usage=True              # REQUIRED for token tracking
)
```

- When `return_usage=True`, the returned dict includes a `_usage` field with `prompt_tokens` and `completion_tokens`.
- When `guided_json` is provided, the response is parsed into a dict matching the schema.
- The same client instance is shared across all components within a module (memory, persona, generator, etc.).

**Thinking mode** — for models that support a reasoning/thinking mode (e.g. Qwen3),
thinking is **disabled by default** via `enable_thinking=False` passed to the chat
template. This prevents the model from generating `<think>` tokens, which consume
output budget without contributing to the structured response. To enable thinking
for a specific run, pass `enable_thinking=True` to `create_llm_client()`:

```python
# Thinking disabled (default)
client = create_client(engine="vllm", model_path="Qwen/Qwen3-1.7B", ...)

# Thinking enabled
client = create_client(engine="vllm", model_path="Qwen/Qwen3-1.7B", ..., enable_thinking=True)
```

For models that do not support thinking mode the parameter is silently ignored.

### Project Directory Structure

```text
{project_root}/
├── llm_module/
│   └── llm_client.py            # Shared unified LLM client
├── dataset/
│   └── implexconv/
│       ├── ImplexConv_opposed_processed.json
│       └── ImplexConv_supportive_processed.json
├── {module_name}/
│   ├── config.py
│   ├── load_dataset.py
│   ├── run_experiment.py
│   ├── merge_results.py
│   ├── {module_specific_files}
│   └── outputs_*/
└── evaluation/
```

---

## 5. LLM Call Logging

All LLM calls are logged per session to enable full traceability of prompts and outputs.

**Location:** `{session_dir}/prompt_log/session_{id}/`

**Structure:** One subdirectory per call type, each containing a `calls.jsonl` file.
Call type names and count are **module-specific** — see each module's README for the full list.

```text
prompt_log/session_{id}/
├── {call_type_phase1_A}/calls.jsonl    # Module-specific Phase 1 memory call(s)
├── {call_type_phase1_B}/calls.jsonl    # ...
└── {call_type_qa}/calls.jsonl          # QA answering (Phase 2)
```

Examples by module:

| Module | Phase 1 call types | Phase 2 call type |
|---|---|---|
| **amem** | `call_2_note_construction`, `call_3_evolution` | `call_4_qa` |
| **ldagent** | `call_2_user_persona`, `call_3_agent_persona`, `call_4_summarization` | `call_5_qa` |
| **memorybank** | `call_2_daily_event`, `call_3_daily_personality`, `call_4_global_event`, `call_5_global_personality` | `call_6_qa` |
| **theanine** | `call_3_summarization`, `call_4_relation` | `call_5_qa` |

**Entry schema** (one JSON object per line):

```json
{
  "timestamp":     "2026-03-25T11:23:45.123456",
  "call_type":     "call_2_daily_event",
  "system_prompt": "...",
  "user_prompt":   "(full prompt string sent to LLM)",
  "output":        "(generated text or structured dict, excluding _usage)"
}
```

- `output` excludes the `_usage` field (token counts are tracked separately)
- Only the final successful call is logged per operation (retries are not recorded)
- Logging can be disabled via `ENABLE_LLM_CALL_LOGGING = False` in config.py

---

## 6. Output Structure

All memory modules must produce **the same output schema**. Each `results_*.json` is a list of session results:

```json
[
  {
    "session_id": 1,

    "config_metadata": {
      "config_name":   "<str>",
      "model":         "<str, full model path>",
      "subset":        "<str, 'opposed' or 'supportive'>",
      "embedding_model": "<str>",
      "temperature":   "<float>",
      "max_tokens":    "<int>",
      "session_range": ["<start_session int>", "<end_session int>"],
      "...module-specific hyperparameters...": "..."
    },

    "memory_at_qa_start": {
      "num_memories":         "<int>",
      "total_content_tokens": "<int, estimated via len(content) // 4>"
    },

    "qa_results": [
      {
        "question": "...",
        "generated_answer": "...",
        "ground_truth_answer": "...",
        "retrieved_memories": [
          "<module-specific — see each module's README for the exact schema>"
        ],
        "qa_tokens": {"input": 300, "output": 50, "model": "meta-llama/..."}
      }
    ],

    "token_statistics": {
      "<call_type_phase1_A>": {"input": 12000, "output": 1800, "llm_calls": 30},
      "<call_type_phase1_B>": {"input": 12000, "output": 2400, "llm_calls": 30},
      "...": "...",
      "<call_type_qa>":       {"input": 9900,  "output": 320,  "llm_calls": 5},
      "total_input":     35900,
      "total_output":    4520,
      "total_llm_calls": 65
    },

    "<module_specific_phase1_stats_key>": {
      "...": "... (optional — only present if the module has notable Phase 1 internal events)"
    },

    "memory_snapshot_path": "memory_snapshots/session_1/"
  }
]
```

**Notes:**
- `config_metadata` fields beyond the common ones above are module-specific (e.g., `retrieve_k`, `forgetting_divisor`, `evolution_threshold`)
- `memory_at_qa_start.total_content_tokens`: estimated via `len(content) // 4`
- `token_statistics` call type names are module-specific; see each module's README
- The module-specific Phase 1 statistics field (e.g., `"evolution_statistics"` in amem, `"phase1_statistics"` in memorybank) is **optional** — omit entirely if the module has no notable Phase 1 internal events. Key name is module-defined.
- `generated_answer` for `supportive` subset: one of `{yes, no}` (or `"unknown"` on error fallback); for `opposed`: free-form text
- `retrieved_memories` schema is **module-specific** — exact fields vary per module. Examples:
  - **amem**: `{"session_id", "conv_id", "turn_id"}`
  - **ldagent**: `{"session_id", "conv_id", "virtual_seconds", "score"}` (no `turn_id`; LTM summaries, not individual turns)
  - See each module's README for the full schema

### Directory Structure (per module)

```text
{module_name}/
├── config.py
├── load_dataset.py
├── run_experiment.py
├── merge_results.py
├── {module_specific_files}
├── {config}_outputs_{model_name}_{subset}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model_name}_{subset}_session_{start}_{end}.json
│   │   ├── checkpoint_{model_name}_{subset}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{id}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{id}/
│   │   ├── prompt_log/
│   │   │   └── session_{id}/
│   │   │       ├── {call_type_1}/calls.jsonl
│   │   │       ├── {call_type_2}/calls.jsonl
│   │   │       └── ...
│   │   └── logs/
│   └── results_{model_name}_{subset}_merged.json
└── logs/
```

### Command-line Interface (shared)

```bash
python run_experiment.py \
    --start-session 0 --end-session 4 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --config config
```

`--config` specifies the config file name (without `.py`) located in the module directory.
The file is **dynamically loaded at runtime**, so different config files (e.g. `config_large.py`,
`config_20turns.py`) can be used without modifying the script. The config name is also used
as the output directory prefix: `{config}_outputs_{model}_{subset}/`. Defaults to `"config"`.

---

## 7. Retrieval Logging

Each memory module must log detailed retrieval information for every retrieval operation. **The exact log format is module-specific and must be designed per module**, as different memory architectures have different retrieval mechanisms.

### What Must Be Logged

At minimum, each retrieval log entry should include:

| Field | Description |
|---|---|
| `timestamp` | When the retrieval occurred |
| `phase` | `"prompt_construction"` (Phase 1) or `"qa"` (Phase 2) |
| `query` | The query used for retrieval (user utterance or QA question) |
| `turn_metadata` | (session_id, conv_id, turn_id) identifying the turn |
| `retrieved_items` | List of retrieved memories with their scores/rankings |
| `retrieval_scores` | Module-specific relevance scores (e.g., cosine similarity, BM25, reranking score) |

> **Note:** Full prompt snapshots are **not** stored in the retrieval log. They are logged separately in `prompt_log/` (in the module-specific QA call folder for Phase 2) and would be redundant here.

### Log Format

Retrieval logs are stored as **JSONL** (one JSON object per line) in `retrieval_logs/session_{id}_retrieval_log.jsonl`. Each line represents one retrieval operation:

```json
{
  "timestamp": "2026-03-12T10:30:00",
  "phase": "prompt_construction",
  "session_id": 1,
  "conv_id": 0,
  "turn_id": 5,
  "query": "What do you think about remote work?",
  "num_retrieved": 3,
  "retrieved_items": [
    {
      "memory_id": "...",
      "content_preview": "...",
      "score": 0.87,
      "source_turn": {"session_id": 1, "conv_id": 0, "turn_id": 2}
    }
  ],
  "module_specific": {}
}
```

The `module_specific` field is a free-form object where each module can log its own retrieval details (e.g., keyword lists, decay factors, graph traversal info).
