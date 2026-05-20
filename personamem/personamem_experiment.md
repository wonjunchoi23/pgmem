# Experiment Specification: Memory-augmented LLM Agents on PersonaMem

This document defines the experiment protocol for evaluating memory-augmented LLM agents
on the PersonaMem benchmark.

> **Response generation is not performed.** Phase 1 builds memory from ground-truth
> conversation turns only. Only QA answering (Phase 2) calls the LLM.

For a diff against the ImplexConv protocol, see [`difference.md`](difference.md).

---

## 1. Task Formulation

### Dataset

- **PersonaMem** benchmark (Jiang et al., 2025)
- Three context-size variants: `32k`, `128k`, `1M`
- Selected via `--benchmark-size` argument

| Size | Shared contexts | Total QA | Avg messages/context |
|---|---|---|---|
| 32k | 37 | 589 | 169 |
| 128k | 110 | 2,727 | 736 |
| 1M | 33 | 2,674 | 3,548 |

Dataset files (under `dataset/`):

```
questions_32k.csv            shared_contexts_32k.jsonl
questions_128k.csv           shared_contexts_128k.jsonl
questions_1M.csv             shared_contexts_1M.jsonl
```

### Dataset Schema

**`questions_[SIZE].csv`** — one row per QA question:

| Column | Type | Description |
|---|---|---|
| `persona_id` | int | Unique persona identifier |
| `question_id` | str (UUID) | Unique question identifier |
| `question_type` | str | One of 7 in-situ question types |
| `topic` | str | Conversation topic (e.g. `musicRecommendation`) |
| `user_question_or_message` | str | The question text |
| `correct_answer` | str | Ground truth: `(a)`, `(b)`, `(c)`, or `(d)` |
| `all_options` | str (Python list) | 4 answer options as a stringified list |
| `shared_context_id` | str (SHA-256) | Key into the shared contexts JSONL |
| `end_index_in_shared_context` | int | `context[:end_index]` gives the conversation history visible at QA time |
| `context_length_in_tokens` | int | Total token count of context |
| `distance_to_ref_in_blocks` | int | Blocks from question to most recent preference mention |
| `distance_to_ref_in_tokens` | int | Tokens from question to most recent preference mention |
| `num_irrelevant_tokens` | int | Tokens from off-topic interactions |
| `distance_to_ref_proportion_in_context` | str | Relative position of latest preference in context |

**`shared_contexts_[SIZE].jsonl`** — one JSON object per line:

```json
{"<shared_context_id>": [
    {"role": "system",    "content": "Current user persona: ..."},
    {"role": "user",      "content": "User: Hi there! ..."},
    {"role": "assistant", "content": "Assistant: That sounds great! ..."},
    ...
]}
```

**Block structure within a shared context:**

A `system` message marks the start of each block. All subsequent `user`/`assistant`
messages until the next `system` message belong to that block.

```
system[0]         → block 0 start  (persona description repeated)
user, assistant, user, ...
system[k]         → block 1 start
user, assistant, ...
...
```

Each block corresponds to one conversation session at a particular time period
(`init`, `next_week`, `next_month`, or `next_year`).

### Task: Personalized QA via Memory-Augmented Retrieval

Answer persona-relevant multiple-choice questions by retrieving from a compressed
memory of the conversation history.

- All questions are **4-choice** (a/b/c/d).
- **Evaluation metric**: accuracy (exact match of predicted letter to `correct_answer`).
- QA exchanges are not included in dialogue history and are not stored in memory.

**7 question types:**

| Type | Description |
|---|---|
| `recall_user_shared_facts` | Recall static facts the user has shared |
| `suggest_new_ideas` | Suggest items not mentioned in history |
| `acknowledge_latest_preferences` | Recognize the user's most recent preference |
| `track_full_preference_evolution` | Track how preferences shifted over time |
| `revisit_reasons_behind_preference_updates` | Recall reasons for preference changes |
| `provide_preference_aligned_recommendations` | Proactively recommend aligned options |
| `generalize_to_new_scenarios` | Transfer learned preferences to new contexts |

---

## 2. Experiment Flow

Unit of processing: **Individual shared context C_i**

Shared contexts are sorted by descending QA count before indexing.
`--start-session` / `--end-session` refer to integer positions in this sorted list.

For each context C_i where i = start_session, ..., end_session:

### Phase 1: Memory Construction on C_i

Process all `user` and `assistant` messages in C_i in order. `system` messages are
skipped (used only as block boundary markers). No LLM call for response generation.

```text
block_idx = -1
local_msg_idx = 0

for each message in C_i:
    if message.role == "system":
        block_idx += 1
        local_msg_idx = 0
        continue                          # skip system messages

    timestamp = f"{context_index:04d}_{block_idx:04d}_{local_msg_idx:04d}"
    memory_content = message.content      # as-is; already contains speaker prefix

    1. Retrieve relevant information from memory module
    2. Store message in memory module (timestamp above)
    3. Update module-specific components (evolution, etc.)
    4. Log retrieval details

    local_msg_idx += 1

After all messages in C_i:
    5. Finalize memory (flush pending summaries, run global synthesis)
    6. Record memory state at QA start (memory_at_qa_start)
    7. Save final memory snapshot for context C_i
```

**Memory content format:** Content is stored as-is from the dataset, e.g.:
```
"User: Hi there! I've recently been diving deeper into my passion for music..."
"Assistant: That sounds absolutely incredible! Your enthusiasm for..."
```

### Phase 2: QA Answering on C_i

QA is performed after all messages have been processed and memory is frozen.
No new memory is constructed during this phase.

```text
for each QA in C_i (ordered by end_index_in_shared_context):
    1. Retrieve relevant memories using the question as query
    2. Build QA prompt: retrieved memory + question + all 4 options
    3. Generate answer (guided_json, return_usage=True)
    4. Log retrieval details
    5. Record: generated letter (a/b/c/d), ground truth letter, retrieved_memories metadata
```

**QA prompt framing:** The LLM is told it is answering a multiple-choice question
about a user based on stored memory, not generating a conversational response.

**Answer normalization:** Extract the letter from `{"answer": "a" | "b" | "c" | "d"}`.
If extraction fails, record `"unknown"`.

**Note on `end_index_in_shared_context`:** This field indicates the message list
position at which each QA was originally inserted (i.e., `context[:end_index]` is
the visible history at QA time in the original benchmark). In this experiment
(Phase 1 uses the full context), `end_index_in_shared_context` is recorded in
results as metadata only and does not limit memory construction.

### Phase 3: Cleanup

```text
6. Aggregate statistics:
   - token_statistics: per-call-type input/output/llm_calls, plus totals
   - memory_at_qa_start: memory state at end of Phase 1
   - evolution_statistics: A-MEM-specific Phase 1 internal stats
7. Save results (atomic write)
8. Clear ALL memory module state for next context
9. Update checkpoint
```

---

## 3. Implementation Constraints

### Memory Content Rule

- Only `user` and `assistant` messages are stored in memory.
- `system` messages are never stored; they serve only as block boundaries.
- QA question/answer pairs are never stored in memory.
- No LLM response is generated during Phase 1.

### Virtual Time Model

```
CONV_IDS_PER_DAY = 1      # 1 block = 1 virtual day
MINUTES_PER_TURN = 10     # each message within a block = 10 minutes
```

Timestamp format: `"{context_index:04d}_{block_idx:04d}_{local_msg_idx:04d}"`

Virtual time display: `"Day {block_idx // CONV_IDS_PER_DAY + 1}, {HH}:{MM}"`

Note: `local_msg_idx` counts only `user`/`assistant` messages within the block
(system messages are excluded from the count).

### Checkpoint & Resumption

- Checkpoint saved after each context completion (atomic write).
- On restart: loads checkpoint and resumes from the next unprocessed context.
- Format: `{"completed_session_ids": [<int>, ...]}` (same as ImplexConv).

### Token Tracking

Same structure as ImplexConv. All LLM calls track tokens via `return_usage=True`.

```json
{
  "call_2_note_construction": {"input": ..., "output": ..., "llm_calls": ...},
  "call_3_evolution":         {"input": ..., "output": ..., "llm_calls": ...},
  "call_4_qa":                {"input": ..., "output": ..., "llm_calls": ...},
  "total_input":     ...,
  "total_output":    ...,
  "total_llm_calls": ...
}
```

### Configuration Parameters

```text
# Experiment protocol (same as ImplexConv unless noted)
CHECKPOINT_INTERVAL = 1
TEMPERATURE = 0.7
MAX_TOKENS = 750
JSON_RETRY = 5

# Batch settings
BATCH_SIZE = 4
QA_BATCH_SIZE = 64

# Memory module (A-MEM specific)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVE_K = 5
EVOLUTION_THRESHOLD = 100

# Virtual time model  ← CHANGED from ImplexConv
CONV_IDS_PER_DAY = 1     # was 2 in ImplexConv
MINUTES_PER_TURN = 10

# Dataset paths
BENCHMARK_SIZES = ["32k", "128k", "1M"]
DATASET_QUESTIONS_32K  = "dataset/questions_32k.csv"
DATASET_CONTEXTS_32K   = "dataset/shared_contexts_32k.jsonl"
DATASET_QUESTIONS_128K = "dataset/questions_128k.csv"
DATASET_CONTEXTS_128K  = "dataset/shared_contexts_128k.jsonl"
DATASET_QUESTIONS_1M   = "dataset/questions_1M.csv"
DATASET_CONTEXTS_1M    = "dataset/shared_contexts_1M.jsonl"
```

---

## 4. Memory Module Interface

Same interface as ImplexConv. See `implexconv_experiment.md` §4 for the full table.

The only behavioral difference is in what is passed to **Store**:

- Content is the raw message content string (already includes speaker prefix).
- `system` messages are never passed to the memory module.

---

## 5. LLM Call Logging

Same structure as ImplexConv. Call types:

| Call type | Phase | Description |
|---|---|---|
| `call_2_note_construction` | 1 | Note analysis prompt |
| `call_3_evolution` | 1 | Memory evolution prompt |
| `call_4_qa` | 2 | Multiple-choice QA answering |

---

## 6. Output Structure

```text
amem/
├── {config}_outputs_{model}_{benchmark_size}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{benchmark_size}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{benchmark_size}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{context_index}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{context_index}/
│   │   ├── prompt_log/
│   │   │   └── session_{context_index}/
│   │   │       ├── call_2_note_construction/calls.jsonl
│   │   │       ├── call_3_evolution/calls.jsonl
│   │   │       └── call_4_qa/calls.jsonl
│   │   └── logs/
│   └── results_{model}_{benchmark_size}_merged.json
```

### Result Schema

```json
[
  {
    "context_index": <int>,
    "persona_id": <int>,
    "shared_context_id": <str>,

    "config_metadata": {
      "config_name":         "<str>",
      "model":               "<str>",
      "benchmark_size":      "<str: 32k | 128k | 1M>",
      "embedding_model":     "<str>",
      "retrieve_k":          "<int>",
      "evolution_threshold": "<int>",
      "temperature":         "<float>",
      "max_tokens":          "<int>",
      "session_range":       ["<int>", "<int>"]
    },

    "memory_at_qa_start": {
      "num_memories":         "<int>",
      "total_content_tokens": "<int>"
    },

    "qa_results": [
      {
        "question":                      "<str>",
        "question_type":                 "<str>",
        "topic":                         "<str>",
        "all_options":                   ["<str>", "<str>", "<str>", "<str>"],
        "generated_answer":              "<str: a | b | c | d | unknown>",
        "ground_truth_answer":           "<str: a | b | c | d>",
        "end_index_in_shared_context":   "<int>",
        "retrieved_memories": [
          {
            "context_index":  "<int>",
            "block_idx":      "<int>",
            "local_msg_idx":  "<int>"
          }
        ],
        "qa_tokens": {"input": "<int>", "output": "<int>", "model": "<str>"}
      }
    ],

    "token_statistics": {
      "call_2_note_construction": {
        "input": "<int>", "output": "<int>", "llm_calls": "<int>",
        "parse_fallback_count": "<int>"
      },
      "call_3_evolution": {"input": "<int>", "output": "<int>", "llm_calls": "<int>"},
      "call_4_qa":        {"input": "<int>", "output": "<int>", "llm_calls": "<int>"},
      "total_input":     "<int>",
      "total_output":    "<int>",
      "total_llm_calls": "<int>"
    },

    "evolution_statistics": {
      "evo_triggered_count": "<int>",
      "actions_taken": {"strengthen": "<int>", "update_neighbor": "<int>"}
    },

    "memory_snapshot_path": "<str or null>"
  }
]
```

### Command-line Interface

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

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start-session` | int | required | First context index (into QA-sorted list) |
| `--end-session` | int | required | Last context index (inclusive) |
| `--benchmark-size` | str | required | `32k`, `128k`, or `1M` |
| `--model` | str | config default | Model path |
| `--tensor-parallel` | int | config default | Tensor parallelism |
| `--gpu-memory` | float | config default | GPU memory utilization |
| `--max-model-len` | int | `None` | Optional vLLM max model length |
| `--batch-size` | int | `BATCH_SIZE` | Sessions processed in parallel |
| `--config` | str | `config_0` | Config filename without `.py` |

---

## 7. Retrieval Logging

Same format as ImplexConv. One JSONL file per context in `retrieval_logs/`.

`source_turn` in each retrieved item uses PersonaMem keys:

```json
{
  "source_turn": {
    "context_index": <int>,
    "block_idx":     <int>,
    "local_msg_idx": <int>
  }
}
```

---

## 8. Validation Checklist

**Invariants:**
1. `system` messages never stored in memory
2. QA exchanges never stored in memory
3. No LLM call for response generation during Phase 1
4. Memory completely cleared between contexts
5. Memory snapshot saved at end of each context (before cleanup)
6. `total_input = sum(call_type["input"] for all call types)`
7. `total_output = sum(call_type["output"] for all call types)`
8. `total_llm_calls = sum(call_type["llm_calls"] for all call types)`

**Common pitfalls:**
- Storing `system` messages in memory
- Using `end_index_in_shared_context` to limit Phase 1 (not done in this protocol)
- Forgetting to increment `block_idx` on each `system` message
- Counting `system` messages in `local_msg_idx`
