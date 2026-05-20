# LoComo Experiment Specification

This document defines the **global experiment protocol** for evaluating a
memory-augmented language agent on the LoComo dataset.

The intent of this document is to be **module-agnostic**. It should help a new
memory module plug into the same experiment setting without needing to copy the
design of any existing implementation. Wherever possible, this document
describes:

- what the experiment must do
- what outputs it should produce
- what is left open to module design

The experiment is a **QA-only evaluation**. The conversation is replayed into
memory, then questions are answered from the resulting memory state.

---

## 1. Goal

The goal is to evaluate whether a memory module can retain and retrieve useful
information from a multi-session conversation well enough to answer LoComo
questions.

The experiment therefore measures the following combined behavior:

1. how the module stores information during conversation replay
2. how it organizes, summarizes, or updates memory internally
3. how it retrieves relevant information at QA time
4. how well an LLM can answer questions from the retrieved context

This document does **not** prescribe one memory architecture. A module may use
raw turns, summaries, personas, graphs, compressed notes, latent states, or any
combination of these, as long as it obeys the experiment contract below.

---

## 2. Core Assumptions

- The experiment unit is an **individual LoComo sample**.
- Each sample is processed independently.
- Memory is **cleared between samples**.
- The experiment is **QA-only**.
- No assistant response generation is performed while replaying the dialogue.
- QA exchanges are **not** stored back into memory.
- The same dataset-level normalization should be used across modules so that
  results remain comparable.

---

## 3. Dataset

### Dataset file

- Dataset: **LoComo**
- Canonical file: `dataset/locomo10.json`

### Simplified schema

```json
{
  "sample_id": "conv-26",
  "conversation": {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      {
        "speaker": "Caroline",
        "dia_id": "D1:1",
        "text": "Hey Mel! Good to see you!"
      }
    ],
    "session_2_date_time": "7:55 pm on 9 June, 2023",
    "session_2": []
  },
  "qa": [
    {
      "question": "When did Caroline go to the LGBTQ support group?",
      "answer": "7 May 2023",
      "evidence": ["D1:3"],
      "category": 3
    },
    {
      "question": "What does Caroline love most about camping with her family?",
      "adversarial_answer": "Being present and bonding with her family",
      "evidence": ["D18:21"],
      "category": 5
    }
  ],
  "event_summary": {},
  "observation": {},
  "session_summary": {}
}
```

### Dataset interpretation

- `conversation` contains a multi-session dialogue between two named speakers.
- Sessions appear as `session_N` / `session_N_date_time` pairs.
- `qa` is sample-level and evaluated only **after** the full conversation has
  been processed.
- `evidence` is the dataset's ground-truth support annotation. It is **not**
  the model retrieval output.
- `event_summary`, `observation`, and `session_summary` may be loaded and made
  available to a module, but the global protocol does not require them to be
  used.

---

## 4. Shared Dataset Normalization

To keep different modules comparable, the loader should normalize LoComo in the
same way.

### Session ordering

- Process sessions in ascending `session_N` order.
- Skip missing or empty sessions.

### Image turns

Some turns contain `img_url` and `blip_caption`. These should be converted to
plain text during dataset loading:

```python
caption = f"[Image: {turn['blip_caption']}]"
text = f"{caption} {text}" if text else caption
```

Recommended interpretation:

- `img_url` is ignored for the experiment.
- `blip_caption` becomes part of the turn text.
- After loading, the rest of the experiment should treat the turn exactly like
  a normal text turn.

### QA normalization

Define a normalized ground-truth answer field:

- Categories `1-4`: use `answer`
- Category `5`: use `adversarial_answer`

A convenient loader-level abstraction is:

```python
final_answer = adversarial_answer if category == 5 else answer
```

---

## 5. Task Formulation

The task is:

1. replay the full conversation into a memory module
2. freeze the memory state at QA start
3. answer every LoComo question from that memory state

The memory representation is intentionally open-ended. A module may store:

- raw dialogue turns
- session summaries
- global summaries
- speaker traits or personas
- graph links
- compressed notes
- other structured or unstructured memory forms

What matters is that the module exposes enough behavior to support the
experiment loop and produces the required outputs.

---

## 6. Global Experiment Flow

### Phase 1: Memory Construction

For a single sample:

1. Iterate through sessions in chronological order.
2. Iterate through turns within each session in original order.
3. Store each turn into memory using the module's chosen representation.
4. Run any module-internal processing needed for memory construction.
   This may include summarization, note extraction, linking, consolidation, or
   other updates.
5. After the full conversation is processed, finalize any remaining pending
   state.
6. Record `memory_at_qa_start`.
7. Optionally save a memory snapshot.

### Phase 2: QA Answering

For each QA item in the sample:

1. Use the QA question as the retrieval query.
2. Retrieve relevant memory.
3. Build a category-aware QA prompt.
4. Generate an answer.
5. Record the generated answer, normalized ground truth, retrieved memory
   metadata, and token usage.

### Phase 3: Cleanup

After QA is finished:

1. Save results.
2. Clear the entire memory state for the sample.
3. Update checkpoint state.

---

## 7. Hard Invariants

These rules should hold for every module.

1. **No response generation during dialogue replay.**
   Phase 1 is for memory construction only.
2. **No new dialogue memory during QA.**
   Phase 2 may retrieve, summarize prompt context, and generate answers, but it
   should not append QA interactions back into long-term memory.
3. **Memory is sample-local.**
   Nothing from one sample should leak into another.
4. **Ground truth for category 5 uses `adversarial_answer`.**
5. **Results and checkpoints should be written atomically.**
6. **Token totals should equal the sum of per-call-type token counts.**

---

## 8. Time Handling

LoComo includes real date/time strings at the session level.

### Required behavior

- Use `session_N_date_time` as the source of temporal information.
- The module should preserve enough time information to answer temporal
  questions and to support any time-sensitive retrieval logic it uses.

### Open design choice

Within a session, LoComo does not provide a unique timestamp for every turn.
If a module needs per-turn timestamps, it may derive them deterministically,
for example by:

- assigning the same session timestamp to all turns in the session
- adding a fixed offset per turn
- storing the session time plus turn index

The exact strategy is module-specific, but it should be:

- deterministic
- documented
- consistent across runs

### What to avoid

- Do not use an unrelated virtual time model that ignores LoComo session dates
  entirely.

---

## 9. QA Categories

LoComo questions use five categories:

| Category | Type | Ground truth |
|---|---|---|
| 1 | Single-hop | `answer` |
| 2 | Multi-hop | `answer` |
| 3 | Temporal | `answer` |
| 4 | Open-domain | `answer` |
| 5 | Adversarial | `adversarial_answer` |

All five categories are included in evaluation.

### Category-specific QA behavior

The exact prompt wording may vary, but the intent should be aligned:

- Categories `1`, `2`, `4`:
  answer with a short phrase grounded in retrieved memory
- Category `3`:
  answer with the shortest reasonable approximate date grounded in the
  conversation timeline
- Category `5`:
  choose between:
  - `adversarial_answer`
  - `"Not mentioned in the conversation"`

### Reproducibility rule for category 5

To keep runs reproducible, the ordering of the two category-5 answer choices
should be deterministic rather than random.

Recommended seed:

```text
f"{sample_id}::{qa_idx}::{question}"
```

---

## 10. Batching

Batching is allowed and encouraged for efficiency, as long as it preserves the
same per-sample semantics as an isolated run.

Examples of acceptable batching strategies:

- batching multiple samples together during Phase 1
- batching QA calls by temperature group
- batching prompts in fixed-size chunks

Requirements:

- batching must not mix memory state across samples
- batching must not change the order of turns within a sample
- batching must not introduce nondeterministic category-5 choice ordering

---

## 11. Recommended Memory Module Interface

The exact class and method names are flexible, but a module should provide
capabilities equivalent to the following.

### Required capabilities

- **Store turn**
  accept one normalized dialogue turn and add it to memory
- **Finalize phase 1**
  flush any pending internal state after conversation replay
- **Retrieve**
  return relevant memory for a query string
- **Retrieve with metadata**
  return enough metadata to support logging, ideally including source `dia_id`
- **Get memory stats**
  return a summary of the memory state at QA start
- **Save snapshot**
  persist memory for inspection if enabled
- **Clear**
  fully reset all state between samples

### Strongly recommended capabilities

- **Per-call-type token accounting**
- **Prompt logging**
- **Retrieval logging**
- **Deterministic QA prompt construction**

---

## 12. Configuration Surface

Different memory modules will need different hyperparameters, but the
experiment-level configuration should stay recognizable.

### Global experiment settings

Recommended common keys:

```text
DATASET_PATH
TEMPERATURE
TEMPERATURE_C5
MAX_TOKENS
JSON_RETRY
ENABLE_CHECKPOINTING
SAVE_MEMORY_SNAPSHOTS
ENABLE_LLM_CALL_LOGGING
```

### Batch-related settings

If batching is used, recommended keys:

```text
BATCH_SIZE
QA_BATCH_SIZE
```

### Module-specific settings

These may vary:

```text
EMBEDDING_MODEL
RETRIEVE_K
SUMMARIZE_MAX_TOKENS
DIST_THRESHOLD
FORGETTING_DIVISOR
DECAY_TEMP
PERSONA_LIMITS
GRAPH_OR_LINKING_PARAMS
OTHER_MEMORY_INTERNALS
```

The key point is:

- experiment-level knobs should stay comparable across modules
- memory-internal knobs are free to vary

---

## 13. Checkpointing and Resumption

Checkpointing should happen after a sample is fully completed.

### Recommended format

Use **sample-ID-based** checkpoints:

```json
{
  "completed_sample_ids": ["conv-1", "conv-2"]
}
```

Why this format is preferred:

- it works for both sequential and batched execution
- it avoids ambiguity when only part of a batch succeeds

### Resumption rule

On restart:

1. load the checkpoint
2. skip samples already listed as complete
3. continue with the remaining target sample range

---

## 14. Token Tracking

All LLM calls should be tracked by **call type**.

This includes:

- QA generation calls
- memory-construction calls
- summarization calls
- persona or attribute extraction calls
- any other module-internal LLM calls

### Required per-call-type schema

```json
{
  "input": 0,
  "output": 0,
  "llm_calls": 0
}
```

### Required aggregate fields

```json
{
  "total_input": 0,
  "total_output": 0,
  "total_llm_calls": 0
}
```

### Aggregation rule

```text
total_input     = sum(call_type.input)
total_output    = sum(call_type.output)
total_llm_calls = sum(call_type.llm_calls)
```

Use the key name **`llm_calls`**, not `api_calls`.

---

## 15. Result File Contract

Each results file should contain a list of sample-level results.

### Recommended top-level sample result schema

```json
{
  "sample_id": "conv-26",
  "config_metadata": {
    "config_name": "config_0",
    "model": "meta-llama/...",
    "temperature": 0.7,
    "temperature_c5": 0.5,
    "max_tokens": 750,
    "sample_range": [0, 9]
  },
  "memory_at_qa_start": {
    "num_memories": 42,
    "total_content_tokens": 1200
  },
  "qa_results": [
    {
      "question": "When did Caroline go to the LGBTQ support group?",
      "category": 3,
      "generated_answer": "7 May 2023",
      "ground_truth_answer": "7 May 2023",
      "evidence": ["D1:3"],
      "retrieved_memories": [
        {
          "dia_id": "D1:3",
          "content_preview": "...",
          "score": 0.91
        }
      ],
      "qa_tokens": {
        "input": 300,
        "output": 20,
        "model": "meta-llama/..."
      }
    }
  ],
  "token_statistics": {
    "call_1_some_internal_step": {
      "input": 1200,
      "output": 300,
      "llm_calls": 10
    },
    "call_2_qa": {
      "input": 900,
      "output": 120,
      "llm_calls": 5
    },
    "total_input": 2100,
    "total_output": 420,
    "total_llm_calls": 15
  },
  "memory_snapshot_path": "memory_snapshots/sample_conv-26/"
}
```

### Required notes

- `ground_truth_answer` should use the normalized final answer rule
- `evidence` should be copied from the dataset as-is
- `retrieved_memories` is module-specific, but should include source metadata
  when available
- `qa_tokens` should reflect the final successful QA generation call

### Optional fields

A module may add additional top-level fields such as:

- phase-1 internal statistics
- evolution statistics
- forgetting statistics
- memory subtype counts
- any other diagnostic information

Optional fields are encouraged as long as the core schema remains stable.

---

## 16. Memory Stats at QA Start

`memory_at_qa_start` is intentionally flexible, but should at minimum give a
useful summary of the memory state just before QA begins.

### Minimum recommended fields

```json
{
  "num_memories": 42,
  "total_content_tokens": 1200
}
```

### Optional extensions

Depending on the module, additional fields may be useful:

- summary count
- graph edge count
- persona entry count
- total retrievable document count
- active short-term memory count
- compressed memory count

The goal is to make QA-start memory state inspectable without forcing every
module into the same internal structure.

---

## 17. Retrieval Logging

Each retrieval event should be written in JSONL format.

### Minimum recommended fields

```json
{
  "phase": "qa",
  "sample_id": "conv-26",
  "session_id": null,
  "dia_id": null,
  "query": "When did Caroline go to the LGBTQ support group?",
  "num_retrieved": 3,
  "retrieved_items": [
    {
      "dia_id": "D1:3",
      "content_preview": "...",
      "score": 0.91
    }
  ],
  "module_specific": {}
}
```

### Notes

- The exact retrieval log format may vary by module.
- `num_retrieved` may be an integer or a parallel breakdown by memory type.
- `retrieved_items` should contain the most useful traceable information the
  module can expose.
- QA retrieval logging is strongly recommended for every module.

If a module performs meaningful retrieval during Phase 1, it may also log those
operations, but the global experiment primarily requires QA-time visibility.

---

## 18. Prompt Logging

All LLM calls should be logged per sample, grouped by call type.

### Recommended directory pattern

```text
prompt_log/
└── sample_{id}/
    ├── call_1_xxx/calls.jsonl
    ├── call_2_yyy/calls.jsonl
    └── call_N_qa/calls.jsonl
```

### Recommended entry schema

```json
{
  "call_type": "call_2_qa",
  "system_prompt": "",
  "user_prompt": "...",
  "output": "..."
}
```

Guidelines:

- log only the final successful call for the operation
- exclude raw token-usage blobs from the logged `output`
- keep token accounting in `token_statistics`

---

## 19. Open Design Choices

The following are deliberately left to module design:

- speaker prefix format used in storage
- exact memory representation
- how much of the conversation to store verbatim
- whether to maintain summaries or personas
- retrieval algorithm
- reranking strategy
- consolidation strategy
- how to format retrieved context for QA
- how to represent time internally within a session

A new module should document these choices in its own module README, but it
does not need to change the global experiment contract.

---

## 20. Validation Checklist

Before considering a new module implementation compatible with this experiment,
check the following:

1. The full conversation is processed before QA starts.
2. No response generation occurs during Phase 1.
3. QA questions are answered only after memory is finalized.
4. QA interactions are not written back into memory.
5. Category-5 ground truth uses `adversarial_answer`.
6. Category-5 answer ordering is deterministic.
7. Memory is cleared between samples.
8. Checkpoint writes are atomic.
9. Result writes are atomic.
10. `token_statistics.total_*` equals the sum of per-call-type values.
11. The loader converts image turns to text consistently.
12. `qa_results.evidence` is copied from the dataset, not inferred from model
    retrieval.

---

## 21. Practical Summary

In short, the global LoComo experiment is:

1. load and normalize the dataset
2. replay one sample's full conversation into memory
3. freeze and inspect memory at QA start
4. answer all sample questions from that memory state
5. save answers, retrieval traces, token usage, and memory snapshots
6. clear state and move to the next sample

Everything inside the memory module may change. The experiment contract should
not.
