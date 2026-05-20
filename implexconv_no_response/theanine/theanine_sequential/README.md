# Theanine — Timeline-based Memory Management

Archived sequential reference kept under `theanine/theanine_sequential/`.
The current main implementation lives in `theanine/` and is the version intended for active runs.

Theanine (Oh et al., NAACL 2025) memory module for the ImplexConv experiment.
Builds a relation-aware memory graph from dialogue summaries; retrieves relevant
memories by tracing chronological **timeline paths** through the graph, then
refines each path into natural language for LLM-based QA answering.

---

## Architecture

Three cooperating components:

- **`MemoryGraph`** (`memory_graph.py`): Summarizes completed conv_id dialogues
  into fact sentences, embeds them with sentence-transformers, and links nodes
  across conv_ids via ATOMIC 2020 relation types (Changed / Cause / Reason /
  HinderedBy / React / Want / SameTopic / None). Memory accumulates across
  conv_ids within a session and is cleared between sessions.
- **`TimelineRetriever`** (`timeline.py`): Retrieves top-k nodes by cosine
  similarity, builds all possible timeline paths (BFS in both temporal
  directions), samples one unique path per retrieved node, and refines each path
  into natural-language context via an LLM call.
- **`Generator`** (`generator.py`): Generates QA answers (Phase 2) using refined
  timeline context. All LLM output is JSON-structured.

---

## Files

| File | Role |
|---|---|
| `memory_graph.py` | `MemoryGraph` — node creation, embedding, relation linking |
| `timeline.py` | `TimelineRetriever` — path traversal, timeline sampling, LLM refinement |
| `generator.py` | `Generator` — QA generation (JSON structured output) |
| `theanine_module.py` | `TheanineModule` — unified entry point wrapping all three components |
| `run_experiment.py` | Experiment loop (session loop, conv-boundary memory construction, checkpointing) |
| `config_0.py` | All hyperparameters and output path helpers |
| `load_dataset.py` | ImplexConv dataset loader (`Session`, `Turn`, `QAPair` dataclasses) |
| `merge_results.py` | Merge results from multiple session-range runs |

---

## Memory Graph Construction

Memory is **not** built per-turn. Instead, the full dialogue of a completed
`conv_id` is summarized into a list of fact sentences, each becoming one
`MemoryNode`. Nodes are linked across conv_ids by running pairwise relation
extraction against the top-j most similar past nodes.

```
conv_id boundary detected (or last turn of session)
    │
    ▼
_summarize_conv(full GT dialogue)
    └─ 1 LLM call (guided_json) → {"sentences": ["fact1", "fact2", ...]}
    │
    ▼
_create_nodes_from_summary()
    └─ node_id = "c{conv_id}-m{idx}"  (1-indexed)
    │
    ▼
_embed_nodes()  (sentence-transformers batch encode)
    │
    ▼
for each new node:
    _find_associative(top-j similar nodes, conv_id < current)
    └─ _find_links():
           for each past candidate (sorted by conv_id desc):
               _extract_relation(sentence1, sentence2, dialogue1, dialogue2)
               → 1 LLM call (guided_json) → {"explanation": "...", "relation": "..."}
               → bidirectional link if relation ≠ "None"
    │
    ▼
register new nodes in self.nodes
(same-conv_id nodes never link to each other)
```

**Node key format:** `c{conv_id}-m{idx}` (e.g. `c2-m3`)

**Adaptation from original:** The original Theanine uses offline batch
preprocessing per episode ("session" in paper). Here, memory is constructed
**online** at each `conv_id` boundary during Phase 1.

| Original Theanine | This experiment |
|---|---|
| "session" (s1, s2, …) | `conv_id` |
| "episode" (all sessions) | `session` |
| Node key `s{n}-m{i}` | Node key `c{conv_id}-m{i}` |

---

## JSON Mode (guided_json)

All three internal LLM calls use **constrained decoding** (`guided_json`) to
prevent `<think>` reasoning tokens from being stored in memory nodes:

| Call | Schema | Result field used |
|---|---|---|
| `_summarize_conv` | `{"sentences": [str, ...]}` | `result["sentences"]` |
| `_extract_relation` | `{"explanation": str, "relation": enum}` | `result["relation"]` |
| `refine_timeline` | `{"refined_text": str}` | `result["refined_text"]` |

The `relation` field is constrained to the enum:
`Changed | Cause | Reason | HinderedBy | React | Want | SameTopic | None`

---

## Timeline Retrieval & Refinement

At each turn during Phase 1:

```
query = current user utterance only  (focused embedding; not accumulated dialogue)
    │
    ▼
MemoryGraph.retrieve(query, k=RETRIEVE_TOP_K)
    └─ cosine similarity → top-k MemoryNodes, sorted by conv_id desc
    │
    ▼
for each retrieved node:
    get_all_path(node_id)
        ├─ build memory_past / memory_future adjacency from node.links
        ├─ BFS backwards  → past_paths  (older → node)
        ├─ BFS forwards   → future_paths (node → newer)
        └─ combine: past_path[:-1] + future_path
    │
    └─ sample 1 unique path (not already used this turn) → use_timeline
    │
    └─ write retrieval log (seed + path_linked counts, timeline info)
```

Phase 1 ends here — no LLM call per turn. The path traversal result is only
used for retrieval logging; no refinement is run in Phase 1.

**Phase 2** runs the same retrieval then continues:

```
for each path in use_timeline:
    get_path_text(path)
        └─ "[fact] - (relation) - [fact] - ..."
    refine_timeline(path_text, current_dialogue)
        └─ 1 LLM call (guided_json) → {"refined_text": "natural-language context"}
    │
    ▼
Generator.generate_qa_answer(question, refined_texts, subset, current_dialogue)
    └─ 1 LLM call (guided_json)
```

**Path format** (alternating): `(node_id, relation, node_id, relation, ..., node_id)`

**Query design:** Only the current user utterance is used as the retrieval query
(not the accumulated dialogue). This produces a more focused embedding that
improves retrieval relevance over long sessions.

**TIMELINE_SAMPLE_N = 1:** One unique path sampled per retrieved node, so at
most `RETRIEVE_TOP_K` paths are refined and passed to the generator per turn.

---

## Experiment Protocol

Processes **individual sessions** following `global_readme.md` (QA-only variant).

### Phase 1 — Memory Construction (no LLM calls per turn)

```
current_dialogue = ""   ← GT turn accumulator; resets at each day boundary
current_day      = -1   ← tracks the current virtual day index
finalize_turns   = []   ← GT turns for the next finalize batch (reset after finalize)

for each (user_turn, assistant_turn) in session:

    new_day = conv_id // CONV_IDS_PER_DAY
    if new_day != current_day:
        current_dialogue = ""              ← daily reset
        current_day = new_day

    if conv_id changed:
        pending_count += 1
        if pending_count >= FINALIZE_EVERY_N_CONVS:   # = CONV_IDS_PER_DAY (daily)
            finalize_conv(prev_conv_id, ...)   ← memory construction (LLM calls)
            reset finalize_turns, pending_count only

    query = "User: {user_utterance}"           ← current turn only

    1. retrieve_for_response(query)               ← cosine retrieval + path traversal
    2. write retrieval log
    3. current_dialogue += "User: ...\nAssistant: {GT}\n"
    4. finalize_turns   += [user_turn, assistant_turn]

after all turns:
    finalize_conv(last_conv_id, ...)           ← memory construction for remaining turns
    final_dialogue = current_dialogue          ← snapshot for Phase 2
```

- **GT agent responses are always used** — no response is generated; GT responses
  are used for `current_dialogue` and `finalize_turns`.
- **`current_dialogue` resets at each day boundary** (`conv_id // CONV_IDS_PER_DAY`
  changes). Only the dialogue from the current virtual day is passed as LLM
  context, preventing context overflow over long sessions.
- **`FINALIZE_EVERY_N_CONVS = CONV_IDS_PER_DAY`** so memory finalization and
  dialogue reset are always aligned to the same daily boundary.
- **QA exchanges are never stored** in memory or included in dialogue history.

### Phase 2 — QA Answering (memory frozen)

No new memory is constructed during this phase. QA uses the **identical
timeline retrieval and refinement pipeline as Phase 1** — cosine retrieval,
BFS path traversal, LLM refinement, and generation. The only differences are:

- The retrieval query is the question string (not the user utterance).
- `current_dialogue` is `final_dialogue + "User: {question}\n"` — the full
  session dialogue with the question appended as the last User turn.
- All QA questions share the same `final_dialogue` base, making each question
  **independent** — answering Q3 uses the same context as answering Q1.
- Memory is frozen: no strength updates, no new nodes written.

```
final_dialogue = full session GT dialogue (from Phase 1)

for each QA pair:
    qa_dialogue = final_dialogue + "User: {question}\n"

    1. retrieve_for_response(question)     ← cosine retrieval + path traversal (same as Phase 1)
    2. answer_qa(question, timelines, subset, qa_dialogue)
           └─ refine_all(use_timelines, qa_dialogue, memory_graph)  ← timeline refinement
           └─ generate_qa_answer(question, refined_texts, subset, qa_dialogue)  ← 1 LLM call
    3. write retrieval log (includes timeline info, path counts, dialogue turns)
```

Refinement tokens from Phase 2 are accumulated in `other_tokens` (same bucket
as memory construction tokens).

### Phase 3 — Cleanup

Save memory snapshot → aggregate token statistics → save results (atomic write)
→ update checkpoint → clear memory.

**Error handling:** Any unhandled exception during a session propagates
immediately (no silent skip). `KeyboardInterrupt` returns exit code 130.
Since checkpointing is enabled by default, a failed run can be resumed from the
last successfully completed session after fixing the root cause.

---

## Token Tracking

All LLM calls use `return_usage=True`. Token counts are split into three
categories per session:

| Category | Description |
|---|---|
| `qa_tokens` | `generate_qa_answer` (1 call per QA pair) |
| `other_tokens` | `_summarize_conv` (1 per finalize batch) + `_extract_relation` (up to j per new node) + `refine_timeline` (up to k per QA in Phase 2) |

`total_input` = `qa_input` + `other_input`.
`total_output` = `qa_output` + `other_output`.
`num_total_api_calls` counts all actual API calls. Phase 1 has no LLM calls per turn.

---

## LLM Call Logging

When `ENABLE_LLM_CALL_LOGGING = True` (default), every successful LLM call is
appended to a per-session JSONL file, grouped by call type.

**Call types:**

| Directory | Call type | Source | Frequency |
|---|---|---|---|
| `call_2_refinement/` | Timeline refinement | `TimelineRetriever.refine_timeline()` | Up to k per QA (Phase 2 only) |
| `call_3_summarization/` | Conv summarization | `MemoryGraph._summarize_conv()` | 1 per daily finalize |
| `call_4_relation/` | Relation extraction | `MemoryGraph._extract_relation()` | Up to j per new node |
| `call_5_qa/` | QA answering | `Generator.generate_qa_answer()` | 1 per QA pair (Phase 2) |

**Entry schema:**

```json
{
  "timestamp":     "2026-03-25T11:23:45.123456",
  "call_type":     "call_3_summarization",
  "system_prompt": "",
  "user_prompt":   "(full prompt string sent to LLM)",
  "output":        {"sentences": ["fact 1", "fact 2"]}
}
```

- `system_prompt` is always `""` (theanine uses a single unified prompt string)
- `output` excludes the `_usage` field; token counts are tracked separately
- Only the final successful call is logged per operation (retries are not recorded)
- `LLMCallLogger` is defined in `theanine_module.py` and injected into all
  sub-components per session via `TheanineModule.set_llm_logger()`

---

## Configuration (`config_0.py`)

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence embedding model for retrieval and linking |
| `LINKING_TOP_J` | `3` | Top-j similar past nodes for relation extraction per new node |
| `RETRIEVE_TOP_K` | `3` | Top-k nodes retrieved per turn and per QA question |
| `TIMELINE_SAMPLE_N` | `1` | Paths sampled per retrieved node (always 1 per paper) |
| `TEMPERATURE` | `0.7` | LLM generation temperature |
| `MAX_TOKENS` | `750` | Max output tokens (covers summarization + relation extraction) |
| `JSON_RETRY` | `3` | Retry count on JSON parse failure |
| `TIMING_CONV_ID` | `0` | Unused in this variant (Phase 1 has no LLM call, no timing) |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint after each session |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save memory graph snapshots after each session |
| `LLM_ENGINE` | `"vllm"` | Engine: `"vllm"`, `"together"`, or `"openai"` |
| `CONV_IDS_PER_DAY` | `2` | Number of conv_ids per virtual day; controls `current_dialogue` reset boundary |
| `MINUTES_PER_TURN` | `10` | Minutes per local `turn_id` within a conv (virtual time model) |
| `FINALIZE_EVERY_N_CONVS` | `= CONV_IDS_PER_DAY` | Always equal to `CONV_IDS_PER_DAY`; memory finalization happens once per day |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Log all LLM call prompts and outputs to `prompt_log/` |

Dataset paths (auto-selected by `--subset`):

```
dataset/implexconv/ImplexConv_opposed_processed.json
dataset/implexconv/ImplexConv_supportive_processed.json
```

---

## Usage

### Basic run

```bash
python theanine/theanine_sequential/run_experiment.py \
    --start-session 0 --end-session 4 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --config config_0
```

### With max_model_len override

```bash
python theanine/theanine_sequential/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
    --max-model-len 8000 \
    --config config_0
```

### Background run with nohup

```bash
CUDA_VISIBLE_DEVICES=0 nohup python theanine/theanine_sequential/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 --gpu-memory 0.5 \
    --max-model-len 8000 \
    --config config_0 \
    > nohup/nohup_opp_8b_session_0_29.out 2>&1 &
```

### Merge results from multiple runs

```bash
python theanine/theanine_sequential/merge_results.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --subset opposed \
    --config config_0

# Dry run (show what would be merged)
python theanine/theanine_sequential/merge_results.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --subset opposed \
    --config config_0 \
    --dry-run
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start-session` | int | **required** | First session to process (inclusive) |
| `--end-session` | int | **required** | Last session to process (inclusive) |
| `--subset` | str | **required** | `opposed` or `supportive` dataset subset |
| `--model` | str | `meta-llama/Llama-3.1-8B-Instruct` | Model path (HF format or local) |
| `--tensor-parallel` | int | `1` | Tensor parallelism degree (vLLM only) |
| `--gpu-memory` | float | `0.5` | GPU memory utilization fraction (0.0–1.0) |
| `--max-model-len` | int | `None` | Override vLLM max_model_len |
| `--config` | str | `config_0` | Config file name (without `.py`) in `theanine/` |

`--config` is **dynamically loaded at runtime** — copy `config_0.py` freely
(e.g. `config_k5.py`, `config_large.py`) and pass via `--config`. Both
`run_experiment.py` and `merge_results.py` accept `--config` so paths stay
consistent. The config name is used as the output directory prefix:
`{config}_outputs_{model}_{subset}/`.

---

## Output Structure

```
theanine/
├── nohup/                                   ← background run logs
├── {config}_outputs_{model}_{subset}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{subset}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{id}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{id}/
│   │   │       ├── nodes.json          ← node metadata + links
│   │   │       ├── embeddings.npy      ← embedding matrix
│   │   │       └── embedding_ids.json  ← node_id order
│   │   ├── prompt_log/
│   │   │   └── session_{id}/
│   │   │       ├── call_2_refinement/calls.jsonl
│   │   │       ├── call_3_summarization/calls.jsonl
│   │   │       ├── call_4_relation/calls.jsonl
│   │   │       └── call_5_qa/calls.jsonl
│   │   └── logs/
│   └── results_{model}_{subset}_merged.json
└── logs/
```

### `results_*.json`

List of session result objects:

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
          {"session_id": 0, "conv_id": 1, "turn_id": -1}
        ],
        "qa_tokens": {"input": 280, "output": 18, "model": "..."}
      }
    ],

    "timing_statistics": {
      "phase2_qa_avg": {"retrieval_time": 0.01, "inference_time": 0.35, "total_time": 0.36}
    },

    "token_statistics": {
      "total_qa_input": 3500,
      "total_qa_output": 600,
      "total_input": 24000,
      "total_output": 5000,
      "num_qa_api_calls": 5,
      "num_total_api_calls": 770
    },

    "memory_snapshot_path": "memory_snapshots/session_0/"
  }
]
```

**Notes:**
- `retrieved_memories[*].turn_id` is always `-1`: Theanine memory nodes are
  created at `conv_id` granularity, not individual turn granularity.
- `generated_answer` for `supportive` subset is one of `"yes"`, `"no"`.
  For `opposed`, it is free-form text.
- `total_input` = QA input + **all Theanine internal API calls**
  (summarization, relation extraction, timeline refinement in Phase 2).
- `num_total_api_calls` counts only actual API calls. Expected breakdown per
  session (rough estimate with default config):
  - Refinement calls: up to k × num_qa (Phase 2 only)
  - Summarization calls: 1 × num_finalize_batches
  - Relation extraction calls: up to j × nodes_per_batch × num_finalize_batches
  - QA calls: 1 × num_qa

### `retrieval_logs/session_{id}_retrieval_log.jsonl`

One JSON object per line, one entry per turn (Phase 1) and per QA question (Phase 2).
Each entry includes the query, retrieved node IDs with cosine similarity scores,
and timeline path metadata. Full prompt strings are logged separately in
`prompt_log/` and are not duplicated here.

Each entry includes parallel `memory_type` and `num_retrieved` arrays.
Both phases use the same structure:

**Phase 1 (`memory_construction`) and Phase 2 (`qa`)**:
```json
{
  "phase": "memory_construction",
  "memory_type":   ["seed", "path_linked", "current_dialogue"],
  "num_retrieved": [3, 5, 42]
}
```

| Field | Description |
|---|---|
| `seed` | Nodes directly retrieved by cosine similarity (`RETRIEVE_TOP_K`) |
| `path_linked` | Additional nodes reached via BFS timeline traversal (appear in sampled paths but not directly retrieved); `seed + path_linked` = total unique nodes across all used timeline paths |
| `current_dialogue` | Number of non-empty lines in the current dialogue string passed to the prompt |

`retrieved_items` contains the seed-node details (content preview, cosine score, source turn).
Path-linked nodes are embedded in the refined timeline texts passed to the Generator.

`module_specific` fields:
- `num_paths_used`: number of timeline paths refined and passed to Generator
- `use_timelines`: sampled path tuples (node_id, relation, ...) as lists
- `timeline_info`: all retrieved nodes with their full path sets

---

## Citation

```bibtex
@inproceedings{oh2025theanine,
  title={Towards Lifelong Dialogue Agents via Timeline-based Memory Management},
  author={Oh, Kai Tzu-iunn and others},
  booktitle={NAACL},
  year={2025}
}
```
