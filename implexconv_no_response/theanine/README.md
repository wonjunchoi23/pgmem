# Theanine — Timeline-based Memory Management (Batched Main Variant)

Theanine (Oh et al., NAACL 2025) memory module for the ImplexConv experiment,
adapted to run **multiple sessions in parallel** on a single GPU by batching
same-type LLM calls together.

`theanine/` is now the main batched implementation. The underlying memory logic
is kept the same:

- memory is still constructed from completed `conv_id` dialogues
- retrieval is still cosine search + timeline path traversal
- QA still uses timeline refinement followed by answer generation
- GT assistant turns are still used in Phase 1

The main difference is execution strategy: summarization, relation extraction,
timeline refinement, and QA generation are grouped across sessions and sent to
vLLM with `generate_batch_raw()`.

---

## Architecture

Three cooperating components, same as the original Theanine:

- **`MemoryGraph`** (`memory_graph.py`): Summarizes completed conv-level
  dialogues into fact sentences, embeds them, and links nodes across convs via
  ATOMIC-style relation extraction.
- **`TimelineRetriever`** (`timeline.py`): Retrieves top-k nodes, builds all
  timeline paths around each node, samples one unique path per retrieved node,
  and refines the paths into natural-language context.
- **`Generator`** (`generator.py`): Generates QA answers using refined timeline
  context.

The archived sequential implementation is kept under `theanine/theanine_sequential/`.
The main `theanine/` implementation adds batch-friendly step-wise helpers around these same
components so that the runner can collect prompts from multiple sessions before
calling the model.

---

## Files

| File | Role |
|---|---|
| `memory_graph.py` | `MemoryGraph` — node creation, embedding, relation linking, batch-friendly summarize/relation helpers |
| `timeline.py` | `TimelineRetriever` — path traversal, timeline sampling, batch-friendly refinement helpers |
| `generator.py` | `Generator` — QA generation plus prompt-build / parse helpers for batch execution |
| `theanine_module.py` | `TheanineModule` — unified entry point wrapping all three components |
| `run_experiment.py` | Batch experiment loop (`BatchedTheanineRunner`) |
| `config_0.py` | All hyperparameters, output path helpers, and batch sizes |
| `load_dataset.py` | ImplexConv dataset loader (`Session`, `Turn`, `QAPair` dataclasses) |

---

## Memory Graph Construction

Memory is **not** built per turn. Instead, the full dialogue of a completed
`conv_id` is summarized into fact sentences, each becoming one `MemoryNode`.
Nodes are linked across older convs by relation extraction.

In the sequential version, one session does:

```text
finalize_conv(conv dialogue)
  -> summarize dialogue
  -> create nodes
  -> embed nodes
  -> relation extraction against top-j past nodes
  -> register nodes
```

In the batch version, the per-session semantics are the same, but the runner
does this:

```text
collect finalize jobs from many sessions
  -> batch summarize all pending conv dialogues
  -> prepare new nodes per session
  -> batch relation extraction prompts across sessions
  -> apply links back to each session's own graph
```

This preserves the original behavior that:

- nodes from the same finalize batch do not link to each other
- only older committed nodes are used as relation candidates
- each session's memory graph remains fully independent

**Node key format:** `c{conv_id}-m{idx}` (e.g. `c2-m3`)

---

## Timeline Retrieval & Refinement

There is still **no retrieval or LLM call per turn** in Phase 1.

During Phase 2 QA:

```text
question
  -> retrieve timeline paths
  -> sample up to `TIMELINE_SAMPLE_N` unique paths per retrieved node
  -> refine sampled paths using current dialogue + separate question
  -> answer QA from refined memories using current dialogue + separate question
```

In `theanine/`, all refinement prompts across the current batch are
grouped together before generation, then all QA prompts are grouped together.

---

## Experiment Protocol

Processes **multiple sessions in parallel** following the same QA-only protocol
as `theanine/`.

### Phase 1 — Memory Construction

For each session:

- `current_dialogue` still resets at each virtual day boundary
- `finalize_turns` still accumulates GT turns until the finalize boundary
- no per-turn retrieval is executed or logged
- memory is still finalized only at daily / conv boundaries

Batch behavior:

1. Iterate turn-by-turn across all active sessions.
2. Detect which sessions reached a finalize boundary.
3. Batch all pending summarization prompts.
4. Batch all pending relation-extraction prompts.
5. Commit each session's nodes to its own graph.

### Phase 2 — QA Answering

For each QA item:

1. Retrieve timeline paths using the question as the query.
2. Sample up to `TIMELINE_SAMPLE_N` unique paths per retrieved node.
3. Build refinement prompts for sampled paths.
4. Batch all refinement prompts.
5. Build QA prompts from refined texts.
6. Batch all QA prompts.
7. Redistribute outputs back to the correct session.

Memory is frozen during QA, exactly as in the original version.

### Phase 3 — Cleanup

Save memory snapshot -> aggregate token statistics -> save results -> update
checkpoint -> clear memory.

---

## LLM Call Logging

When `ENABLE_LLM_CALL_LOGGING = True` (default), every successful LLM call is
appended to a per-session JSONL file, grouped by call type.

**Call types:**

| Directory | Call type | Source | Frequency |
|---|---|---|---|
| `call_2_refinement/` | Timeline refinement | `TimelineRetriever` | Up to `RETRIEVE_TOP_K × TIMELINE_SAMPLE_N` per QA (batched across sessions) |
| `call_3_summarization/` | Conv summarization | `MemoryGraph` | 1 per finalize job (batched across sessions) |
| `call_4_relation/` | Relation extraction | `MemoryGraph` | Up to j per new node (batched across sessions) |
| `call_5_qa/` | QA answering | `Generator` | 1 per QA pair (batched across sessions) |

**Entry schema:**

```json
{
  "timestamp": "2026-04-05T11:23:45.123456",
  "call_type": "call_3_summarization",
  "system_prompt": "",
  "user_prompt": "(full prompt string sent to LLM)",
  "output": {"sentences": ["fact 1", "fact 2"]}
}
```

Notes:

- `system_prompt` is always `""`
- `_usage` is not stored in prompt logs
- only the final successful call is logged
- fallback sequential retries can also appear when batch JSON parsing fails

---

## Configuration (`config_0.py`)

Original Theanine parameters are still used, plus batching-related sizes.

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence embedding model for retrieval and linking |
| `LINKING_TOP_J` | `3` | Top-j similar past nodes for relation extraction per new node |
| `RETRIEVE_TOP_K` | `5` | Top-k nodes retrieved per QA question |
| `TIMELINE_SAMPLE_N` | `1` | Max unique timeline paths sampled per retrieved node |
| `TEMPERATURE` | `0.7` | LLM generation temperature |
| `MAX_TOKENS` | `2048` | Max output tokens for relation/refinement/QA |
| `SUMMARIZE_MAX_TOKENS` | `1500` | Max output tokens for summarization |
| `JSON_RETRY` | `3` | Retry count on JSON parse failure |
| `ENABLE_CHECKPOINTING` | `True` | Save checkpoint after each completed session |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | Save memory graph snapshots after each session |
| `LLM_ENGINE` | `"vllm"` | Engine: `"vllm"`, `"together"`, or `"openai"` |
| `CONV_IDS_PER_DAY` | `2` | Number of conv_ids per virtual day |
| `MINUTES_PER_TURN` | `10` | Minutes per local `turn_id` |
| `FINALIZE_EVERY_N_CONVS` | `= CONV_IDS_PER_DAY` | Memory finalization boundary |
| `ENABLE_LLM_CALL_LOGGING` | `True` | Log prompts/outputs to `prompt_log/` |
| `BATCH_SIZE` | `4` | Sessions processed together |
| `SUMMARIZE_BATCH_SIZE` | `32` | Max summarize prompts per raw batch call |
| `RELATION_BATCH_SIZE` | `64` | Max relation prompts per raw batch call |
| `REFINE_BATCH_SIZE` | `64` | Max refinement prompts per raw batch call |
| `QA_BATCH_SIZE` | `64` | Max QA prompts per raw batch call |

Dataset paths:

```text
dataset/implexconv/ImplexConv_opposed_processed.json
dataset/implexconv/ImplexConv_supportive_processed.json
```

---

## Usage

### Basic run

```bash
python theanine/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config_0
```

### With `max_model_len` override

```bash
python theanine/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config_0
```

### Supportive subset

```bash
python theanine/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset supportive \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --batch-size 4 \
    --config config_0
```

### Background run with `nohup`

```bash
CUDA_VISIBLE_DEVICES=0 nohup python theanine/run_experiment.py \
    --start-session 0 --end-session 99 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --batch-size 4 \
    --config config_0 \
    > nohup/nohup_opp_session_0_99.out 2>&1 &
```

### Merge results from multiple runs

Use the root-level `merge_results.py`.

```bash
python merge_results.py theanine config_0_outputs_Qwen3-1.7B_opposed
```

Dry run:

```bash
python merge_results.py \
    theanine \
    config_0_outputs_Qwen3-1.7B_opposed \
    --dry-run
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start-session` | int | **required** | First session to process (inclusive) |
| `--end-session` | int | **required** | Last session to process (inclusive) |
| `--subset` | str | **required** | `opposed` or `supportive` |
| `--model` | str | config default | Model path |
| `--tensor-parallel` | int | `1` | Tensor parallelism degree |
| `--gpu-memory` | float | config default | GPU memory utilization fraction |
| `--max-model-len` | int | `None` | Override vLLM `max_model_len` |
| `--batch-size` | int | `BATCH_SIZE` | Sessions processed together |
| `--config` | str | `config_0` | Config file name without `.py` |

`--config` is dynamically loaded at runtime. The config name is also used as the
output directory prefix:

```text
{config}_outputs_{model}_{subset}/
```

---

## Checkpointing

Archived sequential Theanine used:

```json
{"last_completed_session_index": 59}
```

Current main `theanine/` uses:

```json
{"completed_session_ids": [0, 1, 2, 3, 4]}
```

Why this changed:

- batched runs can complete sessions out of order relative to a simple index
- saving a set is safer when resuming after a partial batch

Old checkpoints are still readable and auto-converted on load.

---

## Output Structure

```text
theanine/
├── nohup/
├── {config}_outputs_{model}_{subset}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{subset}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{id}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{id}/
│   │   │       ├── nodes.json
│   │   │       ├── embeddings.npy
│   │   │       └── embedding_ids.json
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

---

## Result Format

`results_*.json` is still a list of per-session result objects:

```json
[
  {
    "session_id": 0,
    "config_metadata": {
      "max_tokens": 2048,
      "retrieve_top_k": 5
    },
    "qa_results": [
      {
        "question": "...",
        "generated_answer": "...",
        "ground_truth_answer": "...",
        "retrieved_memories": [
          {"session_id": 0, "conv_id": 1, "turn_id": 2}
        ],
        "qa_tokens": {"input": 280, "output": 18, "model": "..."}
      }
    ],
    "token_statistics": {
      "call_3_summarization": {
        "input": 8200,
        "output": 1400,
        "llm_calls": 8,
        "parse_fallback_count": 0
      },
      "call_4_relation": {
        "input": 9600,
        "output": 700,
        "llm_calls": 24
      },
      "call_2_refinement": {
        "input": 2100,
        "output": 260,
        "llm_calls": 6
      },
      "call_5_qa": {
        "input": 1400,
        "output": 90,
        "llm_calls": 3
      },
      "qa_input": 3500,
      "total_input": 21300,
      "total_output": 2450,
      "total_llm_calls": 41
    },
    "memory_snapshot_path": "memory_snapshots/session_0/"
  }
]
```

Notes:

- `qa_tokens` inside each QA item record only the final QA call tokens
- `qa_input` inside `token_statistics` includes the full Phase 2 input budget:
  refinement prompt tokens + final QA prompt tokens
- `total_input` / `total_output` include all tracked LLM calls from memory
  construction, refinement, and QA
- `total_llm_calls` includes summarize, relation, refine, and QA calls
- `retrieved_memories` keeps the same output structure as the original runner

---

## Retrieval Logs

`retrieval_logs/session_{id}_retrieval_log.jsonl` now records only Phase 2 QA
retrievals, while keeping the same entry shape as before with the module name
changed to `theanine`.

Each entry includes:

- direct retrieved seed nodes
- additional path-linked nodes implied by sampled timelines
- number of current-dialogue turns (question excluded)
- sampled timelines and full timeline metadata

---

## Verification

Static verification completed:

```bash
python -m py_compile \
    theanine/config_0.py \
    theanine/generator.py \
    theanine/memory_graph.py \
    theanine/theanine_module.py \
    theanine/timeline.py \
    theanine/run_experiment.py
```

CLI parsing check also completed:

```bash
python theanine/run_experiment.py --help
```

End-to-end model execution has not yet been run as part of this README update.

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
