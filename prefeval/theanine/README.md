# Theanine Memory Module — PrefEval Port

Theanine (Oh et al., NAACL 2025) memory module adapted for the PrefEval
implicit-persona **chained-cumulative** experiment. See
`../experiment_prefeval.md` for the shared experiment spec and
`../difference.md` for the per-aspect comparison with the ImplexConv variant.

---

## Files

| File | Role |
|---|---|
| `memory_graph.py`     | MemoryGraph: summarization, node creation, embeddings, relation extraction. **Verbatim copy** from the ImplexConv variant |
| `timeline.py`         | TimelineRetriever: path traversal, refinement. **Verbatim copy** |
| `generator.py`        | QA generator. Trimmed: no subset branching, QA cap 200 words |
| `theanine_module.py`  | TheanineModule wrapper. Trimmed: no subset, adds `load_memory_snapshot` |
| `load_dataset.py`     | PrefEval loader (shared shape with `../amem/load_dataset.py`) |
| `run_experiment.py`   | Chained-cumulative runner with resume |
| `config_0.py`         | Hyperparameters, output paths, batch settings |

---

## Experiment flow per checkpoint k

`samples[0..K]` is fixed (no sampling). For each `k = 0..K`:

1. **Boundary detection** — if entering a new conv (`k > 0`), increment
   `pending_count`. If `pending_count >= FINALIZE_EVERY_N_CONVS` AND
   `finalize_turns` non-empty: trigger **natural finalize** for the
   accumulated turns (1 summarize call + N×LINKING_TOP_J relation calls).
   Reset `pending_count` and `finalize_turns`.
2. **Day boundary** — if `k // CONV_IDS_PER_DAY` differs from `current_day`,
   reset `current_dialogue`.
3. **Ingest session k** — append turns to `current_dialogue` and
   `finalize_turns`. **No LLM calls during ingestion** (theanine doesn't
   embed per-turn).
4. **NO force flush at checkpoint** (per-design choice — see "Policy" below).
5. **QA over q_0..q_k** — pipeline:
   - retrieve timeline paths for each question
   - batch all refinement prompts together (chunks of `REFINE_BATCH_SIZE`)
   - batch all QA prompts together (chunks of `QA_BATCH_SIZE`)
6. **Snapshot `m_k`** — saved after QA succeeds. Snapshot existence ⇔ QA done.
7. **Stats** — update cumulative `stats.json`.

### Policy: natural-boundary finalize, no force flush

With `FINALIZE_EVERY_N_CONVS = CONV_IDS_PER_DAY = 2`:

| k | graph contents at QA time | `current_dialogue` |
|---|---|---|
| 0 | empty (no boundary crossed yet) | session 0 |
| 1 | empty (pending=1 < 2) | sessions 0+1 |
| 2 | nodes from sessions 0+1 batch | session 2 |
| 3 | nodes from sessions 0+1 batch | sessions 2+3 |
| 4 | nodes from 0+1, 2+3 | session 4 |
| 5 | nodes from 0+1, 2+3 | sessions 4+5 |

The model can still answer for not-yet-finalized sessions because their
content is visible through `current_dialogue`. Sessions only become
retrievable from the graph after a finalize boundary fires (entering the next
day).

This policy preserves the original ImplexConv adapter's natural finalization
semantics exactly. `current_dialogue` resets at virtual-day boundaries.

---

## Resume behavior

On startup the runner scans `memory_snapshots/m_*/`:

- No snapshots → start fresh from `k = 0`, empty runner state.
- `K_target ≤ k_existing` → no-op, log "already complete".
- `K_target > k_existing`:
  - `load_memory_snapshot(m_{k_existing})` restores the graph (nodes +
    embeddings).
  - Reconstruct runner-side state from `sessions[0..k_existing]`:
    - `current_conv_id = k_existing`
    - `current_day = k_existing // CONV_IDS_PER_DAY`
    - `current_dialogue` = sessions in current virtual day, formatted
    - `pending_count = k_existing - last_natural_finalize_position`
       (where `last_natural_finalize_position = (k_existing // FINALIZE_EVERY_N_CONVS) * FINALIZE_EVERY_N_CONVS`)
    - `finalize_turns` = turns of sessions in `[last_natural_finalize_position .. k_existing]`
  - Continue from `k_existing + 1`.

`memory_graph.load_snapshot` is faithful to the original. As in ImplexConv
theanine, `turn_id_start/end` and `global_turn_id_start/end` are not restored
on load (informational only, not used in retrieval).

---

## Key config values (`config_0.py`)

| Parameter | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL`        | `"all-MiniLM-L6-v2"` | |
| `LINKING_TOP_J`          | 3  | Relation candidates per new node |
| `RETRIEVE_TOP_K`         | 5  | Seed nodes per QA query |
| `TIMELINE_SAMPLE_N`      | 1  | Paths sampled per seed |
| `FINALIZE_EVERY_N_CONVS` | 2  | = `CONV_IDS_PER_DAY`. Natural finalize fires at boundary entering every Nth conv |
| `CONV_IDS_PER_DAY`       | 2  | |
| `MINUTES_PER_TURN`       | 10 | |
| `MAX_TOKENS`             | **2048** | Kept at theanine default — relation/refine/QA all share. QA word cap enforced via prompt template ("maximum 200 words") |
| `SUMMARIZE_MAX_TOKENS`   | 1500 | |
| `REFINE_BATCH_SIZE`      | 64 | Refinement prompts per batch |
| `QA_BATCH_SIZE`          | 64 | QA prompts per batch |
| `RELATION_BATCH_SIZE`    | 64 | Unused at chain runtime (single-agent finalize uses sequential path) |
| `SUMMARIZE_BATCH_SIZE`   | 32 | Same |

Dataset path: `dataset/implicit_persona.json` (single file; no subset split).

---

## Usage

Run from the `exp_prefeval/theanine/` directory.

```bash
python run_experiment.py \
    --end-session 10 \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
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
exp_prefeval/theanine/
├── config_0_outputs_<model>/
│   ├── results.jsonl          # one row per (k, question_session)
│   ├── retrieval_log.jsonl    # one row per QA-time retrieval
│   ├── stats.json             # cumulative across all runs
│   ├── meta.json              # config metadata, written once
│   ├── memory_snapshots/
│   │   ├── m_0/{nodes.json, embeddings.npy, embedding_ids.json}
│   │   ├── m_1/...
│   │   └── m_K/...
│   ├── prompt_log/
│   │   ├── call_2_refinement/calls.jsonl
│   │   ├── call_3_summarization/calls.jsonl
│   │   ├── call_4_relation/calls.jsonl
│   │   └── call_5_qa/calls.jsonl
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
    {"session_id": 0, "conv_id": 1, "turn_id": 0, "score": 0.81}
  ],
  "qa_tokens": {"input": 0, "output": 0, "model": "..."}
}
```

`retrieved_memories[].turn_id` is `turn_id_start` of the source conv for the
seed node. Empty `[]` at checkpoints where the graph has no relevant nodes.

### `stats.json` schema

```json
{
  "call_2_refinement":    {"input": 0, "output": 0, "llm_calls": 0},
  "call_3_summarization": {"input": 0, "output": 0, "llm_calls": 0,
                           "parse_fallback_count": 0},
  "call_4_relation":      {"input": 0, "output": 0, "llm_calls": 0},
  "call_5_qa":            {"input": 0, "output": 0, "llm_calls": 0},
  "checkpoints_completed": [0, 1, 2, ...]
}
```

---

## Notes on differences vs ImplexConv Theanine

| Aspect | ImplexConv | PrefEval port |
|---|---|---|
| QA subset | opposed/supportive | **opposed-style only**, 200-word cap |
| Cross-session batching | Phase 1 prompts batched across N parallel sessions | N/A (chain count = 1) |
| Memory between sessions | Cleared between sessions | **Persisted across whole chain** |
| Finalize granularity | Per-day (`FINALIZE_EVERY_N_CONVS = 2`) batches inside one session | **Same** — natural day-batch preserved across chain. Early checkpoints (k=0,1) have empty graph; QA falls back to `current_dialogue` |
| End-of-session flush | Phase 1 always ends with a final flush | **Removed** — would diverge resume state. Chain leaves trailing turns in `finalize_turns` until the next natural boundary |
| `current_dialogue` reset | At virtual-day boundary | **Same logic**, persists across the chain |

---

## Notes

- Snapshot saved after QA succeeds. Mid-QA crash → resume re-ingests sessions
  from `m_{k_existing}` onwards (cheap — ingestion has no LLM calls).
- The `Turn` dataclass populates `global_turn_id` cumulatively across the
  chain; theanine **does** consume it (memory graph node spans).
- `MAX_TOKENS=2048` for relation/refine/QA preserves theanine's multi-stage
  reasoning headroom. QA-answer length is governed by the prompt's
  "maximum 200 words" instruction.
- Memory_graph snapshot does not restore `turn_id_start/end` /
  `global_turn_id_start/end` (original ImplexConv behaviour kept; metadata
  fields not used in retrieval).
