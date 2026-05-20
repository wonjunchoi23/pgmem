# A-MEM Memory Module — PrefEval Port

A-MEM (Xu et al., 2025) memory module adapted for the PrefEval implicit-persona
**chained-cumulative** experiment. See `../experiment_prefeval.md` for the
shared experiment spec and `../difference.md` for the per-aspect comparison
with the ImplexConv variant.

---

## Files

| File | Role |
|---|---|
| `memory_layer.py`     | Agentic memory system, note construction, evolution, retrieval. **Verbatim copy** from the ImplexConv variant |
| `llm_text_parsers.py` | Parser helpers used by `memory_layer.py`. **Verbatim copy** |
| `agent.py`            | A-MEM experiment wrapper. Trimmed: no subset branching, QA cap raised to 200 words. Adds `load_memory_snapshot` |
| `load_dataset.py`     | PrefEval loader. One sample → one `Session`; turns reordered to `user → assistant` |
| `run_experiment.py`   | Chained-cumulative runner with resume |
| `config_0.py`         | Hyperparameters, output paths, batch settings |

---

## Experiment flow per invocation

`samples[0..K]` is fixed (no sampling). For each `k = 0..K` (incremental):

1. **Ingest** session `k` turn-by-turn via `agent.add_memory(...)` — the
   sequential A-MEM path (analyze → optionally evolve → store). Strictly
   sequential within a session to preserve evolution semantics.
2. **QA** — at checkpoint `k`, retrieve from the current memory state and
   batch-generate answers for `q_0..q_k` together (chunked by
   `QA_BATCH_SIZE`). Each row is appended to `results.jsonl`; retrieval logs
   are appended to `retrieval_log.jsonl`.
3. **Snapshot** — save `memory_snapshots/m_<k>/` only **after** QA succeeds.
   Snapshot existence ⇔ QA at that checkpoint completed.
4. **Stats** — update cumulative `stats.json`.

---

## Resume behavior

On startup the runner scans `memory_snapshots/m_*/`:

- No snapshots → start fresh from `k = 0`.
- `K_target ≤ k_existing` → no-op, log "already complete".
- `K_target > k_existing` → load `m_{k_existing}` and continue from
  `k_existing + 1`. New rows append to existing JSONLs; cumulative
  `stats.json` adds on top of prior totals.

Run `--end-session 10` first, then `--end-session 90` later — the second
invocation will load `m_10`, ingest sessions 11..90, and run QA at every new
checkpoint.

Resume only works if config (model, retrieve_k, evolution_threshold, prompt
templates, etc.) is unchanged between runs. Different models naturally write
to different output dirs because the dir name embeds the model.

---

## Key config values

From `config_0.py`:

| Parameter | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL`     | `"all-MiniLM-L6-v2"` | |
| `RETRIEVE_K`          | 5  | |
| `EVOLUTION_THRESHOLD` | 100 | |
| `QA_BATCH_SIZE`       | 64 | Chunk size for batched QA at a checkpoint |
| `CONV_IDS_PER_DAY`    | 2  | Same as ImplexConv |
| `MINUTES_PER_TURN`    | 10 | Same as ImplexConv |
| `MAX_TOKENS`          | **1500** | Raised from 750 — PrefEval answers are advisory |
| QA word cap           | **200 words** | In `agent.QA_PROMPT` |

Dataset path: `dataset/implicit_persona.json` (single file; no subset split).

---

## Usage

Run from the `exp_prefeval/amem/` directory.

```bash
python run_experiment.py \
    --end-session 10 \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --config config_0
```

With `max_model_len` override:

```bash
python run_experiment.py \
    --end-session 10 \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.9 \
    --max-model-len 8000 \
    --config config_0
```

Extending an existing run (resume):

```bash
# First run
python run_experiment.py --end-session 10 --model <M> ...

# Later — picks up from m_10, ingests 11..90, appends new rows
python run_experiment.py --end-session 90 --model <M> ...
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--end-session`      | int   | required        | Last sample index in chain (chain = `samples[0..K]`) |
| `--model`            | str   | config default  | Model path |
| `--tensor-parallel`  | int   | config default  | Tensor parallelism |
| `--gpu-memory`       | float | config default  | GPU memory utilization fraction |
| `--max-model-len`    | int   | `None`          | Optional vLLM max model length |
| `--config`           | str   | `config_0`      | Config filename without `.py` |

---

## Output structure

```
exp_prefeval/amem/
├── config_0_outputs_<model>/
│   ├── results.jsonl              # one row per (k, question_session)
│   ├── retrieval_log.jsonl        # one row per QA-time retrieval
│   ├── stats.json                 # cumulative across all runs
│   ├── meta.json                  # config_metadata, written once
│   ├── memory_snapshots/
│   │   ├── m_0/{memories.json, retriever.json, embeddings.npy, metadata.json}
│   │   ├── m_1/...
│   │   └── m_K/...
│   ├── prompt_log/
│   │   ├── call_2_note_construction/calls.jsonl
│   │   ├── call_3_evolution/calls.jsonl
│   │   └── call_4_qa/calls.jsonl
│   └── logs/
└── logs/
```

### `results.jsonl` row schema

```json
{
  "k":                  3,
  "question_session":   1,
  "question":           "...",
  "model_answer":       "...",
  "topic":              "education_...",
  "persona":            "a retired ...",
  "preference":         "...",
  "explanation":        "...",
  "retrieved_memories": [{"session_id": 0, "conv_id": 1, "turn_id": 4}, ...],
  "qa_tokens":          {"input": 0, "output": 0, "model": "..."}
}
```

Persona / preference / explanation are ground truth for downstream judging
only — they are not in the model's prompt.

### `stats.json` schema

```json
{
  "call_2_note_construction": {"input": 0, "output": 0, "llm_calls": 0,
                               "parse_fallback_count": 0},
  "call_3_evolution":         {"input": 0, "output": 0, "llm_calls": 0},
  "call_4_qa":                {"input": 0, "output": 0, "llm_calls": 0},
  "evolution": {
    "evo_triggered_count": 0,
    "actions_taken": {"strengthen": 0, "update_neighbor": 0}
  },
  "checkpoints_completed": [0, 1, 2, ...]
}
```

---

## Notes

- The `Turn` dataclass populates `global_turn_id` cumulatively across the
  whole chain; A-MEM does not consume this field but other modules
  (e.g. Theanine) do.
- Memory item timestamps use `"0000_{conv_id:04d}_{turn_id:04d}"`. The leading
  `0000` is a fixed slot so `format_virtual_time(...)`'s 3-part parser keeps
  working unchanged.
- Snapshot is saved after QA. If the process crashes mid-QA at checkpoint k,
  resume re-ingests session k from m_{k-1}.
