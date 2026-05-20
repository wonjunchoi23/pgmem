# LD-Agent Memory Module — PrefEval Port

LD-Agent (Li et al., NAACL 2025) memory module adapted for the PrefEval
implicit-persona **chained-cumulative** experiment. See
`../experiment_prefeval.md` for the shared experiment spec and
`../difference.md` for the per-aspect comparison with the ImplexConv variant.

---

## Files

| File | Role |
|---|---|
| `event_memory.py`    | EventMemory: STM + LTM, forgetting (decay), boundary summarization. **Verbatim copy** from the ImplexConv variant |
| `personas.py`        | User/agent trait extraction. **Verbatim copy** |
| `generator.py`       | QA generator. Trimmed: no subset, no `response_build_json`, QA cap 200 words |
| `ldagent_module.py`  | LDAgentModule wrapper. Trimmed: no subset, adds `load_snapshot` |
| `load_dataset.py`    | PrefEval loader (shared shape with `../amem/load_dataset.py`) + `compute_virtual_seconds` / `convert_seconds_to_full_time` helpers |
| `run_experiment.py`  | Chained-cumulative runner (sequential turns + batched QA + resume) |
| `config_0.py`        | Hyperparameters, output paths |

---

## Experiment flow per checkpoint k

`samples[0..K]` is fixed (no sampling). For each `k = 0..K`:

1. **Sequential per-turn ingestion** of session k (Q2 = a):
   - For each `(user_turn, asst_turn)`:
     - `module.process_turn(...)` → `context_retrieve` (lazy boundary STM→LTM
       summarize, then STM clear if pending_count >= FINALIZE_EVERY_N_CONVS)
       → `_user_traits_update` → `_agent_traits_update` → `add_agent_response`
     - Drain `last_user_token_info`, `last_agent_token_info`,
       `last_summarize_token_info` after each turn.
   - **Per turn pair: 2 LLM calls** (call_2 + call_3); +1 (call_4) only at
     natural boundary.
2. **Batched QA over q_0..q_k** (Phase 2):
   - For each j ∈ [0, k]: `mb.relevance_retrieve(q_j)` → format memories.
   - Build (k+1) QA prompts. System prompt is identical across the batch
     (shared agent_traits at this checkpoint), so a single shared
     `system_prompt` is passed to `generate_batch_raw`.
   - Chunked by `QA_BATCH_SIZE`.
3. **Snapshot `m_k`** — saved after QA succeeds. Snapshot existence ⇔ QA done.
4. **Stats** — update cumulative `stats.json`.

### Policy: natural-boundary STM clear (Q1 = a)

Faithful to LD-Agent's original dual-memory design:

- STM accumulates turns until a natural boundary (`pending_conv_count >=
  FINALIZE_EVERY_N_CONVS = 2`) → `_context_summarize` (LLM) → write LTM →
  **STM cleared**.
- Phase 2 (QA at checkpoint k) consumes both STM (recent context) and LTM
  (older summarized batches).
- **No force flush at checkpoints** — STM is bounded, LTM grows with natural
  day batches.

| k | LTM contents | STM contents at QA time |
|---|---|---|
| 0 | empty (no boundary crossed) | session 0 turns |
| 1 | empty | sessions 0+1 turns |
| 2 | summary of sessions 0+1 (cleared at boundary) | session 2 turns |
| 3 | summary of sessions 0+1 | sessions 2+3 turns |
| 4 | sessions 0+1, 2+3 summaries | session 4 turns |

QA at every checkpoint can refer to BOTH STM (rich recent context) and LTM
(compressed older batches) plus the always-up-to-date persona traits.

---

## Resume behavior

`event_memory.save_snapshot` writes `short_term_memory.json`,
`long_term_memory.json`, `ltm_embeddings.npy`, `memory_state.json` (which
includes `last_conv_id`, `pending_conv_count`, `current_virtual_seconds`).
`personas.save_snapshot` writes `personas.json`. `load_snapshot` restores
**all** internal state needed to continue the chain — no extra runner-side
reconstruction is necessary.

On startup the runner scans `memory_snapshots/m_*/`:

- No snapshots → start fresh from `k = 0`.
- `K_target ≤ k_existing` → no-op, log "already complete".
- `K_target > k_existing` → `load_snapshot(m_{k_existing})`, continue from
  `k_existing + 1`.

---

## Key config values (`config_0.py`)

| Parameter | Default | Notes |
|---|---|---|
| `RELEVANCE_MEMORY_NUMBER` | 3  | LTM candidate pool size during turn processing |
| `RETRIEVE_K`              | 3  | LTM candidate pool size at QA |
| `DIST_THRESHOLD`          | 1.5 | L2 distance threshold (only with `ORI_MEM_QUERY=True`) |
| `DECAY_TEMP`              | 1e-4 | LTM time-decay coefficient |
| `CONV_IDS_PER_DAY`        | 2  | |
| `MINUTES_PER_TURN`        | 10 | |
| `FINALIZE_EVERY_N_CONVS`  | 2  | STM→LTM summarize boundary |
| `MAX_USER_PERSONAS`       | 10 | |
| `MAX_AGENT_PERSONAS`      | 10 | |
| `MAX_TOKENS`              | **1500** | Raised from 750 — PrefEval answers advisory |
| `JSON_RETRY`              | **10** | Raised from 3 |
| `QA_BATCH_SIZE`           | 64 | QA prompts per batch |
| QA word cap               | **200 words** | In `generator._build_qa_prompt` |

Dataset path: `dataset/implicit_persona.json` (single file; no subset split).

---

## Usage

```bash
python run_experiment.py \
    --end-session 10 \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.5 \
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
exp_prefeval/ldagent/
├── config_0_outputs_<model>/
│   ├── results.jsonl          # one row per (k, question_session)
│   ├── retrieval_log.jsonl    # one row per QA-time retrieval
│   ├── stats.json             # cumulative across all runs
│   ├── meta.json              # config metadata, written once
│   ├── memory_snapshots/
│   │   ├── m_0/{short_term_memory.json, long_term_memory.json, ltm_embeddings.npy, memory_state.json, personas.json}
│   │   ├── m_1/...
│   │   └── m_K/...
│   ├── prompt_log/
│   │   ├── call_2_user_persona/calls.jsonl
│   │   ├── call_3_agent_persona/calls.jsonl
│   │   ├── call_4_summarization/calls.jsonl
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
    {"session_id": 0, "conv_id": 1, "virtual_seconds": 7200.0, "score": 0.42}
  ],
  "qa_tokens": {"input": 0, "output": 0, "model": "..."}
}
```

### `stats.json` schema

```json
{
  "call_2_user_persona":  {"input": 0, "output": 0, "llm_calls": 0},
  "call_3_agent_persona": {"input": 0, "output": 0, "llm_calls": 0},
  "call_4_summarization": {"input": 0, "output": 0, "llm_calls": 0},
  "call_5_qa":            {"input": 0, "output": 0, "llm_calls": 0},
  "checkpoints_completed": [0, 1, 2, ...]
}
```

---

## Notes on differences vs ImplexConv LD-Agent

| Aspect | ImplexConv | PrefEval port |
|---|---|---|
| QA subset | opposed/supportive | **opposed-style only**, 200-word cap |
| Cross-session batching | Phase 1 turns batched across N sessions | N/A (chain count = 1). Per-turn calls go sequentially |
| Memory between sessions | STM/LTM/personas cleared between sessions | **Persisted across whole chain** |
| `flush_to_ltm` at Phase 1 end | Always called before QA | **Removed** — natural-boundary STM clear is the only flush. STM stays as in-flight context for QA |
| Persona side-channel | None | None (PrefEval personas are extracted from conversation, not given) |

---

## LLM call counts per checkpoint (worst case)

For a session with ~10 utterances (5 user + 5 assistant):
- Per-turn: 5 user-pairs × 2 calls (user persona + agent persona) = **10**
- Boundary summarize: 0 or 1 (only at FINALIZE_EVERY_N_CONVS boundary)
- QA: (k+1) batched

So each checkpoint adds 10–11 ingestion calls + (k+1) QA calls. For K=20 chain:
- Ingestion: ~20 × 10 = 200 persona calls + ~10 summarize = 210 calls
- QA: 1 + 2 + ... + 21 = 231 calls

---

## Notes

- Snapshot saved after QA succeeds. Mid-QA crash → resume re-ingests the
  affected session(s) via `process_turn` (re-running the persona/summarize
  LLM calls; this is the natural cost of strict invariants).
- The `Turn` dataclass populates `global_turn_id` cumulatively across the
  chain; LD-Agent does **not** consume it but theanine does. Kept for parity.
- Memory snapshot includes `last_conv_id` and `pending_conv_count`, so the
  natural boundary detection continues correctly across resume.
