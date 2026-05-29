# LD-Agent Memory Module (ImplexConv) — Batched Main Variant, QA-Only

Implementation of the LD-Agent memory system adapted for the ImplexConv QA-only experiment
protocol (`global_readme.md`). Adapted from [Li et al., NAACL 2025](https://arxiv.org/abs/2406.05925).

Response generation LLM calls are not used in this variant. Phase 1 builds memory using
GT agent responses directly. Only QA (Task 2) is evaluated.

This `ldagent/` implementation keeps session memory states independent, but batches the
internal LLM calls for boundary summarization, persona extraction, final STM flush, and QA.
The goal is to preserve the original session semantics while improving GPU utilization.

The archived sequential implementation is kept under `ldagent/ldagent_sequential/` for reference.

---

## Architecture

Dual memory system:

- **STM (Short-Term Memory)**: Accumulates all dialogue turns (user + agent) across conv_ids.
  Every `FINALIZE_EVERY_N_CONVS` conv_ids (= 1 virtual day), the STM is summarized → written
  to LTM and then **fully cleared** (original LD-Agent behavior). At the end of Phase 1,
  any remaining STM is flushed to LTM by `flush_to_ltm()` and **kept intact** for QA context.
- **LTM (Long-Term Memory)**: numpy-based vector store of conversation summaries. Embeddings
  are computed with `sentence-transformers/all-MiniLM-L6-v2` and stored as `np.ndarray` lists
  in memory. Each entry is created at an N-conv boundary or at the final flush.

Persona module maintains user and agent trait banks updated every turn.

---

## Files

| File | Role |
|---|---|
| `event_memory.py` | STM + LTM management (numpy/SentenceTransformer, noun-based retrieval, virtual-time decay) |
| `personas.py` | User / agent trait extraction and storage |
| `generator.py` | QA answering (JSON structured output) |
| `ldagent_module.py` | Single entry point wrapping all three components; `LLMCallLogger` class |
| `run_experiment.py` | Batched experiment loop (turn-aligned session batches, checkpointing, token stats, retrieval logging) |
| `config_0.py` | All hyperparameters and output path helpers |
| `load_dataset.py` | ImplexConv dataset loader; virtual-time utilities |

---

## Virtual Time Model

Each turn carries a `virtual_seconds` value computed from its `(conv_id, turn_id)`:

```
1 virtual day  = CONV_IDS_PER_DAY consecutive conv_ids  (default: 2)
1 virtual turn = MINUTES_PER_TURN minutes                (default: 10)
```

This maps to approximately 12,000 virtual seconds per day (2 conv_ids × ~10 turns × 10 min × 60 s).

`virtual_seconds` is used for:
- LTM time-decay scoring: `exp(-DECAY_TEMP × elapsed_virtual_seconds)`
- STM finalization boundary: every `FINALIZE_EVERY_N_CONVS` conv_ids (= 1 day)
- `<MEMORY>` prompt display: human-readable elapsed time via `convert_seconds_to_full_time()`

---

## Memory Retrieval

| Source | Count | Used in |
|---|---|---|
| STM (`context_retrieve`) | all turns since last clear | QA `<CONTEXT>` |
| LTM (`relevance_retrieve`) | 0 or 1 summary entry* | QA `<MEMORY>` |
| Personas — user | latest `MAX_USER_PERSONAS` traits | QA `<USER_TRAITS>` |
| Personas — agent | latest `MAX_AGENT_PERSONAS` traits | QA system prompt |

\* `RETRIEVE_K` / `RELEVANCE_MEMORY_NUMBER` controls the candidate pool size passed to the
embedding search, but the noun overlap × time-decay scorer always returns at most **1 entry**
(the single best match with `overlap_count > 0`).

LTM retrieval: top-`n` candidates by L2 distance (sentence-transformer embeddings), then
rescored by **noun overlap × virtual-time decay** (`exp(-DECAY_TEMP × elapsed_virtual_seconds)`).

Phase 1 (turn processing) **does not** call `relevance_retrieve` — LTM retrieval only occurs
at QA time.

---

## QA Prompt Structure

### QA Generation (`_build_qa_prompt_opposed/supportive`)

```
[SYSTEM]
You are a helpful assistant that answers questions about a user...

[USER]
<CONTEXT>
Recent conversation turns: {stm_context}
<MEMORIES>
The following are conversation memories about the user: {memories}
<USER_TRAITS>
User characteristics: {user_traits}
<QUESTION>
Question: {question}
Respond in JSON format with key "answer".
```

- **opposed**: free-form answer (schema: `{"answer": str}`)
- **supportive**: constrained to `"yes"` or `"no"` only (schema: `{"answer": enum["yes","no"]}`)

---

## JSON Mode (guided_json)

All LLM calls use **constrained decoding** (`guided_json`):

| Call | Schema | Purpose |
|---|---|---|
| `generate_qa_answer` (opposed) | `{"answer": str}` | Free-form QA |
| `generate_qa_answer` (supportive) | `{"answer": enum["yes","no"]}` | Yes/no QA |
| `_user_traits_update` | `{"trait": str}` | User persona extraction |
| `_agent_traits_update` | `{"trait": str}` | Agent persona extraction |
| `_context_summarize` | `{"summary": str}` | STM→LTM boundary/flush summarization |

---

## Experiment Protocol

Processes **individual sessions** following `global_readme.md` (QA-only variant).

### Phase 1 — Memory Construction

For each `(user_turn, assistant_turn)` pair in a session:
1. Compute `virtual_seconds` for the turn from `(conv_id, turn_id)`
2. Append user turn to STM (`append_user_query`)
   — at every `FINALIZE_EVERY_N_CONVS` conv_id boundary: summarize STM → write LTM → **clear STM**
3. Update user persona from user utterance (`call_2_user_persona`)
4. Update agent persona from **GT agent response** (`call_3_agent_persona`)
5. Store **GT agent response** in STM

### Phase 2 — QA Answering

After all turns are processed and memory is fully constructed:
1. `flush_to_ltm()` — commit remaining STM to LTM (`call_4_summarization`); **STM retained** for QA context
2. For each QA pair: retrieve from LTM (`RETRIEVE_K` entries), build STM context,
   generate answer with user + agent traits (`call_5_qa`), write retrieval log

**supportive subset**: answer normalized to `"yes"` or `"no"` (fallback `"unknown"` if
neither matches after normalization)

### Phase 3 — Cleanup

Save memory snapshot → clear all memory state → save results + checkpoint.

**GT agent responses are always used** — generated responses are never fed back into memory
or dialogue history. **QA exchanges are never stored in memory** and do not appear in
dialogue history.

---

## Key Config Values

```python
RELEVANCE_MEMORY_NUMBER = 3     # candidate pool size for LTM retrieval (scorer returns at most 1)
RETRIEVE_K              = 3     # candidate pool size for LTM retrieval at QA time
DIST_THRESHOLD          = 1.5   # L2 distance threshold — only active when ORI_MEM_QUERY=True
DECAY_TEMP              = 1e-4  # LTM virtual-time decay (exp(-1e-4 × 12000) ≈ 0.30 after 1 day)
CONV_IDS_PER_DAY        = 2     # conv_ids per virtual day (finalization granularity)
MINUTES_PER_TURN        = 10    # minutes per local turn_id step
FINALIZE_EVERY_N_CONVS  = 2     # STM→LTM flush every N conv_ids (= 1 virtual day)
MAX_USER_PERSONAS       = 10    # traits shown in prompt
MAX_AGENT_PERSONAS      = 10
MAX_TOKENS              = 750   # LLM max output tokens (QA and internal calls)
ENABLE_LLM_CALL_LOGGING = True  # log all LLM calls to prompt_log/
```

---

## LLM Call Logging

When `ENABLE_LLM_CALL_LOGGING = True`, all LLM calls are logged per session to
`prompt_log/session_{id}/` with one JSONL file per call type:

| Folder | Call |
|---|---|
| `call_2_user_persona/calls.jsonl` | User trait extraction |
| `call_3_agent_persona/calls.jsonl` | Agent trait extraction |
| `call_4_summarization/calls.jsonl` | STM→LTM summarization (boundary + flush) |
| `call_5_qa/calls.jsonl` | QA answer generation |

Each line is a JSON object with `call_type`, `system_prompt`, `user_prompt`, `output`.

---

## Token Tracking

Token counts are tracked per call type:

| Key | Description |
|---|---|
| `call_2_user_persona` | User trait extraction — one call per turn |
| `call_3_agent_persona` | Agent trait extraction — one call per turn |
| `call_4_summarization` | STM→LTM summarization — boundary (Phase 1) + flush (Phase 2) merged |
| `call_5_qa` | QA answer generation — one call per QA question |
| `total_input` | Sum of input tokens across all call types |
| `total_output` | Sum of output tokens across all call types |
| `total_llm_calls` | Sum of llm_calls across all call types |

`call_5_qa.llm_calls` counts only successful QA generation calls (exceptions count as 0).

---

## Usage

### Basic run (single GPU, batched sessions)

```bash
python ldagent/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.11 \
    --batch-size 4 \
    --config config_0
```

### With large model context window (max_model_len)

```bash
python ldagent/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.11 \
    --max-model-len 3000 \
    --config config_0
```

### Run in background with nohup

```bash
CUDA_VISIBLE_DEVICES=0 nohup python ldagent/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 --gpu-memory 0.11 \
    --max-model-len 3000 \
    --config config_0 \
    > nohup/nohup_opp_1.7b_session_0_29.out 2>&1 &
```

### Merge results from multiple runs

```bash
python merge_results.py ldagent config_0_outputs_Qwen3-1.7B_opposed

# Dry run (show what would be merged)
python merge_results.py \
    ldagent \
    config_0_outputs_Qwen3-1.7B_opposed \
    --dry-run
```

### Arguments explained

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start-session` | int | **required** | First session to process (inclusive) |
| `--end-session` | int | **required** | Last session to process (inclusive) |
| `--subset` | str | **required** | `opposed` or `supportive` dataset subset |
| `--model` | str | `meta-llama/Llama-3.1-8B-Instruct` | Model path (HF format or local) |
| `--tensor-parallel` | int | `1` | Tensor parallelism degree (for vLLM) |
| `--gpu-memory` | float | `0.5` | GPU memory utilization fraction (0.0–1.0) |
| `--max-model-len` | int | `None` | vLLM max_model_len (optional, for large models) |
| `--config` | str | `config_0` | Config file name (without `.py`) in `ldagent/` |

---

## Output Structure

```
ldagent/
├── nohup/
├── {config}_outputs_{model}_{subset}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{subset}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{id}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{id}/
│   │   │       ├── short_term_memory.json
│   │   │       ├── long_term_memory.json   ← documents + metadatas (no embeddings)
│   │   │       ├── ltm_embeddings.npy      ← LTM embedding matrix (n × 384 float32)
│   │   │       ├── memory_state.json
│   │   │       └── personas.json
│   │   ├── prompt_log/
│   │   │   └── session_{id}/
│   │   │       ├── call_2_user_persona/calls.jsonl
│   │   │       ├── call_3_agent_persona/calls.jsonl
│   │   │       ├── call_4_summarization/calls.jsonl
│   │   │       └── call_5_qa/calls.jsonl
│   │   └── logs/
│   └── results_{model}_{subset}_merged.json
```

Results are saved atomically (via temp file + rename) to prevent corruption.

### Results JSON Schema

Each session result:

```json
{
  "session_id": 1,
  "config_metadata": {
    "config_name": "config_0",
    "model": "Qwen/Qwen3-1.7B",
    "subset": "opposed",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "temperature": 0.7,
    "max_tokens": 750,
    "session_range": [0, 99],
    "relevance_memory_number": 3,
    "retrieve_k": 3,
    "dist_threshold": 1.5,
    "conv_ids_per_day": 2,
    "minutes_per_turn": 10,
    "finalize_every_n_convs": 2,
    "decay_temp": 0.0001,
    "max_user_personas": 10,
    "max_agent_personas": 10
  },
  "memory_at_qa_start": {
    "num_memories": 12,
    "total_content_tokens": 85
  },
  "qa_results": [
    {
      "question": "...",
      "generated_answer": "...",
      "ground_truth_answer": "...",
      "retrieved_memories": [{"session_id": 1, "conv_id": 3, "virtual_seconds": 7200.0, "score": 0.42}],
      "qa_tokens": {"input": 512, "output": 48, "model": "Qwen/Qwen3-1.7B"}
    }
  ],
  "token_statistics": {
    "call_2_user_persona":  {"input": 8200, "output": 1100, "llm_calls": 82},
    "call_3_agent_persona": {"input": 8200, "output": 1100, "llm_calls": 82},
    "call_4_summarization": {"input": 950,  "output": 180,  "llm_calls": 7},
    "call_5_qa":            {"input": 1800, "output": 240,  "llm_calls": 3},
    "total_input":  19150,
    "total_output": 2620,
    "total_llm_calls": 174
  },
  "memory_snapshot_path": "memory_snapshots/session_1"
}
```

### Retrieval Log (`retrieval_logs/session_{id}_retrieval_log.jsonl`)

One JSON object per line, one entry per QA question (Phase 2 only).
Phase 1 (turn processing) no longer writes retrieval log entries.

```json
{
  "phase": "qa",
  "session_id": 1,
  "conv_id": -1,
  "turn_id": 0,
  "query": "...",
  "memory_type":   ["ltm", "stm", "user_trait", "agent_trait"],
  "num_retrieved": [1, 14, 3, 2],
  "retrieved_items": [...],
  "module_specific": {"ltm_entry_count": 4, "stm_context_turns": 14, ...}
}
```

---

## Citation

```bibtex
@inproceedings{li2024hello,
  title={Hello Again! LLM-powered Personalized Agent for Long-term Dialogue},
  author={Li, Hao and others},
  booktitle={NAACL},
  year={2025}
}
```
