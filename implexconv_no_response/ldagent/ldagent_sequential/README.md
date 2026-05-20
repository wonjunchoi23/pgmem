# LD-Agent Memory Module (ImplexConv) — QA-Only Variant

Archived sequential reference kept under `ldagent/ldagent_sequential/`.
The current main implementation lives in `ldagent/` and is the version intended for active runs.

Implementation of the LD-Agent memory system adapted for the ImplexConv QA-only experiment
protocol (`global_readme.md`). Adapted from [Li et al., NAACL 2025](https://arxiv.org/abs/2406.05925).

Response generation LLM calls are removed. Phase 1 constructs and logs the response prompt
each turn but does NOT call the LLM. Memory is built identically using GT agent responses.
Only QA (Task 2) is evaluated.

---

## Architecture

Dual memory system:

- **STM (Short-Term Memory)**: Accumulates all dialogue turns (user + agent) across conv_ids.
  Every `FINALIZE_EVERY_N_CONVS` conv_ids (= 1 virtual day), the STM is summarized → written
  to LTM and then **fully cleared** (original LD-Agent behavior). At the end of Phase 1,
  any remaining STM is flushed to LTM by `flush_to_ltm()` and **kept intact** for QA context.
- **LTM (Long-Term Memory)**: ChromaDB vector store of conversation summaries. Each entry is
  created at an N-conv boundary or at the final flush.

Persona module maintains user and agent trait banks updated every turn.

---

## Files

| File | Role |
|---|---|
| `event_memory.py` | STM + LTM management (ChromaDB, noun-based retrieval, virtual-time decay) |
| `personas.py` | User / agent trait extraction and storage |
| `generator.py` | Response prompt construction (no LLM call) + QA answering (JSON structured output) |
| `ldagent_module.py` | Single entry point wrapping all three components; `LLMCallLogger` class |
| `run_experiment.py` | Experiment loop (session loop, checkpointing, token stats, retrieval logging) |
| `config_0.py` | All hyperparameters and output path helpers |
| `load_dataset.py` | ImplexConv dataset loader; virtual-time utilities |
| `merge_results.py` | Merge results from multiple session-range runs |

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

## Memory Retrieval per Turn

| Source | Count | Used in prompt |
|---|---|---|
| STM (`context_retrieve`) | all turns since last clear | `<CONTEXT>` |
| LTM (`relevance_retrieve`) | 0 or 1 summary entry | `<MEMORY>` |
| Personas — user | latest `MAX_USER_PERSONAS` traits | `<USER_TRAITS>` (prompt + QA) |
| Personas — agent | latest `MAX_AGENT_PERSONAS` traits | system prompt (prompt + QA) |

LTM retrieval scores candidates by **noun overlap × virtual-time decay**
(`exp(-DECAY_TEMP × elapsed_virtual_seconds)`).

---

## Prompt Structure

### Response Prompt (`_select_prompts`) — constructed but NOT executed

```
[SYSTEM]
As a communication expert... you embody the role of {agent_name}.
Here are some of your distinctive personal traits: {agent_traits}.

[USER]
<CONTEXT>
Drawing from your recent conversation with {usr_name}: {context}
<MEMORY>
The memories linked to the ongoing conversation are: {memories}
<USER_TRAITS>
You found that {usr_name} has the following characteristics: {user_traits}

Now, please role-play as {agent_name} to continue the dialogue...
Respond in JSON format with key "response".
```

The prompt is constructed and logged to `call_1_response` with `output: null`. The LLM is
never called for response generation. Input tokens are estimated via the tokenizer.

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

All internal LLM calls use **constrained decoding** (`guided_json`) to prevent
`<think>` reasoning tokens from being stored in memory or traits:

| Call | Schema | Purpose |
|---|---|---|
| `build_response_prompt_only` | — | Prompt construction only (no LLM call) |
| `generate_qa_answer` (opposed) | `{"answer": str}` | Free-form QA |
| `generate_qa_answer` (supportive) | `{"answer": enum["yes","no"]}` | Yes/no QA |
| `_user_traits_update` | `{"trait": str}` | User persona extraction |
| `_agent_traits_update` | `{"trait": str}` | Agent persona extraction |
| `_context_summarize` | `{"summary": str}` | STM→LTM boundary summarization |

---

## Experiment Protocol

Processes **individual sessions** following `global_readme.md` (QA-only variant).

### Phase 1 — Memory Construction (no response generation LLM call)

For each `(user_turn, assistant_turn)` pair in a session:
1. Compute `virtual_seconds` for the turn from `(conv_id, turn_id)`
2. Retrieve full STM context (`context_retrieve`) + LTM memories (`relevance_retrieve`)
   — at every `FINALIZE_EVERY_N_CONVS` conv_id boundary: summarize STM → write LTM → **clear STM**
3. Update user persona from user utterance
4. Build response prompt (NO LLM call) — log to `call_1_response` with `output: null`, count input tokens
5. Update agent persona from **GT agent response**
6. Store **GT agent response** in STM
7. Write retrieval log entry (`phase="prompt_construction"`)

### Phase 2 — QA Answering

After all turns are processed and memory is fully constructed:
1. `flush_to_ltm()` — commit remaining STM to LTM; **STM retained** for QA context
2. For each QA pair: retrieve from LTM (`RETRIEVE_K` entries), build STM context,
   generate answer with user + agent traits, write retrieval log

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
RELEVANCE_MEMORY_NUMBER = 1     # LTM entries retrieved per response turn
RETRIEVE_K              = 1     # LTM entries retrieved for QA
DIST_THRESHOLD          = 1.5   # ChromaDB distance threshold
DECAY_TEMP              = 1e-4  # LTM virtual-time decay (exp(-1e-4 × 12000) ≈ 0.30 after 1 day)
CONV_IDS_PER_DAY        = 2     # conv_ids per virtual day (finalization granularity)
MINUTES_PER_TURN        = 10    # minutes per local turn_id step
FINALIZE_EVERY_N_CONVS  = 2     # STM→LTM flush every N conv_ids (= 1 virtual day)
MAX_USER_PERSONAS       = 5     # traits shown in prompt
MAX_AGENT_PERSONAS      = 5
MAX_TOKENS              = 750   # LLM max output tokens (QA and internal calls)
ENABLE_LLM_CALL_LOGGING = True  # log all LLM calls to prompt_log/
```

`TIMING_CONV_ID` is defined in config but unused in this variant (no Phase 1 LLM call).

---

## LLM Call Logging

When `ENABLE_LLM_CALL_LOGGING = True`, all LLM calls are logged per session to
`prompt_log/session_{id}/` with one JSONL file per call type:

| Folder | Call |
|---|---|
| `call_1_response/calls.jsonl` | Response prompt only (`output: null` — no LLM call) |
| `call_2_user_persona/calls.jsonl` | User trait extraction |
| `call_3_agent_persona/calls.jsonl` | Agent trait extraction |
| `call_4_summarization/calls.jsonl` | STM→LTM summarization |
| `call_5_qa/calls.jsonl` | QA answer generation |

Each line is a JSON object with `timestamp`, `call_type`, `system_prompt`, `user_prompt`, `output`.
For `call_1_response`, `output` is always `null`.

---

## Token Tracking

Token counts are split into three categories per session:

- `total_response_input` — estimated input tokens for constructed (but not executed) response prompts
  (via tokenizer; no output tokens since the LLM is never called)
- `qa_tokens` — QA answer generation calls
- `other_tokens` — persona extraction (×2 per turn) + STM→LTM boundary summarization + flush summarization

`total_input = total_response_input + total_qa_input + total_other_input`
`total_output = total_qa_output + total_other_output`
`num_qa_api_calls` counts only successful QA generation calls (exceptions count as 0).
`num_total_api_calls` counts only actual API calls (QA + internal); response prompt construction
does not count.

---

## Usage

### Basic run (single GPU, single process)

```bash
python ldagent/ldagent_sequential/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 \
    --gpu-memory 0.11 \
    --config config_0
```

### With large model context window (max_model_len)

```bash
python ldagent/ldagent_sequential/run_experiment.py \
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
CUDA_VISIBLE_DEVICES=0 nohup python ldagent/ldagent_sequential/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model Qwen/Qwen3-1.7B \
    --tensor-parallel 1 --gpu-memory 0.11 \
    --max-model-len 3000 \
    --config config_0 \
    > nohup/nohup_opp_1.7b_session_0_29.out 2>&1 &
```

Output will be saved to `ldagent/nohup/` (auto-created if needed).

### Override max tokens from CLI

```bash
python ldagent/ldagent_sequential/run_experiment.py \
    --start-session 0 --end-session 29 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-tokens 300 \
    --config config_0
```

### Merge results from multiple runs

```bash
python ldagent/ldagent_sequential/merge_results.py \
    --model Qwen/Qwen3-1.7B \
    --subset opposed \
    --config config_0

# Dry run (show what would be merged)
python ldagent/ldagent_sequential/merge_results.py \
    --model Qwen/Qwen3-1.7B \
    --subset opposed \
    --config config_0 \
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
| `--config` | str | `config` | Config file name (without `.py`) in `ldagent/` |
| `--max-tokens` | int | `None` | Override MAX_TOKENS from config |

`--config` specifies the config file name (without `.py`) in the `ldagent/` directory.
The file is **dynamically loaded at runtime** — rename or copy `config_0.py` freely
(e.g. `config_k5.py`, `config_large.py`) and pass via `--config`. Both `run_experiment.py`
and `merge_results.py` accept `--config` so paths remain consistent.

---

## Output Structure

```
ldagent/
├── nohup/                                  ← auto-created for background run logs
│   ├── nohup_opp_1.7b_session_0_29.out
│   └── ...
├── {config}_outputs_{model}_{subset}/
│   ├── session_{start}_{end}/
│   │   ├── results_{model}_{subset}_session_{start}_{end}.json
│   │   ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
│   │   ├── retrieval_logs/
│   │   │   └── session_{id}_retrieval_log.jsonl
│   │   ├── memory_snapshots/
│   │   │   └── session_{id}/
│   │   ├── prompt_log/
│   │   │   └── session_{id}/
│   │   │       ├── call_1_response/calls.jsonl      ← prompt only, output=null
│   │   │       ├── call_2_user_persona/calls.jsonl
│   │   │       ├── call_3_agent_persona/calls.jsonl
│   │   │       ├── call_4_summarization/calls.jsonl
│   │   │       └── call_5_qa/calls.jsonl
│   │   └── logs/
│   └── results_{model}_{subset}_merged.json
└── logs/                                    ← auto-created for experiment logs
```

Results are saved atomically (via temp file + rename) to prevent corruption.

### Results JSON Schema

Each session result:

```json
{
  "session_id": 1,
  "qa_results": [...],
  "timing_statistics": {
    "phase2_qa_avg": {"retrieval_time": 0.031, "inference_time": 0.142, "total_time": 0.173}
  },
  "token_statistics": {
    "total_response_input": 9800,
    "total_qa_input": 4500,
    "total_qa_output": 1200,
    "total_input": 25100,
    "total_output": 5500,
    "num_qa_api_calls": 3,
    "num_total_api_calls": 273
  },
  "memory_snapshot_path": "memory_snapshots/session_1/"
}
```

- `response_prompts` is not saved in results. The constructed response prompt is logged
  separately in `prompt_log/call_1_response/calls.jsonl` only.
- `num_qa_api_calls` counts only **successful** QA generation calls; questions where the
  LLM call raised an exception count as 0.

### Retrieval Log (`retrieval_logs/session_{id}_retrieval_log.jsonl`)

One JSON object per line, one entry per turn (Phase 1) and per QA question (Phase 2).

Each entry includes parallel `memory_type` and `num_retrieved` arrays covering all
four memory sources used in the prompt:

```json
{
  "memory_type":   ["ltm", "stm", "user_trait", "agent_trait"],
  "num_retrieved": [1, 14, 3, 2]
}
```

| Field | Description |
|---|---|
| `ltm` | LTM entries retrieved by relevance (0 or 1 per turn; up to `RETRIEVE_K` for QA) |
| `stm` | STM turns in the context window at retrieval time |
| `user_trait` | Number of user persona traits passed to the prompt |
| `agent_trait` | Number of agent persona traits passed to the prompt |

Phase 1 entries use `"phase": "prompt_construction"` (base protocol used `"response_generation"`).
Phase 2 entries use `"phase": "qa"` (unchanged).

`retrieved_items[].source_turn` contains `session_id`, `conv_id`, and `virtual_seconds`
(instead of `turn_id`) to identify when the memory entry was created.

Full prompt snapshots are **not** stored in the retrieval log. They are logged separately
in `prompt_log/` (`call_1_response` for Phase 1, `call_5_qa` for Phase 2).

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
