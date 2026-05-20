# OnlyLLM — Batched QA-Only Baseline

Top-level `only_llm/` is the batched main implementation. The original sequential implementation and historical outputs are preserved under `only_llm/only_llm_sequential/`.

LLM-only baseline with no memory or retrieval system. The context strategy is controlled by a single config flag:

| `HISTORY_ALL_GIVEN` | Strategy |
|---|---|
| `True` | All accumulated turns provided, trimmed from the oldest end when the token budget is exceeded |
| `False` | Sliding window: only the last `MAX_CONTEXT_TURNS` turns provided |

Follows the **QA-only experiment protocol** (`global_readme.md`) with one simplification in this implementation: Phase 1 builds context only and writes retrieval logs, while only Phase 2 (QA) makes real LLM calls.

---

## Files

```
only_llm/
├── config_lb.py        # Window-mode batch config
├── config_ub.py        # All-history batch config
├── run_experiment.py   # Batched main experiment script
├── load_dataset.py     # ImplexConv dataset loader
```

---

## Configuration (`config_lb.py`, `config_ub.py`)

```python
# Core switch
HISTORY_ALL_GIVEN = True

# Used when HISTORY_ALL_GIVEN = False
MAX_CONTEXT_TURNS = 20          # sliding window size (utterances)

# Used when HISTORY_ALL_GIVEN = True
OUTPUT_TOKEN_RESERVE  = 600     # token headroom reserved for model output
CONTEXT_SAFETY_MARGIN = 200     # buffer for fixed prompt parts
# token budget = max_model_len − OUTPUT_TOKEN_RESERVE − CONTEXT_SAFETY_MARGIN − tokens(fixed)
```

Multiple config files can be used without modifying the script. The config name becomes the output directory prefix:

```
config_lb.py      → config_lb_outputs_{model}_{subset}/
config_ub.py      → config_ub_outputs_{model}_{subset}/
```

---

## Experiment Flow

### Phase 1 — Context construction

For each `(user_turn, assistant_turn)` in the session:

1. Write retrieval/context log for the current state
2. Add `user_turn` + GT `assistant_turn` to context (QA exchanges excluded)

### Phase 2 — QA answering

After all turns are processed (context is frozen):

1. Format context → construct QA prompt
2. Call LLM → record answer and token usage

### Context trimming (`HISTORY_ALL_GIVEN = True`)

Two-phase strategy when the accumulated context exceeds the token budget:

1. **Rough cut** — sample 5 turns spread across history to estimate average tokens/turn; compute a rough start index keeping 80% of the estimated fit count (intentional overshoot)
2. **Fine trim** — compute exact token counts from the rough start; remove one turn at a time from the oldest end until the budget is satisfied

---

## Usage

```bash
# HISTORY_ALL_GIVEN=True  →  --max-model-len is required
python run_experiment.py \
    --start-session 0 --end-session 4 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 --gpu-memory 0.5 \
    --max-model-len 4096 \
    --batch-size 4 \
    --config config_ub

# HISTORY_ALL_GIVEN=False  →  --max-model-len is optional
python run_experiment.py \
    --start-session 0 --end-session 4 \
    --subset opposed \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 --gpu-memory 0.5 \
    --batch-size 4 \
    --config config_lb
```

| Argument | Required | Description |
|---|---|---|
| `--start-session` | ✓ | First session index (inclusive) |
| `--end-session` | ✓ | Last session index (inclusive) |
| `--subset` | ✓ | `opposed` or `supportive` |
| `--model` | | Model path (default from config) |
| `--tensor-parallel` | | Tensor parallel size |
| `--gpu-memory` | | GPU memory utilization (0–1] |
| `--max-model-len` | ✓ when `HISTORY_ALL_GIVEN=True` | Model max context length |
| `--batch-size` | | Number of sessions processed in parallel |
| `--config` | | Config file name without `.py` (default: `config_lb`) |

### Merging results

```bash
python merge_results.py only_llm config_lb_outputs_Llama-3.1-8B-Instruct_opposed
```

---

## Output

### Directory layout

```
{config}_outputs_{model}_{subset}/
└── session_{start}_{end}/
    ├── results_{model}_{subset}_session_{start}_{end}.json
    ├── checkpoint_{model}_{subset}_session_{start}_{end}.json
    ├── retrieval_logs/
    │   └── session_{id}_retrieval_log.jsonl
    └── prompt_log/
        └── session_{id}/
            └── call_5_qa/calls.jsonl
```

### `results_*.json`

```json
[
  {
    "session_id": 1,
    "config_metadata": {
      "config_name": "config_lb",
      "model": "...",
      "subset": "opposed",
      "llm_engine": "vllm",
      "temperature": 0.7,
      "max_tokens": 500,
      "max_model_len": 4096,
      "session_range": [0, 99],
      "batch_size": 4,
      "qa_batch_size": 64,
      "history_all_given": false,
      "max_context_turns": 20,
      "output_token_reserve": 600,
      "context_safety_margin": 200
    },
    "memory_at_qa_start": {
      "num_memories": 0,
      "total_content_tokens": 0
    },
    "qa_results": [
      {
        "question": "...",
        "generated_answer": "...",
        "ground_truth_answer": "...",
        "retrieved_memories": [],
        "qa_tokens": {"input": 300, "output": 50, "model": "..."}
      }
    ],
    "token_statistics": {
      "call_5_qa": {
        "input": 4500,
        "output": 1200,
        "llm_calls": 3,
        "parse_fallback_count": 1
      },
      "total_input": 4500,
      "total_output": 1200,
      "total_llm_calls": 3
    },
    "memory_snapshot_path": null
  }
]
```

- `retrieved_memories`: always `[]` — no memory module
- `memory_at_qa_start`: always zeroed because `only_llm` has no standalone memory store
- `prompt_log/`: stores only real QA LLM calls
