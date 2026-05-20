# A-MEM — LoComo Experiment

Agentic Memory System (A-MEM) evaluated on the LoComo dataset.
Sessions are processed in parallel by batching vLLM calls across multiple samples.

## Architecture

```
run_experiment.py        Batch orchestration, checkpointing, result I/O
agent.py                 BaseAgent: memory interface + QA answering
memory_layer.py          AgenticMemorySystem, LLMWrapper, SimpleEmbeddingRetriever
llm_text_parsers.py      ANALYZE_CONTENT_PROMPT, parse/validate helpers
load_dataset.py          LoComo dataset loader (Sample, Session, Turn, QAPair)
config_0.py              Hyperparameters
merge_results.py         Merge outputs from parallel nohup runs
```

### Batch execution flow

For each batch of N samples processed simultaneously:

**Phase 1 — Memory Construction** (interleaved across samples, turn by turn):
```
For turn_idx = 0, 1, ..., max_turns:
  1. Build analyze prompts for all active turns → [BATCH] vLLM plain-text
  2. apply_analyze_result() → split into evolve / no-evolve
  3. Store no-evolve notes immediately
  4. [BATCH] vLLM JSON (evolution prompts)
  5. apply_evolve_result() → store evolved notes
```

**Phase 2 — QA Answering** (all samples batched, grouped by temperature):
```
1. Retrieve memories + build QA prompts for all samples
2. [BATCH] vLLM JSON in QA_BATCH_SIZE chunks (per temperature group)
3. JSON parse failures retried sequentially
4. Distribute results
```

**Phase 3 — Cleanup**: save snapshots, clear memory, write results, update checkpoint.

### LLM call types

| Call | Purpose | Format |
|------|---------|--------|
| `call_1_note_construction` | Analyze turn content → keywords/context/tags | plain text |
| `call_2_evolution` | Decide whether/how to evolve memory graph | JSON |
| `call_3_qa` | Answer QA question from retrieved memory | JSON |

## Configuration (`config_0.py`)

| Key | Description |
|-----|-------------|
| `TEMPERATURE` | Sampling temperature (default/temporal QA) |
| `TEMPERATURE_C5` | Temperature for category-5 adversarial QA |
| `JSON_RETRY` | Max retries for JSON generation |
| `BATCH_SIZE` | Samples processed in parallel |
| `QA_BATCH_SIZE` | QA prompts per vLLM batch call |
| `RETRIEVE_K` | Number of memories retrieved per query |
| `EVOLUTION_THRESHOLD` | Memory consolidation interval |
| `EMBEDDING_MODEL` | SentenceTransformer model name |

## Usage

```bash
python run_experiment.py \
    --start-sample 0 --end-sample 9 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel 1 --gpu-memory 0.5 \
    --config config_0
```

Outputs are written to `config_0_outputs_<model>/sample_<start>_<end>/`.

To merge results from multiple parallel runs:
```bash
python merge_results.py --config config_0 --model <model>
```

## Output schema

Each entry in `results.json`:

```json
{
  "sample_id": "...",
  "memory_at_qa_start": {"num_memories": 42, "total_content_tokens": 1200},
  "qa_results": [
    {
      "question": "...",
      "category": 1,
      "generated_answer": "...",
      "ground_truth_answer": "...",
      "evidence": "...",
      "retrieved_memories": [...],
      "qa_tokens": {"input": 512, "output": 20, "model": "..."}
    }
  ],
  "token_statistics": {
    "call_1_note_construction": {"input": 0, "output": 0, "llm_calls": 0, "parse_fallback_count": 0},
    "call_2_evolution":         {"input": 0, "output": 0, "llm_calls": 0},
    "call_3_qa":                {"input": 0, "output": 0, "llm_calls": 0},
    "total_input": 0, "total_output": 0, "total_llm_calls": 0
  },
  "evolution_statistics": {
    "evo_triggered_count": 5,
    "actions_taken": {"strengthen": 3, "update_neighbor": 2}
  },
  "memory_snapshot_path": "memory_snapshots/sample_0/",
  "config_metadata": {...}
}
```
