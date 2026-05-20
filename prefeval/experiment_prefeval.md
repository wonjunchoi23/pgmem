# PrefEval Experiment Specification

This document describes the experiment design for adapting the memory modules
(originally built for the ImplexConv dataset, see `../exp_implexconv_no_response/`)
to the **PrefEval implicit-persona** dataset.

When porting a memory module to this directory, follow the conventions described
here. The intent is that any future AI assistant (or human) editing a module's
runner can read this file and the companion `difference.md` and produce code
that is consistent across modules.

---

## 1. Dataset

- File: `dataset/implicit_persona.json`
- 1,000 independent samples. Each sample has the schema:

```
{
  "preference":      str,   # the user's true preference (ground truth label)
  "question":        str,   # the final test query (asked at evaluation time only)
  "explanation":     str,   # 1-sentence explanation of why the question is hard
  "persona":         str,   # persona assigned to this user (e.g. "a retired postal worker")
  "topic":           str,   # topic category, e.g. "education_learning_styles"
  "preference_type": str,   # always "implicit_persona" for this file
  "conversation": {
      "0": {"assistant": str, "user": str},
      "1": {"assistant": str, "user": str},
      ...
      "5": {"assistant": str, "user": str},
      "6": null              # placeholder; not used
  }
}
```

- Conversations are 5–6 turns on average (min 3, max 7). The last key (`"6"`)
  is always `null`.
- **Within each turn dict, the data stores `assistant` first and `user` second,
  but the natural conversation flow is `user → assistant`. Reorder when feeding
  to memory modules: emit the `user` utterance first, then the `assistant`
  utterance, for every turn.**
- The `question` field is the evaluation query and is **NEVER appended to the
  conversation that the memory module ingests**. It is only used at QA time.
- The `preference` and `explanation` fields are ground-truth signals for the
  judge. They are **NEVER passed to the memory module nor to the answering
  model**. They are only written to the output JSONL for downstream judging.

---

## 2. Experiment design: chained sessions

### 2.1 Treat unrelated samples as the same user

We pick the prefix `samples[0..K]` and pretend they are `K+1` consecutive
sessions of the **same** user, even though each sample has a different persona/
topic/preference. This is intentional: it injects noise/interference and tests
whether the memory module can still recall and apply the right preference for
each question, despite later sessions about unrelated topics being layered on
top.

### 2.2 Sample selection (deterministic, no random sampling)

Sessions are taken sequentially from the dataset:

```
chain = [ samples[0], samples[1], ..., samples[K] ]
```

The user passes `K` as `--end-session K`. The chain length is `K+1` sessions,
indexed `0..K`. Only **one chain** is run per invocation. There is no chain
sampling, no shuffling, no seed.

### 2.3 Cumulative memory build (incremental)

Memories are built **incrementally**, never re-built from scratch:

```
m_0     = build_memory(empty,   session_0)        # ingest session 0
m_1     = update_memory(m_0,    session_1)        # add session 1 on top of m_0
m_2     = update_memory(m_1,    session_2)        # ...
...
m_K     = update_memory(m_{K-1}, session_K)
```

Each `m_i` is the memory state after the user has finished sessions `0..i`.

The exact `update_memory` operation is module-specific (LD-Agent appends
extracted events; Theanine updates its memory graph; A-MEM stores notes and
optionally evolves them; etc.). The runner must call the module's native
incremental-update path, not a re-build.

A single in-memory module instance is reused across the chain. Between
checkpoints, the runner only **dumps** a snapshot to disk; it does not destroy
or rebuild the in-memory state.

### 2.4 QA evaluation per checkpoint (in-line)

After each memory checkpoint `m_i` is reached **in memory**, immediately
evaluate the questions of all sessions seen so far against that memory state,
**before moving on to session i+1**:

```
m_0  →  evaluate q_0
m_1  →  evaluate q_0, q_1
m_2  →  evaluate q_0, q_1, q_2
...
m_K  →  evaluate q_0, q_1, ..., q_K
```

Total QA calls per chain = `(K+1)(K+2)/2`.

The same question (e.g. `q_0`) is evaluated multiple times — once at every
later checkpoint. This is intentional: the goal is to measure whether the
answer for `q_0` degrades as more unrelated sessions are layered on top of `m_0`.
**Do not cache** answers across checkpoints.

This in-line evaluation order matters: it avoids re-loading every snapshot
from disk later.

### 2.5 What the answering model sees

When evaluating `q_j` against `m_i` (with `j ≤ i`):

- Input: `m_i` (in whatever form the module exposes — retrieved snippets,
  full memory dump, graph subset, etc.) + `q_j` (the question text only).
- The model is expected to produce a free-form answer to `q_j`, ideally
  consistent with `samples[j].preference`.
- **Do not** pass `samples[j].persona`, `samples[j].preference`, or
  `samples[j].explanation` into the prompt. These are evaluator-side signals only.

---

## 3. Virtual time and turn indexing

PrefEval has no native time axis, so we superimpose the same virtual-time
scheme used in ImplexConv (`format_virtual_time` / `compute_virtual_seconds`).

**Mapping (PrefEval → ImplexConv concepts):**

| ImplexConv field | PrefEval value |
|---|---|
| `session_id` | always `0` (one chain = one logical session) |
| `conv_id` | chain position of the PrefEval sample (`0..K`) |
| `turn_id` | sequential utterance index **local to the PrefEval conversation**, starting at `0`. User utterance at conversation position `k` → `turn_id = 2k`; assistant utterance → `turn_id = 2k+1`. Resets at each new session |
| `global_turn_id` | cumulative count of utterances **across the entire chain** so far. Never resets |

**Time constants (same as ImplexConv):**

- `CONV_IDS_PER_DAY = 2` (i.e. 2 sessions = 1 virtual day)
- `MINUTES_PER_TURN = 10`

**Memory item timestamp string (modules using A-MEM-style 3-part format):**

```
"0000_{conv_id:04d}_{turn_id:04d}"
```

i.e. `session_id` slot is fixed `0000` filler. `format_virtual_time` parses
parts[1] / parts[2] and is unaffected.

**Module ports**: reuse each module's existing time-format helpers verbatim,
just feed values from this mapping. `global_turn_id` is unused by amem /
memorybank / ldagent run logic but is required by Theanine (memory-graph
node spans), so the `Turn` dataclass should always populate it for parity.

---

## 4. Persona handling

**Do not pass `persona` to the memory module or to the answering model.**

Reasoning:
- Different sessions in a chain have different personas; passing them
  explicitly would tell the module "this is a different user", which defeats
  the noise-injection design.
- Memory modules should infer user characteristics from conversation alone,
  matching how LD-Agent / Theanine extract persona internally during ingestion.
- ImplexConv similarly does not provide persona side-channel information.

If a module's original ImplexConv pipeline accepts an explicit persona slot,
pass an empty string (or `None`, depending on the module's API) and let the
module do its own extraction from the conversation.

Persona text is still copied verbatim into the JSONL output for use by the
downstream judge.

---

## 5. CLI arguments

The runner CLI mirrors the ImplexConv runner (`run_experiment.py`) closely.

### Differences from ImplexConv
- `--end-session K` — chain is `samples[0..K]` (K+1 sessions, indexed `0..K`).
  This replaces `--start-session` / `--end-session` in ImplexConv, which
  selected a session range from the dataset.
- No `--subset` (PrefEval has no opposed/supportive split).
- No `--batch-size` for sessions (only one chain runs at a time).

### Same as ImplexConv (keep names)
- `--model`, `--tensor-parallel`, `--gpu-memory`, `--max-model-len`
- `--config <name>` (selects `config_<name>.py`)
- output paths derived from config helpers

There is no `--num_chains`, no `--seed`, no `--sampling_strategy`. Chain count
is always 1 and chain content is always `samples[0..K]`.

---

## 6. Resume behavior

The runner is **idempotent and extensible** across re-invocations on the same
output directory.

### 6.1 Detection
On startup, scan `memory_snapshots/m_*/` for the largest existing checkpoint
index. Call this `k_existing`. No separate checkpoint file is needed — the
snapshot directories themselves are the source of truth.

### 6.2 Branching
- If `k_existing` does not exist (no `m_*/` dirs): start fresh from `k=0`.
- If `K_target ≤ k_existing`: log "already complete up to K_target" and exit
  (no-op). Do not re-run QA, do not overwrite results.
- If `K_target > k_existing`:
  1. Load `m_{k_existing}` into the in-memory module via `load_snapshot()`.
  2. Continue ingestion from session `k_existing + 1` through `K_target`,
     saving each new snapshot and running its QA inline (§2.4).
  3. Append new rows to `results.jsonl` and `retrieval_log.jsonl`.
  4. Update cumulative `stats.json`.

### 6.3 Caveats
- Resume only works correctly if **config is unchanged** between runs (model,
  RETRIEVE_K, EVOLUTION_THRESHOLD, prompt templates, etc.). To start over
  with a different config, use a new output directory or delete the existing
  `memory_snapshots/`.
- Output dir is keyed by `{config_name}_outputs_{model}/`, so different models
  naturally write to different directories.

---

## 7. Output format

### 7.1 Directory layout

```
exp_prefeval/<module>/
└── config_<n>_outputs_<model>/
    ├── results.jsonl             # append-only, one row per (k, question_session)
    ├── retrieval_log.jsonl       # append-only, includes `k` field
    ├── stats.json                # cumulative token / evolution stats across runs
    ├── meta.json                 # config_metadata; written once on first run
    ├── memory_snapshots/
    │   ├── m_0/                  # module-specific snapshot dir
    │   ├── m_1/
    │   └── m_K/
    ├── prompt_log/               # per-call-type LLM prompt/output logs
    │   ├── call_2_note_construction/calls.jsonl
    │   ├── call_3_evolution/calls.jsonl
    │   └── call_4_qa/calls.jsonl
    └── logs/
```

There is no `chain_K{K}/` subfolder. Different `K` runs share the same
directory and extend incrementally per §6.

### 7.2 `results.jsonl` row schema

One line per `(checkpoint_index k, question_session_index j)` pair, `j ≤ k`:

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
  "retrieved_memories": [...],
  "qa_tokens": {"input": 0, "output": 0, "model": "..."}
}
```

Total rows per chain when run end-to-end = `(K+1)(K+2)/2`. On resume, only the
new `(k, j)` pairs (those with `k > k_existing`) are appended.

### 7.3 `retrieval_log.jsonl`

One line per QA-side retrieval, single file, append-only:

```json
{
  "timestamp":         "...",
  "phase":             "qa",
  "k":                 3,
  "question_session":  1,
  "query":             "...",
  "retrieved_items":   [...],
  "retrieval_scores":  [...],
  ...
}
```

The `k` field replaces ImplexConv's per-session-file split.

### 7.4 `memory_snapshots/`

Save **every** memory state `m_0, m_1, ..., m_K` to disk. Each snapshot is a
sub-directory whose contents are module-specific (e.g. `memories.json`,
`retriever.json`, `embeddings.npy`, `metadata.json` for A-MEM; pickled graph
for Theanine; raw event list JSON for LD-Agent).

### 7.5 `stats.json`

Aggregate token counts and module-specific stats (e.g. evolution counters)
**accumulated across all runs**. On resume, the existing file is loaded and
new totals are added before re-writing.

---

## 8. Constants

| Constant | Value | Notes |
|---|---|---|
| `CONV_IDS_PER_DAY` | 2 | Same as ImplexConv |
| `MINUTES_PER_TURN` | 10 | Same as ImplexConv |
| `MAX_TOKENS` | **1500** | Increased from ImplexConv's 750 — PrefEval answers are advisory and need more headroom |
| QA answer word cap | **200 words** | Reflected in the QA prompt template's "Be concise (maximum 200 words)" line |
| `RETRIEVE_K` | module default | Carry over from each module's config |
| `EVOLUTION_THRESHOLD` (A-MEM) | module default | Carry over |

These values should be reflected in each module's `config_<n>.py` and in the
QA prompt templates.

---

## 9. Batching

Concurrency / batching opportunities, ranked by expected speedup:

1. **QA generation at a checkpoint** — at checkpoint `k`, the questions
   `q_0, ..., q_k` are independent. Batch all `k+1` together via
   `generate_batch_raw` (or equivalent). For very large `k`, chunk by
   `QA_BATCH_SIZE`.
2. **Judge calls** — fully independent across all `(i, j)` pairs once
   answers are produced. Judging is deferred (user runs it as a separate
   step), but when it happens, batch everything.
3. **Memory build within a session** — sequential within a session because
   later turns' evolution may depend on memory written by earlier turns.
   Across sessions in the chain it is also strictly sequential because
   `m_i = f(m_{i-1}, session_i)`.

There is no cross-session parallelism for the chain itself (chain count = 1).

---

## 10. Adapting an ImplexConv module — checklist

When porting a module from `../exp_implexconv_no_response/<module>/` to a
new subdirectory under `exp_prefeval/<module>/`:

- [ ] Replace `load_dataset.py` with a PrefEval loader that yields a
      `Session`-shaped object per PrefEval sample. Inside each session,
      reorder `(assistant, user)` to `(user, assistant)` per turn.
      Populate `conv_id`, `turn_id` (local), and `global_turn_id`
      (cumulative across chain) per §3.
- [ ] Drop `subset` everywhere (no opposed/supportive split).
- [ ] Drop persona side-channel input. Pass empty/None.
- [ ] Keep all module-internal files (`memory_layer.py`, `agent.py`,
      `theanine_module.py`, etc.) **unchanged** unless they contain
      ImplexConv-specific assumptions (e.g. yes/no QA schema).
- [ ] In the runner:
  - [ ] Replace the multi-session loop with the cumulative-chain loop (§2.3).
  - [ ] Use a single in-memory module instance; only `save_snapshot` per
        checkpoint, never `clear()` until the chain is fully done.
  - [ ] At each checkpoint, run QA for sessions `0..k` immediately (§2.4).
  - [ ] Append rows to `results.jsonl` and `retrieval_log.jsonl`.
  - [ ] Save snapshot to `memory_snapshots/m_{k}/`.
- [ ] Implement the resume logic (§6): scan `memory_snapshots/m_*/` for
      `k_existing` and either no-op, start fresh, or `load_snapshot` and
      continue.
- [ ] Update `config_<n>.py` to set `MAX_TOKENS=1500` and the dataset path
      to `exp_prefeval/dataset/implicit_persona.json`. Drop `DATASET_OPPOSED`
      / `DATASET_SUPPORTIVE`.
- [ ] Update the QA prompt template's word cap to **200 words**.
- [ ] CLI: `--end-session K`. Drop `--start-session`, `--end-session` (range),
      `--subset`, `--batch-size`.
- [ ] Apply batching at the QA stage (§9 #1).

See `difference.md` for a side-by-side comparison with the ImplexConv setup.
