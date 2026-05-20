# Differences: ImplexConv vs. PrefEval setup

This file is a quick-reference for any AI assistant porting a memory module
from `../exp_implexconv_no_response/<module>/` to `exp_prefeval/<module>/`.
For the full PrefEval design, see `experiment_prefeval.md`.

---

## 1. Dataset shape

| Aspect | ImplexConv | PrefEval (this directory) |
|---|---|---|
| Top-level unit | "Session" containing many conv_ids and turns | One sample = one short conversation |
| Schema | `metadata` + `conversations[]` (flat list of utterances) + `qa[]` | `preference` + `question` + `explanation` + `persona` + `topic` + `conversation` (dict keyed by turn index) |
| Conversation length | Tens to hundreds of turns | 5–6 turns on average (max 7) |
| Turn ordering in raw data | `speaker`-tagged utterances in chronological order | Each turn dict has `assistant` first then `user` — must be **reordered to `user → assistant`** before feeding the module |
| Multiple QAs per unit | Yes, `qa[]` list with `retrieved_conv_ids` grounding | One question per sample, no grounding pointers |
| Ground-truth format | `answer` string | `preference` (a stated preference) + `explanation` — judging is preference-adherence, not string match |
| Subset split | `opposed` / `supportive` | None — only one variant. Drop all subset branching |
| Persona | Not provided | Provided per sample, but **deliberately not passed to the module** (see §3) |

---

## 2. Experiment topology

| Aspect | ImplexConv | PrefEval |
|---|---|---|
| What gets memorized | One long session of one user | A chain of `K+1` short PrefEval samples concatenated as if they were the same user |
| User identity assumption | True — single user across the session | **False but pretended-true** — every sample has a different persona; we treat them as one user to inject noise |
| Number of "chains" / runs per invocation | One session at a time (or as configured) | Always exactly **one** chain per invocation |
| Sample selection | Driven by dataset's session_id | Strict prefix `samples[0..K]`. No randomness, no seed, no sampling strategy |
| Cumulative checkpoints | Not a thing — memory is built once | Core of the experiment: memory snapshots `m_0, m_1, ..., m_K` taken after every session |
| Memory build mode | Single pass | **Incremental**: `m_i = update(m_{i-1}, session_i)`. Never rebuild |
| In-memory instance lifecycle | Built then cleared at session end | One module instance reused across the entire chain. Only `save_snapshot` is called between checkpoints |
| QA evaluation timing | After full session ingestion | At every checkpoint, against questions of all sessions seen so far. Same question gets re-asked at every later checkpoint to measure degradation |
| Total QA calls | ≈ number of QAs in the session | `(K+1)(K+2)/2` per fully-completed chain |
| Resume support | Per-session checkpoint file | **Snapshot-based**: scan `memory_snapshots/m_*/` for largest existing `k`, `load_snapshot` and continue from `k+1` |

---

## 3. Persona handling

| | ImplexConv | PrefEval |
|---|---|---|
| Persona in dataset | Not present | Present per sample |
| Persona passed to memory module | N/A | **No.** Pass empty/None even though it is available. Reason: chained samples have different personas; revealing them defeats the noise-injection setup. The module should infer user traits from conversation only |
| Persona in JSONL output | N/A | Yes — copied as ground truth for downstream judging only |

If the original ImplexConv module had no persona input slot, no change is
needed. If it had a placeholder (e.g. some loaders accept a persona string),
keep the placeholder and pass empty.

---

## 4. Virtual time and turn indexing

| | ImplexConv | PrefEval |
|---|---|---|
| Source of time | `conv_id` and `turn_id` in the data | Synthesized from chain position |
| `session_id` | Per-session (varies) | **Always `0`** (one chain = one logical session) |
| `conv_id` | Per-conv | PrefEval sample's chain position (`0..K`) |
| `turn_id` | Local utterance index within conv (resets at conv boundary) | Local utterance index within the PrefEval conversation. User at conv position `k` → `turn_id = 2k`, assistant → `turn_id = 2k+1` |
| `global_turn_id` | Cumulative across whole session | Cumulative across the **whole chain**, never resets |
| Time helpers | `format_virtual_time`, `compute_virtual_seconds` | Same helpers, unchanged. `CONV_IDS_PER_DAY=2`, `MINUTES_PER_TURN=10` |
| 3-part timestamp string (A-MEM style) | `"SSSS_CCCC_TTTT"` | `"0000_{conv_id:04d}_{turn_id:04d}"` (session slot is fixed `0000`) |

`global_turn_id` is unused by amem / memorybank / ldagent run logic but is
required by Theanine for memory-graph node spans, so the `Turn` dataclass
must always populate it.

---

## 5. Evaluation (QA + judge)

| | ImplexConv | PrefEval |
|---|---|---|
| What is being judged | Whether the answer matches the gold `answer` (string match or LLM judge) | Whether the answer **adheres to** `samples[j].preference`, given that this preference is not stated in the prompt |
| Judge inputs | gold answer, model answer | `preference`, `explanation`, `question`, model answer |
| Judge implementation | Module-side or shared eval script | **Deferred** — the runner just produces JSONL with the model answers and ground-truth fields. The user runs the judge separately later |
| Re-judging at multiple checkpoints | N/A | Yes — same `(question_session_j)` is judged once per checkpoint `m_i ≥ m_j`. Do not deduplicate |
| QA answer max length | 100 words / `MAX_TOKENS=750` | **200 words / `MAX_TOKENS=1500`** — PrefEval answers are advisory and need more headroom |

The QA prompt template is otherwise reused verbatim (only the word cap line
changes from "100 words" to "200 words").

---

## 6. CLI arguments

| | ImplexConv | PrefEval |
|---|---|---|
| Session selection | `--start-session N --end-session M` (range) | `--end-session K` (chain is `samples[0..K]`) |
| Subset | `--subset opposed/supportive` | Removed |
| Cross-session batching | `--batch-size N` | Removed (chain count is always 1) |
| Model / engine flags | `--model`, `--tensor-parallel`, `--gpu-memory`, `--max-model-len`, `--config` | Same |

---

## 7. Batching

| | ImplexConv | PrefEval |
|---|---|---|
| Cross-session memory build | Batched across N parallel sessions | N/A (chain = 1). Use only the module's intra-session machinery |
| QA generation | Possibly batched within a session | **At each checkpoint `k`**, batch all `q_0..q_k` together (chunked by `QA_BATCH_SIZE` for large `k`) |
| Judge | Often batched | Same. All `(k, j)` judge calls are independent |
| Memory build within a session | Sequential | Sequential (preserves evolution semantics — see Q3 in design notes) |

---

## 8. Output

| | ImplexConv | PrefEval |
|---|---|---|
| Output dir | `config_<n>_outputs_<model>_<subset>/session_<start>_<end>/` | `config_<n>_outputs_<model>/` (flat — no subset, no session range, no chain-K subdir) |
| Per-QA log | One JSON file per session range | **Single `results.jsonl`**, append-only, one row per `(k, question_session)` pair |
| Retrieval log | Per-session JSONL | **Single `retrieval_log.jsonl`**, append-only, with `k` field |
| Memory snapshots | Typically only the final state | **Save every checkpoint** `m_0, ..., m_K` under `memory_snapshots/m_<k>/`. Format is module-specific |
| Token / module stats | Per-session | **`stats.json`** cumulative across all runs (resume-safe) |
| Config metadata | Embedded per session result | **`meta.json`** written once on first run |

---

## 9. Resume behavior (PrefEval-only)

ImplexConv's runner uses an explicit checkpoint file (`completed_session_ids`)
to resume mid-range. PrefEval is simpler:

- **Source of truth**: existing `memory_snapshots/m_*/` directories. The
  largest `k` present is `k_existing`.
- **Smaller K rerun** (`K_target ≤ k_existing`): no-op, log "already complete".
- **Larger K rerun** (`K_target > k_existing`): load `m_{k_existing}` via
  `load_snapshot`, continue ingestion from `k_existing + 1`, append to all
  output files.
- **Fresh K rerun** (no existing snapshots): start from `k=0`.

Resume only works if config (model, RETRIEVE_K, prompt templates, etc.) is
unchanged between runs. Different models naturally land in different output
directories because the dir name embeds the model.

---

## 10. Things that DO NOT change

These can be lifted from the ImplexConv module verbatim:

- LLM client wrappers (model invocation code, retry/timeout logic)
- Module-internal data structures (event lists, memory graphs, A-MEM notes /
  retriever, etc.)
- The module's prompt templates for ingestion / retrieval / answering, **except**
  for the QA-answer word cap (100 → 200 words) and any subset-specific schema
  branches (drop them)
- `format_virtual_time`, `compute_virtual_seconds` and other time helpers
- Module hyperparameters and config files (start from a copy, then change
  `MAX_TOKENS` to 1500, dataset paths, and remove subset)

---

## 11. Quick port checklist

1. Copy the module folder from `../exp_implexconv_no_response/<module>/` to
   `exp_prefeval/<module>/`.
2. Rewrite `load_dataset.py` to read `dataset/implicit_persona.json` and yield
   `Session`-shaped objects (re-using ImplexConv's `Session` dataclass shape).
   - Reorder turns to `user → assistant`.
   - Populate `session_id=0`, `conv_id=chain_pos`, `turn_id=local utterance
     index`, `global_turn_id=chain-cumulative count`.
3. In `run_experiment.py`:
   - Replace the per-session loop with the cumulative-chain loop.
   - Single in-memory instance; `save_snapshot` per checkpoint.
   - At each checkpoint, batch QA over `q_0..q_k`, append JSONL.
   - Save snapshot to `memory_snapshots/m_<k>/`.
   - Implement resume by scanning snapshot dirs (§9).
4. Strip any persona side-channel and any subset branching.
5. In `config_<n>.py`:
   - Drop `DATASET_OPPOSED` / `DATASET_SUPPORTIVE`; add a single dataset path.
   - Change `MAX_TOKENS = 1500`.
   - Update output-path helpers (no subset, no session range).
6. In QA prompt template: change word cap from 100 to 200 words.
7. CLI: `--end-session K`. Drop `--start-session`, `--end-session` (range),
   `--subset`, `--batch-size`.
