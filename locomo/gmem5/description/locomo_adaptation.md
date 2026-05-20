# GraphMem v5 — LoComo Adaptation Notes

The other documents in this folder (`gmem5_implementation.md`, `gmem5_storage_extraction.md`, `gmem5_retrieval.md`, `gmem5_prompt.md`, `gmem5_config.md`, `gmem5_bigflow.md`, `difference.md`) are copied verbatim from the GraphMem v5 ImplexConv design. This file describes what changed when porting v5 to LoComo.

The memory module's algorithm — the ②/②b/③/③b/④/⑤a/⑤b/⑤c/⑤d call graph, the JUDGMENT_RETRY policy, the global reservoir for state-state and state-memory pair mining, the retrieval pipeline (seed → expansion → support-ratio scoring → final set assembly) — is unchanged. What changed is the experiment contract that drives the module.

---

## 1. Dataset and processing unit

| ImplexConv (v5) | LoComo (v5) |
|---|---|
| Session = unit of processing | Sample = unit of processing |
| `dataset/implexconv/ImplexConv_<subset>_processed.json` | `dataset/locomo10.json` |
| Flat conversation with `conv_id`, `turn_id`, `global_turn_id` | Multi-session sample with `session_N`, per-session `session_N_date_time`, per-turn `dia_id` |
| `speaker ∈ {user, assistant}` | named `speaker_a`, `speaker_b` |
| QA: `answer`, `retrieved_conv_ids`, optional `opposed_implicit_reasoning` | QA: `question`, `category`, `answer` for cat 1-4, `adversarial_answer` for cat 5, `evidence` |

Loader: `load_dataset.py` is shared with `exp_locomo/gmem4`. It applies image-caption normalization (`[Image: …]` prepended to the turn text) and exposes `Sample.qa[i].final_answer` (returns `adversarial_answer` for category 5, `answer` otherwise).

CLI / paths: `--start-session`/`--end-session` → `--start-sample`/`--end-sample`. Output dir is `{config}_outputs_{model}/sample_{start}_{end}/`. Checkpoint key is `completed_sample_ids` (sample-ID based), not session indices.

---

## 2. Replay loop

ImplexConv had a paired (user, assistant) turn loop with explicit assistant-response handling. LoComo does not — both speakers are observed participants. The driver iterates turns in original order:

```
for sample in samples:
    for session in sessions (chronological):
        for turn in session.turns:
            store turn (single speaker, single text)
```

Each turn enters the module via `process_turn_core(speaker, text, session_id, turn_idx, timestamp_seconds, dia_id)`. There is no `gt_response` and no User/Assistant role assumption.

---

## 3. Time model

ImplexConv used a virtual time scheme (`TIME_PER_CONV_ID_HOURS`, `TIME_PER_TURN_MINUTES`, `CONV_IDS_PER_DAY`, `MINUTES_PER_TURN`). LoComo uses real session datetimes parsed from `session_N_date_time`.

- `parse_session_datetime` parses the session string to epoch seconds.
- Per-turn timestamp is derived deterministically as `base_session_ts + turn_idx * MINUTES_PER_TURN_IN_SESSION * 60`.
- `format_elapsed_str(entry_timestamp_seconds, current_timestamp_seconds)` renders an absolute date string (e.g. `8 May 2023`). The retrieval serialization and prompt blocks use this string in place of the ImplexConv `[N min ago]` form.
- `format_session_gap(ts_a, ts_b)` (replaces `format_conv_gap`) is used by ⑤d's "Relation context" line. The same/different-session check uses `node.session_id` equality; "different sessions, {gap} apart" uses the timestamp difference.

`Node` carries an extra `timestamp_seconds: float`, plus `dia_id: str` and `speaker: str`. The legacy `conv_id`/`turn_id` fields are repurposed as aliases of `session_id`/`turn_idx` so that storage helpers, snapshots, and downstream tooling continue to work without renames.

---

## 4. Chunk boundary

The chunk boundary is the **session boundary**: when an incoming turn's `session_id` differs from the current chunk's `session_id`, ③ memory extraction fires for the just-closed chunk. Within a session, all turns belong to the same chunk regardless of length. This matches the gmem4 LoComo policy and confirms with the v5 design (chunks are still 1 conversation = 1 chunk; "1 conversation" maps to "1 session" in LoComo).

`prepare_finalize_call` is invoked at end-of-sample to flush the last open chunk through ③ (and onward if the chunk count hits the trait-extraction window).

---

## 5. State / Memory / Trait extraction prompts

The structure of ②, ③, ④ prompts is unchanged (definition + good/NOT examples + skip rule + scope/impact metadata + label-discipline block + data block). Surface adaptations:

- **②**: the prompt addresses the *current speaker* explicitly (`{speaker_label}`), not "the user". The state must "begin with `{speaker_label}`" rather than "begin with The user". The PRIOR CONTEXT block uses single-speaker turns (`[date] {speaker} says: {text}`) drawn from `ContextCache.get_prior_turns(...)`, ignoring session boundaries (most recent N prior turns).
- **③**: the prompt no longer says "begin with The user". The episodic memory summarizes *both speakers'* exchange in the chunk and is told to mention both speakers by name when both participated. The chunk-states block is still removed (it is delegated to ③b).
- **④**: extracts 0 or 1 trait about *one of the two speakers*. The schema includes a `speaker` field on each trait. The prompt instructs traits to "begin with the speaker's name". `TRAIT_MAX_COUNT = 1` is unchanged.

⑤a/⑤b/⑤c/⑤d prompts are unchanged in semantics. ⑤d's "Relation context" line now reads "same session" / "different sessions, {gap} apart" instead of "same conversation" / "different conversations".

---

## 6. QA layer

ImplexConv had subset-specific QA (free-text for `opposed`, yes/no for `supportive`). LoComo replaces this with category-aware QA (1/2/4 short phrase, 3 approximate date, 5 binary choice with deterministic A/B ordering).

- `GraphGenerator.build_qa_prompt(question, retrieved_memory, category, adversarial_answer, choice_order_seed)` returns `(prompt, system_prompt, schema, temperature)`.
- Categories 1, 2, 4 use a short-phrase prompt at `TEMPERATURE`. Category 3 uses a temporal prompt. Category 5 uses a binary-choice prompt at `TEMPERATURE_C5`.
- Category 5 ordering of `{adversarial_answer, "Not mentioned in the conversation"}` is determined by `hashlib.md5(seed)` where `seed = f"{sample_id}::{qa_idx}::{question}"`.
- The `[Current Constraints]`, `[Traits]`, `[Challenged Traits]` semantics from v5 are preserved in the system prompt; phrasing was updated from "user" to "speakers".
- Output schema aligned with ImplexConv v5 (2026-05-10): the `reasoning` field has been removed. Schema is `{answer}` for categories 1-4 and `{choice, answer}` for category 5. Token accounting still keys on `_usage`.
- All QA prompts repeat the `[Question]` block both above and below `[Retrieved Memory]` so the question stays in attention during long retrieval blocks (ported from ImplexConv v5 2026-05-10).

The retrieval serialization carries an absolute-date prefix on every line (e.g. `[8 May 2023] ...`), which the QA prompts explicitly call out for category 3 ("Use the date shown in brackets to answer with an approximate date").

---

## 7. Configuration changes

Removed (ImplexConv-specific):
- `DATASET_OPPOSED`, `DATASET_SUPPORTIVE`, `get_dataset_path(subset)`
- `TIME_PER_CONV_ID_HOURS`, `TIME_PER_TURN_MINUTES`, `CONV_IDS_PER_DAY`, `MINUTES_PER_TURN`
- `TIMING_CONV_ID`
- `--subset` CLI argument and all `subset` parameters

Added (LoComo):
- `DATASET_PATH = dataset/locomo10.json`
- `TEMPERATURE_C5` (lower, deterministic-leaning temperature for category 5)
- `MINUTES_PER_TURN_IN_SESSION` (deterministic per-turn offset within a session)
- `QA_CONTEXT_TURNS` (number of most-recent turns sliced from the rolling context cache when assembling the QA `[Recent Conversation]` block; non-QA retrieval still sees the full cache). ImplexConv uses `QA_CONTEXT_PAIRS` because it tracks (user, agent) pairs; LoComo tracks single-speaker turns, hence the unit difference.

Renamed:
- `LLM_CALL_LOG_FIRST_N_SESSIONS` → `LLM_CALL_LOG_FIRST_N_SAMPLES`. Logging is gated on the sample's index in the run's pending list, so only the first N samples persist per-call prompts/outputs under `prompt_log/`.

All other v5 knobs (`MAX_TOKENS_*`, `JUDGMENT_RETRY`, `STATE_EXTRACTION_H = 1`, `STATE_MAX_COUNT = 1`, `STATE_REF_CONTEXT_TURNS = 3`, `K_*`, `W_SR`, `STATE_NEW_REL_PREV_WINDOW`, `ENABLE_EXTRA_RELATION_EXTRACTION`, `APS_EXCLUDE_SHIFT_SOURCE`, `ENABLE_SHIFT_CHAIN_PRUNING`, etc.) are unchanged.

Path helpers are sample-range based:
- `get_sample_dir`, `get_results_file`, `get_checkpoint_file`, `get_retrieval_log_dir`, `get_memory_snapshots_dir`, `get_prompt_log_dir`, `get_merged_results_file` all take `(start_sample, end_sample, ...)` instead of `(subset, start_session, end_session, ...)`.

---

## 8. Result and log schema

- `sample_id`, not `session_id`, identifies a result record.
- `sample_range`, not `session_range`.
- Retrieval log entries use `sample_id` + `dia_id` (not `conv_id`/`turn_id`). The `module_specific.num_by_slot` block reflects v5's slot inventory (no `traits_tentative` — v5 only distinguishes stable vs challenged).
- `qa_results[i].evidence` is copied directly from the dataset.
- `qa_results[i].ground_truth_answer` uses `qa.final_answer` (handles category 5 normalization).
- `retrieved_memories[i]` includes `dia_id` and `speaker` alongside `session_id`/`conv_id`/`turn_id`/`node_type`.

Checkpoint format:
```json
{ "completed_sample_ids": ["conv-1", "conv-2", ...] }
```

---

## 9. Speaker prefix format

Throughout the module, turn lines are rendered as:

```
[<absolute date>] <speaker> says: <text>
```

This is used in:
- the rolling context cache (`ContextCache.get_formatted_context`)
- the ② PRIOR CONTEXT block
- the ③ Recent Conversation block
- the ④ Recent Two Conversations block
- the SOURCE-edge context node `content` (`{speaker} says: {text}`)

---

## 10. Removed components

- `run_corruption.py` is not ported to LoComo. The module is QA-only on the standard LoComo dataset.
- `analyze_snapshots*.ipynb` notebooks are not ported.
- `nohup/`, `logs/`, `config_*_outputs_*` runtime artefacts are not copied.

---

## 11. What did not change

The internal algorithm, including:

- ②/②b/③/③b/④/⑤a/⑤b/⑤c/⑤d call graph and apply-side dispatch rules
- per-call `expected_judgment_count` and `expected_pairs` for IRRELEVANT-fallback
- `JUDGMENT_RETRY` empty-judgment retry policy (sequential and batched)
- global state-state and state-memory reservoirs for ⑤c/⑤d
- pair similarity scoring (`W_PAIR_SEM`, `W_PAIR_LEX`)
- retrieval pipeline: seed retrieval (BROAD/NARROW scope-dependent weights, K_CONTEXT/K_MEMORY/K_STATE/K_TRAIT/K_APS), graph expansion (SUP-multi-hop + 1-hop CON + SHIFT_TO forward-only), support-ratio scoring with `W_SR`, top-k final selection, shift-chain collapse
- APS construction (HIGH-impact, non-SHIFT_TO-source states; partition disjoint from `s_seed`)
- `keywords` / `domain_label` disjointness enforcement on every node creation
- atomic checkpoint/result writes
- batched generation infrastructure with prefix caching enabled in vLLM
