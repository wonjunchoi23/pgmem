# GraphMem: Implementation Notes

> **Scope**: Runtime and implementation conventions only. Schema and extraction logic are defined in `gmem6_storage_extraction.md`; retrieval semantics are defined in `gmem6_retrieval.md`; prompt templates and node-listing formats are defined in `gmem6_prompt.md`; tunable parameters are defined in `gmem6_config.md`.

---

## 1. Execution Flow

```text
process_turn(user_utterance, gt_response, conv_id, turn_id, session_id, measure_timing=False)

  if conv_id changed and previous chunk exists:
      ③ Episode extraction (extraction-only)
      if (|chunk_state_ids| ≥ 1) or (previous_episode exists):
          ③b New-episode relation judgments (new↔chunk_states + new↔prev_episode)
      if 2-chunk boundary:
          ④ Trait extraction
          if new trait exists:
              ⑤a Local trait evidence judgment
              ⑤b Trait-centered extra relation extraction  [ENABLE_EXTRA_RELATION_EXTRACTION only]
          ⑤c Extra state-state relation extraction         [ENABLE_EXTRA_RELATION_EXTRACTION only; always, regardless of new trait]
          ⑤d Extra state-episode relation extraction       [ENABLE_EXTRA_RELATION_EXTRACTION only; always, regardless of new trait]

  create ContextNode
  retrieval → serialization
  update context cache
  # Note: this variant is QA-only — there is no ① Response prompt call during process_turn.
  # The retrieval-serialized context is reused later by ⑥ QA.

  if user_turn_count % STATE_EXTRACTION_H == 0:    # STATE_EXTRACTION_H = 1: every user turn
      ② State extraction (extraction-only, ≤ STATE_MAX_COUNT states)
      if (|new_state_ids| ≥ 2) or (|new_state_ids| ≥ 1 and |previous_state_ids| ≥ 1):
          ②b New-state relation judgments (new↔new + new↔prev)
```

`finalize_chunk()` must be called before Phase 2 so the last chunk runs ③ and, when applicable, ④-⑤d.

---

## 2. Serialization

The retrieval object serialized by `GraphRetriever` (sections `[Current
Constraints]`, `[Traits]`, `[Challenged Traits]`, `[Relevant States]`,
`[Relevant Episodes]`, `[Recent Conversation]`) is consumed by ⑥ QA in this
QA-only variant. See `gmem6_prompt.md` §3.4 for the full section layout and
`gmem6_retrieval.md` §6 for which pool produces each section.

`[Recent Conversation]` is included in the QA serialization when
`INCLUDE_RECENT_CONVERSATION_FOR_QA = True`. The current default in
`config_0.py` is `True` (on).

---

## 3. Prompt Templates

Prompt templates (system + user) for every call live in `gmem6_prompt.md`.
Per-call inputs, triggers, outputs, and apply-side rules are covered in
`gmem6_storage_extraction.md` §6–§9. This document does not duplicate them.

---

## 4. Token / Call Accounting

| # | Call | Trigger | Token category | Executed? |
|---|------|---------|----------------|-----------|
| ① | Response prompt construction | — | — | NO (this variant is QA-only; no ① call exists in code) |
| ② | State extraction (extraction-only, no judgments) | every user turn (`STATE_EXTRACTION_H = 1`) | internal | YES |
| ③ | Episode extraction (extraction-only, no judgments) | chunk boundary | internal | YES |
| ③b | New-episode relations (e↔chunk_states + e↔prev_episode) | chained after ③ when judgeable pair exists | internal | YES |
| ④ | Trait extraction | 2-chunk boundary | internal | YES |
| ⑤a | Local trait evidence | after ④, new trait only | internal | YES |
| ⑤b | Trait-centered extra relation extraction | after ⑤a, new trait only, ENABLE_EXTRA_RELATION_EXTRACTION | internal | YES |
| ⑤c | Extra state-state | every 2-chunk boundary, ENABLE_EXTRA_RELATION_EXTRACTION (regardless of new trait) | internal | YES |
| ⑤d | Extra state-episode | after ⑤c, ENABLE_EXTRA_RELATION_EXTRACTION | internal | YES |
| ⑥ | QA answering | per QA question | qa | YES |

---

## 5. Runtime Conventions

Sign propagation:
- `SHIFT_TO` is stored only as `old → new`; no reverse edge.
- `_ordinary_evidence_neighbors` folds `SHIFT_TO` into the signed traversal with a same-type / cross-type split (see `graph_store.py:_ordinary_evidence_neighbors`):
  - **Outgoing `SHIFT_TO`**: same-type → SUP (flow forward to the newer node); cross-type → CON (the source invalidates the cross-type target).
  - **Incoming `SHIFT_TO`** (same- or cross-type) → CON (the newer node halts the older one under the SUP×CON → CON-stop rule).
  - Ordinary evidence edges (`SUPPORT`, `CONTRADICT`, `IRRELEVANT`) follow the per-node-type outgoing-direction whitelist `_EVID_EXPAND_OUT` (C: ∅; E → {S, E, T}; S → {S, T}; T → {T}). Cross-type incoming ordinary edges are ignored — only same-type incoming ordinary edges are followed back. The dedicated forward BFS and the bidirectional bonus block from earlier revisions are removed; the same signal is carried entirely by `signed_cache`.
- `IRRELEVANT` is propagated as a third sign in the same traversal: it does not act like SUP and does not produce a SUP×CON match (see `signed_cache` users in `retriever.py`).
- CON terminates sign propagation immediately (no multi-hop CON propagation).
- Among multiple paths between two nodes, the shortest path wins; on equal length, **CON beats SUP**.
- Per-origin BFS dedup: within a single `get_ordinary_signed_reachable(start)` call a node is visited at most once; visited sets are not shared across origins.
- `get_ordinary_signed_reachable(start_id, hop_cap)` is memoized inside `GraphRetriever` for the duration of one `retrieve()` and cleared at the top of every call (safe because the graph is immutable during retrieval).

Scoring / final-set:
- All node types use the same final score: `w_sr · support_ratio + (1 - w_sr) · seed_score`.
- Evidence flows under the directional constraints listed above (`_EVID_EXPAND_OUT` plus the same-type-only incoming rule for ordinary edges) — not symmetrically across all type pairs.
- A node with no evidence falls back to `support_ratio = 0.5`.
- `_shift_chain_collapse` is applied to `t_final`, `s_final`, and `s_aps` (drop nodes that have a forward `SHIFT_TO` to another node in the same set).

Extra relation extraction (⑤b / ⑤c / ⑤d):
- ⑤a / ⑤b run only when ④ produced a new trait at the 2-chunk boundary.
- ⑤c / ⑤d run at every 2-chunk boundary regardless of whether a new trait was produced.
- Between triggers, the updater only tracks IDs of newly added states (and episodes, for ⑤d) in pending sets; no per-anchor candidate fetch is performed at ingestion time.
- At each ⑤c / ⑤d trigger, the candidate pair pool is enumerated from those pending IDs vs. all compatible nodes (unconnected only) and reduced by a pair-level `sem_topK ∪ lex_topK`. The pending sets are cleared at consumption time and refill incrementally until the next trigger.
- When `ENABLE_EXTRA_RELATION_EXTRACTION = False`, ⑤b/⑤c/⑤d and pending-set maintenance are all skipped.

APS construction (runtime):
- APS is constructed at retrieval time as the top-`k_aps` HIGH recall-priority, non-`SHIFT_TO`-source states ranked by `seed_score`. APS members are excluded from `s_seed`; HIGH states beyond rank `k_aps` fall through to `s_seed`.
- APS construction does not look at `scope` (which only affects seed-score weights).
- APS members are added to the post-seed pool but **excluded from `origin_ids`** — they contribute evidence to other nodes through `_cross_type_sup_con` but do not drive sign-propagation expansion.
- ±1 turn neighbors of `s_seed` and `e_seed` (same `conv_id`, `|Δturn| = 1`) are added to the pool as plain members, never as origins. Helper: `HeterogeneousGraph.get_turn_neighbors(node_id, types, delta=1)`.

Judgment-retry policy (`JUDGMENT_RETRY`):
- For every relation-extraction call the wrapper computes `expected_judgment_count` and `call.expected_pairs: List[Tuple[str, str]]` at build time.
- If `expected > 0` and the LLM returns an empty `judgments` array, retry up to `JUDGMENT_RETRY` times. On retry attempts (2+), the hint `Previous attempt returned empty judgments; you MUST output exactly N judgments.` is appended to the user prompt.
- Both the sequential path (`updater.py:execute_call`) and the batched path (`run_experiment.py:_drain_pending_calls`) implement the empty-judgment retry.
- **IRRELEVANT-fallback wiring differs by path.** After retry exhaustion, the batched runner calls `apply_irrelevant_fallback(call)` (see `run_experiment.py:_drain_pending_calls`), which writes `IRRELEVANT` edges for every `(src, dst)` in `call.expected_pairs` (skipping missing nodes and pairs that already have a direct edge). The sequential `GraphMemModule.process_turn` path only loops `execute_pending_call` → `apply_pending_call` and **does not invoke the fallback**; an exhausted empty-judgment call therefore leaves `expected_pairs` unwritten in that path. `GraphMemModule.apply_irrelevant_fallback` is exposed for callers that want this behavior.
- `IRRELEVANT` is a valid judgment; a non-empty all-`IRRELEVANT` array is **not** retried.
- `JSON_RETRY` and `JUDGMENT_RETRY` stack: each judgment-retry attempt internally allows up to `JSON_RETRY` JSON-parse retries.

Label hygiene:
- `deduplicate_labels(keywords, domain_label)` runs on every node-creation path (state, episode, trait). Case-insensitive; `keywords` win on overlap.
- `_validate_keywords` / `_validate_domain_labels` silently drop any item containing whitespace (single-token contract; see `gmem6_storage_extraction.md` §4.6).
- After whitespace/duplicate/keyword-overlap drops, if fewer than `MIN_DOMAIN_LABELS` items survive, `_validate_domain_labels` appends generic fillers `"general"`, `"general_2"`, `"general_3"`, ... until `MIN_DOMAIN_LABELS` is reached. Keyword backfill is intentionally **not** used because `HeterogeneousGraph.add_node` would strip the duplicate at storage time.

---

## 6. Interface Reminder

```python
result = module.process_turn(...)
module.finalize_chunk(...)
qa_result = module.get_qa_answer(question)
module.clear()
module.save_snapshot(directory)
module.load_snapshot(directory)
```

No generated response is written into memory. Ground-truth responses are always used for storage.
