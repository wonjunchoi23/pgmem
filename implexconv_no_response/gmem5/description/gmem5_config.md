# GraphMem: Configuration

> **Scope**: Consolidated hyperparameters only.

---

## 1. Extraction

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `STATE_EXTRACTION_H` | 1 | state extraction interval in user turns (per-turn extraction) |
| `STATE_MAX_COUNT` | 1 | max number of states produced by one ② call |
| `STATE_REF_CONTEXT_TURNS` | 3 | prior `(user, assistant)` pairs shown as read-only reference in ② (ignores `conv_id` boundary) |
| `STATE_NEW_REL_PREV_WINDOW` | 3 | number of global recent states (by `created_at` desc, excluding newly extracted) used as prev side in ②b |
| `CHUNK_SIZE_CONV` | 1 | chunk size in `conv_id` units |
| `TRAIT_EXTRACTION_CHUNKS` | 2 | trait extraction interval in chunks |
| `TRAIT_MAX_COUNT` | 1 | max number of traits produced by one ④ call |
| `MAX_KEYWORDS` | 5 | max number of extracted keywords |
| `MAX_DOMAIN_LABELS` | 5 | max number of domain labels per node |
| `MIN_DOMAIN_LABELS` | 3 | min number of domain labels per node |

Notes:
- `STATE_EXTRACTION_H = 1` and `STATE_MAX_COUNT = 1`: ② runs every user turn and produces at most one state per call. The judgments array of ② is therefore always empty by construction (no `new ↔ new` pair); the caller must skip judgment retry for ② (`expected_judgment_count = 0`). See §13 (`JUDGMENT_RETRY`) and `gmem5_implementation.md` §4.1.
- `STATE_REF_CONTEXT_TURNS` controls only the prompt-side reference window in ②. It does not affect storage, the context cache, or any other call.

---

## 2. Seed Retrieval — Top-k

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `k_c` | 20 | seed context count |
| `k_m` | 6 | seed memory count |
| `k_s` | 17 | seed state count |
| `k_t` | 6 | seed trait count |
| `k_aps` | 5 | max active persona set size |

---

## 3. Seed Retrieval — Scoring Weights

All seed scores use two components: semantic similarity (`sem`, [0,1]) and query-side normalized overlap (`overlap_norm`, [0,1]). Weights sum to 1.0.

### 3.1 Context

Context nodes do not have `scope` and use a fixed weight pair.

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `w_sem_c` | 0.65 | semantic similarity weight |
| `w_ov_c` | 0.35 | overlap weight |

### 3.2 State / Memory / Trait

States, memories, and traits all use the same scope-dependent weight pair.

**NARROW**
| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `w_sem_narrow` | 0.60 | semantic similarity weight |
| `w_ov_narrow` | 0.40 | overlap weight |

**BROAD**
| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `w_sem_broad` | 0.85 | semantic similarity weight |
| `w_ov_broad` | 0.15 | overlap weight |

---

## 4. Final Set Assembly

### 4.1 Final Set — Top-k

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `k_t_final` | 4 | number of traits kept in the final set |
| `k_sf` | 12 | number of relevant states kept in the final set (excluding APS) |
| `k_m_final` | 6 | number of relevant memories kept in the final set |

### 4.2 Final Set — Scoring

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `w_sr` | 0.5 | support-ratio weight in final scoring; seed-score weight is `1 - w_sr` |
| `τ` | 0.7 | threshold for stable vs. challenged trait classification (within selected top-k traits) |

---

## 5. Sign Propagation

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `SIGN_PROP_HOP_CAP` | 10 | max hop count for multi-hop sign propagation |

Notes:
- CON terminates traversal immediately (no multi-hop CON propagation);
- `SHIFT_TO` is not used during expansion sign propagation. During support-ratio computation, `SHIFT_TO(A → B)` contributes **bidirectionally**: A receives `con_w += 1` (penalized as outdated), B receives `sup_w += 1` (boosted as current). See `gmem5_retrieval.md` §3.5 / §4.

---

## 6. Active Persona Set

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `k_aps` | 5 | size of the APS top-k slot (see §2). Top-`k_aps` HIGH-impact, non-`SHIFT_TO`-source states ranked by `seed_score`. APS members are excluded from the `s_seed` candidate pool. |
| `APS_EXCLUDE_SHIFT_SOURCE` | `True` | exclude `SHIFT_TO` sources from APS candidacy |
| `ENABLE_SHIFT_CHAIN_PRUNING` | `True` | enable shift-chain collapse on `t_final`, `s_final`, and `s_aps` |
| `STRICT_HIGH_DEFAULT_LOW` | `True` | require conservative assignment of `current_decision_impact = HIGH`; default to `LOW` when uncertain |

Notes:
- APS membership is `current_decision_impact = HIGH` only; `scope` is **not** a membership criterion (it still drives seed-score weights). See `gmem5_retrieval.md` §2.7.

---

## 7. Additional Relation Extraction

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `ENABLE_EXTRA_RELATION_EXTRACTION` | `True` | master switch for ⑤b/⑤c/⑤d |
| `EXTRA_REL_ONLY_IF_UNCONNECTED` | `True` | judge only pairs with no existing direct evidence edge |
| `TRAIT_EXTRA_REL_TOPK_STATE` | 7 | post-union cap for extra trait-to-state candidates in ⑤b |
| `TRAIT_EXTRA_REL_TOPK_MEMORY` | 2 | post-union cap for extra trait-to-memory candidates in ⑤b |
| `STATE_STATE_EXTRA_REL_TOPK` | 5 | size of the global unconnected state-state candidate reservoir, and number consumed in ⑤c |
| `STATE_MEMORY_EXTRA_REL_TOPK` | 2 | size of the global unconnected state-memory candidate reservoir, and number consumed in ⑤d |

Notes:
- in ⑤b, semantic and lexical candidate lists are merged, deduplicated, reranked, then capped;
- in ⑤c/⑤d, top-k refers to the current global unconnected candidate reservoir.

---

## 8. Pair Similarity for Additional Candidate Mining

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `w_pair_sem` | 0.7 | pairwise semantic similarity weight |
| `w_pair_lex` | 0.3 | pairwise lexical overlap weight |

---

## 9. Context Cache

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `k0` | 10 | number of recent `(user, gt_response)` pairs kept outside the graph |

---

## 10. QA / Response Serialization

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `INCLUDE_RECENT_CONVERSATION_FOR_QA` | `True` | include recent conversation in QA serialization |
| `QA_CONTEXT_PAIRS` | 5 | number of most-recent (user, agent) pairs included in the QA prompt's `[Recent Conversation]` block; non-QA retrieval still sees the full cache |

---

## 11. Timestamp

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `TIME_PER_CONV_ID_HOURS` | 12 | hours per `conv_id` increment |
| `TIME_PER_TURN_MINUTES` | 10 | minutes per `turn_id` increment |

---

## 12. Embedding and NLP

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline used for noun extraction |

---

## 13. Shared Protocol Constants

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `TIMING_CONV_ID` | 0 | `conv_id` used for timing measurement |
| `TEMPERATURE` | 0.7 | LLM generation temperature |
| `MAX_TOKENS` | 1000 | max output tokens for ⑥ (QA) |
| `MAX_TOKENS_STATE` | 1000 | max output tokens for ② (state extraction) |
| `MAX_TOKENS_STATE_NEW_REL` | 800 | max output tokens for ②b (state new-relation judgments) |
| `MAX_TOKENS_MEMORY` | 1000 | max output tokens for ③ (memory extraction) |
| `MAX_TOKENS_MEMORY_NEW_REL` | 2000 | max output tokens for ③b (memory new-relation judgments) |
| `MAX_TOKENS_TRAIT` | 1200 | max output tokens for ④ (trait extraction) |
| `MAX_TOKENS_TRAIT_EVIDENCE_5A` | 2500 | max output tokens for ⑤a (local trait evidence) |
| `MAX_TOKENS_TRAIT_EXTRA_REL_5B` | 1500 | max output tokens for ⑤b (trait extra relations) |
| `MAX_TOKENS_STATE_STATE_5C` | 1000 | max output tokens for ⑤c (state-state relations) |
| `MAX_TOKENS_STATE_MEMORY_5D` | 800 | max output tokens for ⑤d (state-memory relations) |
| `JSON_RETRY` | 3 | structured-output retry count (JSON parse failures) |
| `JUDGMENT_RETRY` | 3 | retry count when a judgments array is empty but `expected_judgment_count > 0` |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | whether to save memory snapshots |
| `ENABLE_LLM_CALL_LOGGING` | `True` | master switch for per-call prompt/output JSONL logging |
| `LLM_CALL_LOG_FIRST_N_SESSIONS` | 10 | when logging is enabled, only sessions with `session_id < N` have their LLM calls persisted |

Notes:
- `JSON_RETRY` and `JUDGMENT_RETRY` are independent and stack: each `JUDGMENT_RETRY` attempt internally allows up to `JSON_RETRY` JSON parse retries.
- `JUDGMENT_RETRY` applies to all relation-extraction calls (②b, ③, ③b, ⑤a, ⑤b, ⑤c, ⑤d). ② is excluded because its `new ↔ new` pair set is always empty under `STATE_MAX_COUNT = 1`.
- `IRRELEVANT` is a valid judgment, not an empty case. A judgments array fully populated with `IRRELEVANT` entries is **not** retried.
- After `JUDGMENT_RETRY` exhaustion the call falls back: all expected pairs are treated as `IRRELEVANT` (no edges created); the call is considered complete, not failed.
- On each judgment-retry attempt (attempts 2..N), a one-line hint is appended to the user prompt: `"Previous attempt returned empty judgments; you MUST output exactly N judgments."` The original prompt is unchanged for the first attempt.
- See `gmem5_implementation.md` §4 (per-call) and §7 (overall policy).

---

## 14. Common Ablations

- disable APS → set `k_aps = 0`
- disable APS shift filtering → set `APS_EXCLUDE_SHIFT_SOURCE = False`
- disable strict HIGH gating → set `STRICT_HIGH_DEFAULT_LOW = False`
- disable shift-chain collapse → set `ENABLE_SHIFT_CHAIN_PRUNING = False`
- disable extra relation extraction → set `ENABLE_EXTRA_RELATION_EXTRACTION = False`
- disable multi-hop sign propagation → set `SIGN_PROP_HOP_CAP = 1` (direct-only evidence)
- disable evidence-based scoring → set `w_sr = 0` (final score uses seed score only)
- disable seed-score contribution → set `w_sr = 1` (final score uses support ratio only)
- disable lexical overlap → set `w_ov_narrow = 0`, `w_ov_broad = 0`, `w_ov_c = 0`
