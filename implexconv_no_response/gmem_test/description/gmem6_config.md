# GraphMem: Configuration

> **Scope**: Consolidated hyperparameters only.

---

## 1. Extraction

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `STATE_EXTRACTION_H` | 1 | state extraction interval in user turns (per-turn extraction) |
| `STATE_MAX_COUNT` | 1 | max number of states produced by one ② call |
| `STATE_REF_CONTEXT_TURNS` | 3 | prior `(user, assistant)` pairs shown as read-only reference in ② (ignores `conv_id` boundary) |
| `STATE_NEW_REL_PREV_WINDOW` | 5 | number of global recent states (by `created_at` desc, excluding newly extracted) used as prev side in ②b |
| `CHUNK_SIZE_CONV` | 1 | chunk size in `conv_id` units |
| `TRAIT_EXTRACTION_CHUNKS` | 2 | trait extraction interval in chunks |
| `TRAIT_MAX_COUNT` | 1 | max number of traits produced by one ④ call |
| `MAX_KEYWORDS` | 7 | max number of extracted keywords |
| `MAX_DOMAIN_LABELS` | 7 | max number of domain labels per node |
| `MIN_DOMAIN_LABELS` | 5 | min number of domain labels per node. When fewer than this survive validation, generic `"general"` / `"general_N"` fillers are appended (see `gmem6_implementation.md` §3). |

Notes:
- `STATE_EXTRACTION_H = 1` and `STATE_MAX_COUNT = 1`: ② runs every user turn and produces at most one state per call. ② is extraction-only; all new-state relation judgments are produced by ②b. See `gmem6_storage_extraction.md` §6/§6b and §13 below (`JUDGMENT_RETRY`).
- `STATE_REF_CONTEXT_TURNS` controls only the prompt-side reference window in ②. It does not affect storage, the context cache, or any other call.
- **Change 3 (gmem6) — single-token labels**: every entry of `keywords` and `domain_label` must be a single token (no whitespace). Enforced by the extraction prompt and by `_validate_keywords` / `_validate_domain_labels` (items with whitespace are silently dropped). No new parameter; the count parameters `MAX_KEYWORDS`, `MIN_DOMAIN_LABELS`, `MAX_DOMAIN_LABELS` are unchanged.

---

## 1b. Pool Seeding (Change 2 / gmem6)

After Step 1 of graph expansion (context → SOURCE-derived nodes), for each node in `s_seed` and `m_seed` the retriever pulls S/M neighbors whose `(conv_id, turn_id)` differs by exactly ±1 turn within the **same `conv_id`** into the pool. These nodes are added as plain pool members; they are **not** added to `origin_ids`. The helper is `HeterogeneousGraph.get_turn_neighbors(node_id, types, delta=1)`. No tunable parameter; `delta` is fixed at 1. See `gmem6_retrieval.md` §3.2.

---

## 2. Seed Retrieval — Top-k

| Parameter (formula symbol → code identifier) | Recommended | Description |
|-----------|-------------|-------------|
| `k_c` (`K_CONTEXT`) | 20 | seed context count |
| `k_e` (`K_EPISODE`) | 6 | seed episode count |
| `k_s` (`K_STATE`) | 17 | seed state count |
| `k_t` (`K_TRAIT`) | 6 | seed trait count |
| `k_aps` (`K_APS`) | 6 | max active persona set size |

---

## 3. Seed Retrieval — Scoring Weights

All seed scores use two components: semantic similarity (`sem`, [0,1]) and query-side normalized overlap (`overlap_norm`, [0,1]). Weights sum to 1.0.

### 3.1 Context

Context nodes do not have `scope` and use a fixed weight pair.

| Parameter (formula → code identifier) | Recommended | Description |
|-----------|-------------|-------------|
| `w_sem_c` (`W_SEM_C`) | 0.65 | semantic similarity weight |
| `w_ov_c` (`W_OV_C`) | 0.35 | overlap weight |

### 3.2 State / Episode / Trait

States, episodes, and traits all use the same scope-dependent weight pair. When a node has `scope = None` (no explicit scope), the **BROAD** pair is applied as a fallback.

**NARROW**
| Parameter (formula → code identifier) | Recommended | Description |
|-----------|-------------|-------------|
| `w_sem_narrow` (`W_SEM_NARROW`) | 0.60 | semantic similarity weight |
| `w_ov_narrow` (`W_OV_NARROW`) | 0.40 | overlap weight |

**BROAD**
| Parameter (formula → code identifier) | Recommended | Description |
|-----------|-------------|-------------|
| `w_sem_broad` (`W_SEM_BROAD`) | 0.85 | semantic similarity weight |
| `w_ov_broad` (`W_OV_BROAD`) | 0.15 | overlap weight |

---

## 4. Final Set Assembly

### 4.1 Final Set — Top-k

| Parameter (formula → code identifier) | Recommended | Description |
|-----------|-------------|-------------|
| `k_t_final` (`K_T_FINAL`) | 5 | number of traits kept in the final set |
| `k_sf` (`K_SF`) | 18 | number of relevant states kept in the final set (excluding APS) |
| `k_e_final` (`K_EPISODE_FINAL`) | 4 | number of relevant episodes kept in the final set |

### 4.2 Final Set — Scoring

| Parameter (formula → code identifier) | Recommended | Description |
|-----------|-------------|-------------|
| `w_sr` (`W_SR`) | 0.5 | support-ratio weight in final scoring; seed-score weight is `1 - w_sr` |
| `τ` (`TRAIT_VALIDATION_TAU`) | 0.7 | threshold for trait `evidence_ratio` (`SUP / (SUP + CON)`). Used together with the `SHIFT_TO` signal in trait classification (see `gmem6_retrieval.md` §4.3). |

---

## 5. Sign Propagation

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `SIGN_PROP_HOP_CAP` | 10 | max hop count for multi-hop sign propagation |

Notes:
- CON terminates traversal immediately (no multi-hop CON propagation);
- **Change 4 (gmem6) — `SHIFT_TO` is unified into sign propagation**. Outgoing `SHIFT_TO` is emitted as a SUP-typed edge (continue to the newer version); incoming `SHIFT_TO` is emitted as a CON-typed edge (record the older version, then halt). The dedicated forward BFS and the final-set bidirectional bonus from gmem5 are removed; the same signal is now carried inside `signed_cache`. See `gmem6_retrieval.md` §3.3 / §3.4;
- **Change 6 (gmem6) — multi-path tie resolution**: shortest path wins across hop levels; among equal-length paths, **CON wins on ties** (gmem5 had SUP wins on ties). See `gmem6_retrieval.md` §3.6;
- **Change 7 (gmem6) — caching**: `get_ordinary_signed_reachable(start_id, hop_cap)` is memoized per-`retrieve()` inside `GraphRetriever`. No tunable parameter; engineering only.

---

## 6. Active Persona Set

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `k_aps` (`K_APS`) | 6 | size of the APS top-k slot (see §2). Top-`k_aps` HIGH recall-priority, non-`SHIFT_TO`-source states ranked by `seed_score`. APS members are excluded from the `s_seed` candidate pool. **Change 1 (gmem6)** — APS members remain in the post-seed pool as evidence but are **excluded from `origin_ids` during graph expansion**. |
| `APS_EXCLUDE_SHIFT_SOURCE` | `True` | exclude `SHIFT_TO` sources from APS candidacy |
| `ENABLE_SHIFT_CHAIN_PRUNING` | `True` | enable shift-chain collapse on `t_final`, `s_final`, and `s_aps` |
| `STRICT_HIGH_DEFAULT_LOW` | `True` | require conservative assignment of `recall_priority = HIGH`; default to `LOW` when uncertain |

Notes:
- APS membership is `recall_priority = HIGH` only; `scope` is **not** a membership criterion (it still drives seed-score weights). See `gmem6_retrieval.md` §2.7.

---

## 7. Additional Relation Extraction

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `ENABLE_EXTRA_RELATION_EXTRACTION` | `True` | master switch for ⑤b/⑤c/⑤d |
| `EXTRA_REL_ONLY_IF_UNCONNECTED` | `True` | judge only pairs with no existing direct evidence edge |
| `TRAIT_EXTRA_REL_TOPK_STATE` | 7 | per-list `k` for ⑤b state candidates (semantic top-`k` and lexical top-`k` each fetched separately, then unioned) |
| `TRAIT_EXTRA_REL_TOPK_EPISODE` | 3 | per-list `k` for ⑤b episode candidates (same scheme) |
| `STATE_STATE_EXTRA_REL_TOPK` | 5 | per-list `k` for the pair-level `sem_topK ∪ lex_topK` selection over the ⑤c candidate pair pool (at most `2k = 10` pairs per ⑤c call) |
| `STATE_EPISODE_EXTRA_REL_TOPK` | 3 | per-list `k` for the pair-level `sem_topK ∪ lex_topK` selection over the ⑤d candidate pair pool (at most `2k = 6` pairs per ⑤d call) |

Notes:
- ⑤b uses single-anchor candidate selection: `sem_topk(anchor) ∪ lex_topk(anchor)` with `node_id`-level dedup, no rerank, no post-union cap. See `gmem6_storage_extraction.md` §9.2 / §9.3.
- ⑤c/⑤d use pair-level selection: between triggers, only the IDs of newly added states (and episodes, for ⑤d) accumulate in pending sets; at the next ⑤c/⑤d trigger, the candidate pair pool (unconnected pairs touching at least one pending new node) is reduced by `sem_topK ∪ lex_topK` over pairs. See `gmem6_storage_extraction.md` §9.4 / §9.5.

---

## 8. Pair Similarity for Additional Candidate Mining

(Removed.) `pair_score`, `w_pair_sem`, and `w_pair_lex` are no longer used —
⑤b/⑤c/⑤d compute semantic and lexical rankings independently and union
them. The two config keys `W_PAIR_SEM` / `W_PAIR_LEX` may still exist in
`config_0.py` but are dead.

---

## 9. Context Cache

| Parameter (formula → code identifier) | Recommended | Description |
|-----------|-------------|-------------|
| `k0` (`CONTEXT_CACHE_SIZE`) | 10 | number of recent `(user, gt_response)` pairs kept outside the graph |

---

## 10. QA / Response Serialization

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `INCLUDE_RECENT_CONVERSATION_FOR_QA` | `True` | include recent conversation in QA serialization. **Current default is `True` (on).** |
| `QA_CONTEXT_PAIRS` | 5 | number of most-recent (user, agent) pairs included in the QA prompt's `[Recent Conversation]` block; non-QA retrieval still sees the full cache |

---

## 11. Timestamp

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `TIME_PER_CONV_ID_HOURS` | 12 | hours per `conv_id` increment |
| `TIME_PER_TURN_MINUTES` | 10 | minutes per `turn_id` increment |
| `CONV_IDS_PER_DAY` | 2 | conv-id count per day (auxiliary alias used by some helpers) |
| `MINUTES_PER_TURN` | 10 | minutes per turn (auxiliary alias, equal to `TIME_PER_TURN_MINUTES`) |

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
| `MAX_TOKENS_EPISODE` | 1000 | max output tokens for ③ (episode extraction) |
| `MAX_TOKENS_EPISODE_NEW_REL` | 2000 | max output tokens for ③b (episode new-relation judgments) |
| `MAX_TOKENS_TRAIT` | 1200 | max output tokens for ④ (trait extraction) |
| `MAX_TOKENS_TRAIT_EVIDENCE_5A` | 2500 | max output tokens for ⑤a (local trait evidence) |
| `MAX_TOKENS_TRAIT_EXTRA_REL_5B` | 1500 | max output tokens for ⑤b (trait extra relations) |
| `MAX_TOKENS_STATE_STATE_5C` | 1000 | max output tokens for ⑤c (state-state relations) |
| `MAX_TOKENS_STATE_EPISODE_5D` | 800 | max output tokens for ⑤d (state-episode relations) |
| `JSON_RETRY` | 5 | structured-output retry count (JSON parse failures) |
| `JUDGMENT_RETRY` | 3 | retry count when a judgments array is empty but `expected_judgment_count > 0` |
| `SAVE_MEMORY_SNAPSHOTS` | `True` | whether to save memory snapshots |
| `ENABLE_LLM_CALL_LOGGING` | `True` | master switch for per-call prompt/output JSONL logging |
| `LLM_CALL_LOG_FIRST_N_SESSIONS` | 10 | when logging is enabled, only sessions with `session_id < N` have their LLM calls persisted |

Notes:
- `JSON_RETRY` and `JUDGMENT_RETRY` are independent and stack: each `JUDGMENT_RETRY` attempt internally allows up to `JSON_RETRY` JSON parse retries.
- `JUDGMENT_RETRY` applies to all relation-extraction calls (②b, ③b, ⑤a, ⑤b, ⑤c, ⑤d). ② and ③ are extraction-only — their schemas have no `judgments` field — so the retry wrapper skips them.
- `IRRELEVANT` is a valid judgment, not an empty case. A judgments array fully populated with `IRRELEVANT` entries is **not** retried.
- After `JUDGMENT_RETRY` exhaustion, `apply_irrelevant_fallback` writes `IRRELEVANT` edges for every `(src, dst)` in `call.expected_pairs` (skipping missing nodes and pairs already directly connected). The call is considered complete, not failed.
- On each judgment-retry attempt (attempts 2..N), a one-line hint is appended to the user prompt: `"Previous attempt returned empty judgments; you MUST output exactly N judgments."` The original prompt is unchanged for the first attempt.
- Per-call formulas (`expected_judgment_count`, canonical `expected_pairs`) are listed in `gmem6_storage_extraction.md` §6b/§7b/§9 and `gmem6_prompt.md` §5/§7/§9–§12. Overall retry policy is in `gmem6_implementation.md` §5.

---

## 14. LLM Engine

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `LLM_ENGINE` | `"vllm"` | active LLM backend; `get_llm_config(engine)` dispatches between vLLM / Together / OpenAI |
| `DEFAULT_VLLM_CONFIG` | (dict) | vLLM config: `model_path`, `tensor_parallel_size`, `gpu_memory_utilization`, `download_dir` |
| `TOGETHER_CONFIG` | (dict) | Together backend: `model_name`, `api_key` (`TOGETHER_API_KEY` env) |
| `OPENAI_CONFIG` | (dict) | OpenAI backend: `model_name`, `api_key` (`OPENAI_API_KEY` env) |

---

## 15. Batch Sizes

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `BATCH_SIZE` | 4 | batch size for the batched extraction/judgment runner (see `run_experiment.py`) |
| `QA_BATCH_SIZE` | 128 | batch size for ⑥ (QA) calls |

---

## 16. Logging and Checkpointing

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `LOG_TO_FILE` | `True` | mirror stdout logger output to a file under `LOG_DIR` |
| `LOG_LEVEL` | `"INFO"` | logger level |
| `LOG_DIR` | `<config_dir>/logs` | logger output directory |
| `ENABLE_CHECKPOINTING` | `True` | persist a per-session checkpoint so the runner can resume |
| `CHECKPOINT_INTERVAL` | 1 | checkpoint write interval (in sessions) |

---

## 17. Common Ablations

- disable APS → set `k_aps = 0`
- disable APS shift filtering → set `APS_EXCLUDE_SHIFT_SOURCE = False`
- disable strict HIGH gating → set `STRICT_HIGH_DEFAULT_LOW = False`
- disable shift-chain collapse → set `ENABLE_SHIFT_CHAIN_PRUNING = False`
- disable extra relation extraction → set `ENABLE_EXTRA_RELATION_EXTRACTION = False`
- disable multi-hop sign propagation → set `SIGN_PROP_HOP_CAP = 1` (direct-only evidence)
- disable evidence-based scoring → set `w_sr = 0` (final score uses seed score only)
- disable seed-score contribution → set `w_sr = 1` (final score uses support ratio only)
- disable lexical overlap → set `w_ov_narrow = 0`, `w_ov_broad = 0`, `w_ov_c = 0`
