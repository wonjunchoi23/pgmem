# GraphMem: Architecture Overview

> **Scope**: High-level overview only. This document describes module boundaries, main data flow, and component responsibilities. Detailed schema is in `gmem6_storage_extraction.md`, retrieval rules are in `gmem6_retrieval.md`, hyperparameters are in `gmem6_config.md`, prompt templates are in `gmem6_prompt.md`, and runtime conventions are in `gmem6_implementation.md`.

---

## Document Map

| Document | Content |
|----------|---------|
| **gmem6_bigflow.md** | overall structure, execution flow, component relationships |
| **gmem6_storage_extraction.md** | node/edge schema, label system, extraction triggers, evidence creation |
| **gmem6_retrieval.md** | seed retrieval, graph expansion, final-set assembly, serialization contract |
| **gmem6_config.md** | tunable hyperparameters |
| **gmem6_prompt.md** | prompt templates and node-listing formats |
| **gmem6_implementation.md** | prompt orchestration, runtime conventions, module interface |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                               GraphMemModule                                │
│                                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────────────┐  │
│  │ Context    │   │ Graph      │   │ Retriever  │   │ Generator          │  │
│  │ Cache      │   │ Store      │   │            │   │                    │  │
│  │            │   │            │   │ Seed+APS   │   │ ⑥ QA Answering     │  │
│  │ k₀ recent  │   │ Nodes:     │   │ Expansion  │   │   (QA-only variant │  │
│  │ pairs      │   │  c, e, s,t │   │ Final Set  │   │    — no ① Response │  │
│  │            │   │            │   │ Serialize  │   │    Prompt call)    │  │
│  │            │   │ Edges:     │   │            │   │                    │  │
│  │            │   │  SOURCE    │   │            │   │                    │  │
│  │            │   │  EVIDENCE  │   │            │   │                    │  │
│  └──────┬─────┘   └──────┬─────┘   └──────┬─────┘   └─────────┬──────────┘  │
│         │                │                │                   │             │
│  ┌──────┴────────────────┴────────────────┴───────────────────┴──────────┐  │
│  │                                Updater                                 │  │
│  │                                                                        │  │
│  │  ② State extraction (extraction-only)                                  │  │
│  │  ②b New-state relation judgments (new↔new + new↔prev, chained after ②)│  │
│  │  ③ Episode extraction (extraction-only)                              │  │
│  │  ③b New-episode relation judgments (e↔chunk_states + e↔prev_episode) │  │
│  │  ④ Trait extraction                                                   │  │
│  │  ⑤ Additional evidence expansion                                       │  │
│  │     ├─ ⑤a Local trait evidence judgment        (new trait only)       │  │
│  │     ├─ ⑤b Trait-centered extra relation extraction (new trait only)   │  │
│  │     ├─ ⑤c Global unconnected state-state pair mining  (always*)       │  │
│  │     └─ ⑤d Global unconnected state-episode pair mining (always*)      │  │
│  │          * always = every 2-chunk boundary, regardless of new trait   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Graph Structure Summary

### Node Types

```text
c (Context) ─ every turn ─────────────→ raw utterance pair
s (State)   ─ every user turn ───────→ short-term persona signal (per-turn, ≤ 1 per call)
e (Episode) ─ every chunk ───────────→ episode summary
t (Trait)   ─ every 2 chunks ────────→ long-term persona signal
```

### Evidence Families

Unified enum across all pair families: `SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT`.
`SHIFT_TO` is natural only for same-type pairs (`s ↔ s`, `t ↔ t`); for cross-type it is stored verbatim but discouraged by prompt calibration. `e ↔ s` is stored canonically as `e → s`.

### Temporal Transition Relation

```text
old_state  →SHIFT_TO→ new_state
old_trait  →SHIFT_TO→ new_trait
```

`SHIFT_TO` is directional and stored only once in the `old → new` direction.

---

## Experiment Protocol Integration

```text
For each session S_i:

  Phase 1: Memory Construction
    for each (user_turn, gt_response) in S_i:
        process_turn() → prompt construction + memory update
    finalize_chunk() → run ③ and, when triggered, ④ and ⑤ for the last chunk

  Phase 2: QA
    for each QA in S_i:
        retrieval → ⑥ answer
        (no memory mutation during QA)

  Phase 3: Cleanup
    save snapshot → aggregate stats → clear all memory
```

Rules:
- memory always stores ground-truth assistant responses;
- generated responses are never written back into memory;
- QA does not mutate memory.

---

## `process_turn()` Overview

```text
process_turn(user_utterance, gt_response, conv_id)
│
├── 1) Chunk-boundary update (if conv_id changed)
│     ├── ③ Episode extraction (extraction-only)
│     ├── if (|chunk_state_ids| ≥ 1) or (previous_episode exists):
│     │     └── ③b New-episode relation judgments (e↔chunk_states + e↔prev_episode)
│     └── if 2-chunk boundary:
│           ├── ④ Trait extraction
│           ├── if new trait exists:
│           │     ├── ⑤a Local trait evidence judgment
│           │     └── ⑤b Trait-centered extra relation extraction
│           ├── ⑤c Global state-state pair mining    (ENABLE_EXTRA_RELATION_EXTRACTION only)
│           └── ⑤d Global state-episode pair mining  (ENABLE_EXTRA_RELATION_EXTRACTION only)
│
├── 2) Turn processing
│     ├── Create ContextNode
│     ├── Retrieval (seed + APS + expansion) → Serialization
│     │   (this variant is QA-only: there is no ① Response prompt call;
│     │    serialized context is reused later by ⑥ QA)
│     └── Update context cache
│
└── 3) State extraction trigger
      └── every user turn (STATE_EXTRACTION_H = 1):
            ├── ② State extraction (extraction-only, ≤ 1 state per call)
            └── if (|new_state_ids| ≥ 2) or (|new_state_ids| ≥ 1 and |previous_state_ids| ≥ 1):
                  └── ②b New-state relation judgments (new↔new + new↔prev)
```

---

## Retrieval Pipeline Summary

```text
Query (user utterance or QA question)
│
├── Step 1: Seed Retrieval
│   ├── retrieve top-k per node type (c, e, s, t)
│   ├── unified seed scoring: w_sem(scope) · sem + w_ov(scope) · overlap_norm
│   ├── overlap uses query-side normalized: |query_kw ∩ label_set| / |query_kw|
│   ├── keywords + domain_label are single-token (Change 3 / gmem6)
│   └── APS = top-k_aps HIGH recall-priority, non-SHIFT_TO-source states ranked by seed_score
│       APS and s_seed are disjoint; HIGH states beyond rank k_aps fall through to s_seed
│
├── Step 2: Graph Expansion
│   ├── context → source-derived e, s (context nodes removed from pool)
│   ├── ±1 turn neighbors of s_seed/e_seed pulled into pool, NOT as origins (Change 2 / gmem6)
│   ├── APS members in pool but NOT in origin_ids (Change 1 / gmem6)
│   ├── Rule A: 1-hop follows SUP, CON, and SHIFT_TO; 2-hop+ follows SUP-typed edges only
│   ├── Rule B (gmem5) removed — SHIFT_TO folded into sign propagation (Change 4):
│   │     outgoing SHIFT_TO: same-type → SUP / cross-type → CON
│   │     incoming SHIFT_TO (any): → CON
│   │     ordinary evidence follows a per-node-type outgoing-direction whitelist
│   │     (_EVID_EXPAND_OUT: C:∅; E→{S,E,T}; S→{S,T}; T→{T});
│   │     ordinary incoming edges are followed only same-type
│   ├── Rule C: contradiction preservation for later scoring
│   ├── CON terminates sign propagation immediately
│   ├── multi-path tie resolution: CON wins on ties (Change 6 / gmem6)
│   ├── per-origin BFS dedup (Change 5 / spec only)
│   └── deduplication after expansion
│
└── Step 3: Final Set Assembly
    ├── uniform scoring: w_sr · support_ratio + (1 - w_sr) · seed_score
    ├── evidence flows under _EVID_EXPAND_OUT (directional; cross-type only forward) plus same-type-only incoming ordinary edges
    ├── no-evidence nodes default to support_ratio = 0.5
    ├── SHIFT_TO contribution carried by signed_cache (no separate bidirectional bonus — Change 4)
    ├── trait classification: challenged if (has SHIFT_TO out-edge to another pooled trait) OR (evidence_ratio ≤ τ); otherwise stable (no tentative)
    ├── state selection: top-k with shift-chain collapse
    ├── episode selection: top-k from pooled episodes
    ├── per-retrieve memoization of get_ordinary_signed_reachable (Change 7 / engineering)
    └── shift-chain collapse on t_final, s_final, and s_aps
```

---

## Additional Relation Extraction Summary

Extra relation extraction (⑤b/⑤c/⑤d) follows two different candidate-selection
rules depending on whether the call has a single anchor or a pair-level pool.

### Trait-centered expansion (⑤b)
Anchored on a single new trait; consumes immediately after extraction.

For each new trait, compute its semantic top-k and lexical top-k against
unconnected compatible nodes (states or episodes), union the two lists and
deduplicate by `node_id`. No `pair_score`, no rerank, no post-union cap.

### Pair-level top-k over pending pool (⑤c / ⑤d)
Both ⑤c and ⑤d defer scoring until flush time:

- between triggers, only the **IDs** of newly added states (and episodes, for
  ⑤d) accumulate in pending sets,
- at the next ⑤c / ⑤d trigger, the candidate pair pool is built by enumerating
  every unconnected pair where at least one endpoint is in the pending set,
- the pool is then reduced by a **pair-level** `sem_topK ∪ lex_topK`
  (`pair_sem = embedding dot product`, `pair_lex = keyword/domain-label
  intersection size`), deduplicated.

This keeps each ⑤c / ⑤d call bounded by `2 * K` pairs regardless of how many
new nodes accumulated between triggers, while still surfacing long-range
evidence.

---

## Serialization Sections

| Section | Source |
|---------|--------|
| Current Constraints | `s_aps` |
| Traits | stable traits from `t_final` |
| Challenged Traits | challenged traits from `t_final` with nested conflict evidence |
| Relevant States | `s_final` |
| Relevant Episodes | `e_final` |
| Recent Conversation | context cache |

Notes:
- this variant is QA-only — there is no ① Response prompt call;
- the `[Recent Conversation]` section is included in QA serialization when `INCLUDE_RECENT_CONVERSATION_FOR_QA = True`; the current default in `config_0.py` is `True` (on);
- traits with no evidence (and no SHIFT_TO out-edge to another pooled trait) are classified as stable;
- APS is constructed once at retrieval time as the top-`k_aps` HIGH recall-priority, non-`SHIFT_TO`-source states ranked by `seed_score`; APS members are excluded from the `s_seed` candidate pool. HIGH states ranked beyond `k_aps` fall through to `s_seed` and compete normally there.

---

## Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Direct-only storage** | only direct LLM judgments are stored; transitive reasoning is retrieval-time only |
| **Unified relation enum** | every relation-extraction call uses `SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT`; `SHIFT_TO` is the natural form only for same-type pairs, but cross-type emissions are stored verbatim |
| **Directional temporal transition** | `SHIFT_TO` is stored only as `old → new` |
| **Sign propagation** | CON terminates immediately; no multi-hop CON propagation; shortest path priority; **CON wins on equal-length ties (Change 6 / gmem6)** |
| **SHIFT_TO unified into sign propagation (Change 4 / gmem6)** | outgoing SHIFT_TO emits SUP for same-type targets (continue forward) and CON for cross-type targets (source invalidates the cross-type target); incoming SHIFT_TO emits CON regardless of type (record old version, stop). The dedicated forward BFS and the final-set bidirectional bonus from gmem5 are removed; the same signal is now carried by `signed_cache` |
| **Single-token labels (Change 3 / gmem6)** | every keyword and every domain_label is a single token; multi-word phrases break the token-level set intersection used by `overlap_norm` and are dropped at storage time |
| **±1 turn pool seeding (Change 2 / gmem6)** | ±1 turn S/E neighbors of `s_seed` and `e_seed` are pulled into the pool (within the same `conv_id`) as a recall safety net for sparse extraction; they are pool members only, not expansion origins |
| **APS is evidence-only (Change 1 / gmem6)** | APS members remain in the pool and contribute SUP/CON signals to other nodes, but they are **excluded from `origin_ids`** — APS is a query-independent persona slot, not a query-driven expansion driver |
| **Uniform final scoring** | all node types use `w_sr · support_ratio + (1 - w_sr) · seed_score`; evidence flow is directional (`_EVID_EXPAND_OUT`: C:∅; E→{S,E,T}; S→{S,T}; T→{T}) and ordinary incoming edges are followed only same-type |
| **Unified seed scoring** | states, episodes, and traits share the same scope-dependent weight pair; no type-specific scoring formulas |
| **Sparse additional relation extraction** | extra evidence is added only for small top-k candidate sets (⑤b anchor top-k; ⑤c/⑤d pair-level top-k over the unconnected-pair pool at flush time) |
| **APS for implicit persona reasoning** | HIGH recall-priority states are surfaced via a dedicated top-`k_aps` slot ranked by seed score; APS and `s_seed` are disjoint partitions of the state pool |
| **Shift-aware current validity** | shift sources are removed from APS and collapsed out of final sets when newer nodes are present |
| **Per-turn single-state extraction** | ② runs every user turn and emits at most one state per call; downstream cross-state evidence comes primarily from ⑤c (global state-state pair mining) |
| **Minimal evidence edges** | EVIDENCE edges carry only `relation` and direction; no `rationale` / `evidence_quote_{a,b}` fields (per-call LLM output is also minimal — integer `source_id`/`target_id` + `relation` only) |
| **Configurable QA context** | recent conversation is gated by `INCLUDE_RECENT_CONVERSATION_FOR_QA` for QA serialization (current default `True`) |
