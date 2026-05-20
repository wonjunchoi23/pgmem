# GraphMem: Architecture Overview

> **Scope**: High-level overview only. This document describes module boundaries, main data flow, and component responsibilities. Detailed schema is in `gmem5_storage_extraction.md`, retrieval rules are in `gmem5_retrieval.md`, hyperparameters are in `gmem5_config.md`, prompt templates are in `gmem5_prompt.md`, and runtime conventions are in `gmem5_implementation.md`.

---

## Document Map

| Document | Content |
|----------|---------|
| **gmem5_bigflow.md** | overall structure, execution flow, component relationships |
| **gmem5_storage_extraction.md** | node/edge schema, label system, extraction triggers, evidence creation |
| **gmem5_retrieval.md** | seed retrieval, graph expansion, final-set assembly, serialization contract |
| **gmem5_config.md** | tunable hyperparameters |
| **gmem5_prompt.md** | prompt templates and node-listing formats |
| **gmem5_implementation.md** | prompt orchestration, runtime conventions, module interface |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                               GraphMemModule                                │
│                                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────────────┐  │
│  │ Context    │   │ Graph      │   │ Retriever  │   │ Generator          │  │
│  │ Cache      │   │ Store      │   │            │   │                    │  │
│  │            │   │            │   │ Seed+APS   │   │ ① Response Prompt  │  │
│  │ k₀ recent  │   │ Nodes:     │   │ Expansion  │   │   Construction     │  │
│  │ pairs      │   │  c, m, s,t │   │ Final Set  │   │ ⑥ QA Answering     │  │
│  │            │   │            │   │ Serialize  │   │                    │  │
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
│  │  ③ Memory extraction (extraction-only)                               │  │
│  │  ③b New-memory relation judgments (m↔chunk_states + m↔prev_memory)   │  │
│  │  ④ Trait extraction                                                   │  │
│  │  ⑤ Additional evidence expansion                                       │  │
│  │     ├─ ⑤a Local trait evidence judgment        (new trait only)       │  │
│  │     ├─ ⑤b Trait-centered extra relation extraction (new trait only)   │  │
│  │     ├─ ⑤c Global unconnected state-state pair mining  (always*)       │  │
│  │     └─ ⑤d Global unconnected state-memory pair mining (always*)       │  │
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
m (Memory)  ─ every chunk ───────────→ episodic summary
t (Trait)   ─ every 2 chunks ────────→ long-term persona signal
```

### Evidence Families

```text
s↔s  → SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
t↔t  → SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
m↔m  → SUPPORT | CONTRADICT | IRRELEVANT
m↔s  → SUPPORT | CONTRADICT | IRRELEVANT   (stored canonically as m→s)
s→t  → SUPPORT | CONTRADICT | IRRELEVANT
m→t  → SUPPORT | CONTRADICT | IRRELEVANT
```

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
│     ├── ③ Memory extraction (extraction-only)
│     ├── if (|chunk_state_ids| ≥ 1) or (previous_memory exists):
│     │     └── ③b New-memory relation judgments (m↔chunk_states + m↔prev_memory)
│     └── if 2-chunk boundary:
│           ├── ④ Trait extraction
│           ├── if new trait exists:
│           │     ├── ⑤a Local trait evidence judgment
│           │     └── ⑤b Trait-centered extra relation extraction
│           ├── ⑤c Global state-state pair mining    (ENABLE_EXTRA_RELATION_EXTRACTION only)
│           └── ⑤d Global state-memory pair mining   (ENABLE_EXTRA_RELATION_EXTRACTION only)
│
├── 2) Turn processing
│     ├── Create ContextNode
│     ├── Retrieval (seed + APS + expansion) → Serialization
│     ├── ① Response prompt construction
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
│   ├── retrieve top-k per node type (c, m, s, t)
│   ├── unified seed scoring: w_sem(scope) · sem + w_ov(scope) · overlap_norm
│   ├── overlap uses query-side normalized: |query_kw ∩ label_set| / |query_kw|
│   └── APS = top-k_aps HIGH-impact, non-SHIFT_TO-source states ranked by seed_score
│       APS and s_seed are disjoint; HIGH states beyond rank k_aps fall through to s_seed
│
├── Step 2: Graph Expansion
│   ├── context → source-derived m, s (context nodes removed from pool)
│   ├── Rule A: 1-hop follows SUP and CON; 2-hop+ follows SUP only
│   ├── Rule B: SHIFT_TO forward traversal (old → new only); chain + SUP resumption
│   ├── Rule C: contradiction preservation for later scoring
│   ├── CON terminates sign propagation immediately
│   ├── SHIFT_TO not in sign table during expansion (discovery only)
│   └── deduplication after expansion
│
└── Step 3: Final Set Assembly
    ├── uniform scoring: w_sr · support_ratio + (1 - w_sr) · seed_score
    ├── all node types serve as evidence for all other node types
    ├── no-evidence nodes default to support_ratio = 0.5
    ├── SHIFT_TO contributes bidirectionally in support ratio (A: con_w+=1, B: sup_w+=1)
    ├── trait selection: top-k → stable/challenged classification (no tentative)
    ├── state selection: top-k with shift-chain collapse
    ├── memory selection: top-k from pooled memories
    └── shift-chain collapse on t_final, s_final, and s_aps
```

---

## Additional Relation Extraction Summary

GraphMem uses two different sparse expansion modes:

### Trait-centered expansion
- semantic top-k and lexical top-k provisional candidates,
- union + dedup + rerank,
- capped **after** union.

### Global unconnected pair mining
- maintain a global top-k reservoir for unconnected `state-state` pairs,
- maintain a global top-k reservoir for unconnected `state-memory` pairs,
- update incrementally as new nodes arrive,
- evict the current weakest pair when a stronger unconnected pair appears.

This keeps relation growth sparse while still surfacing long-range evidence.

---

## Serialization Sections

| Section | Source |
|---------|--------|
| Current Constraints | `s_aps` |
| Traits | stable traits from `t_final` |
| Challenged Traits | challenged traits from `t_final` with nested conflict evidence |
| Relevant States | `s_final` |
| Relevant Memories | `m_final` |
| Recent Conversation | context cache |

Notes:
- recent conversation is always available for response prompting;
- for QA it is optional and off by default;
- traits with no evidence are classified as stable;
- APS is constructed once at retrieval time as the top-`k_aps` HIGH-impact, non-`SHIFT_TO`-source states ranked by `seed_score`; APS members are excluded from the `s_seed` candidate pool. HIGH states ranked beyond `k_aps` fall through to `s_seed` and compete normally there.

---

## Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Direct-only storage** | only direct LLM judgments are stored; transitive reasoning is retrieval-time only |
| **Pair-family-specific relation space** | `SHIFT_TO` is same-type only; cross-type pairs use ordinary support/contradiction/irrelevance |
| **Directional temporal transition** | `SHIFT_TO` is stored only as `old → new` |
| **Simplified sign propagation** | CON terminates immediately; no multi-hop CON propagation; shortest path priority |
| **Bidirectional SHIFT_TO in scoring** | `SHIFT_TO(A→B)` contributes both signs during support-ratio computation: A receives `con_w += 1` (penalized as outdated), B receives `sup_w += 1` (boosted as current) |
| **Uniform final scoring** | all node types use `w_sr · support_ratio + (1 - w_sr) · seed_score`; all node types serve as evidence for all others |
| **Unified seed scoring** | states, memories, and traits share the same scope-dependent weight pair; no type-specific scoring formulas |
| **Sparse additional relation extraction** | extra evidence is added only for small top-k candidate sets or global unconnected pair reservoirs |
| **APS for implicit persona reasoning** | high-impact states are surfaced via a dedicated top-`k_aps` slot ranked by seed score; APS and `s_seed` are disjoint partitions of the state pool |
| **Shift-aware current validity** | shift sources are removed from APS and collapsed out of final sets when newer nodes are present |
| **Per-turn single-state extraction** | ② runs every user turn and emits at most one state per call; downstream cross-state evidence comes primarily from ⑤c (global state-state pair mining) |
| **Minimal evidence edges** | EVIDENCE edges carry only `relation` and direction; no `rationale` / `evidence_quote_{a,b}` fields (per-call LLM output is also minimal — integer `source_id`/`target_id` + `relation` only) |
| **Configurable QA context** | recent conversation is optional for QA serialization |
