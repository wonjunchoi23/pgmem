# GraphMem: Storage and Extraction

> **Scope**: This document defines what the graph stores and how nodes and edges are created. It covers node and edge schema, label systems, extraction triggers, extraction I/O, and evidence-creation rules. Retrieval-time scoring, graph expansion, and final-set assembly are defined separately in `gmem5_retrieval.md`.

---

## 1. Design Principles

GraphMem stores only **direct judgments** and keeps construction sparse.

Core principles:

1. **Direct-only storage**
   - only direct LLM judgments are stored;
   - no transitive edges are materialized.

2. **Pair-family-specific relation spaces**
   - `SHIFT_TO` is allowed only for same-type temporal transitions:
     - `s ↔ s`
     - `t ↔ t`
   - cross-type pairs use only:
     - `SUPPORT`
     - `CONTRADICT`
     - `IRRELEVANT`

3. **Canonical storage direction**
   - some relation families are semantically symmetric but stored in a fixed direction for implementation simplicity;
   - retrieval may inspect incoming and outgoing edges as needed.

4. **Sparse additional relation extraction**
   - beyond mandatory local judgments, extra relation extraction is limited to small top-k candidate sets;
   - extra candidates are judged only when the pair is not already connected.

---

## 2. Node Types

### 2.1 Context Node (`c`)

```python
ContextNode = {
    id:              str,
    type:            "c",
    content:         str,        # "User: {utt}\nAgent: {gt_resp}"
    keywords:        List[str],  # spaCy noun extraction
    embedding:       vector,
    created_at:      int,        # global turn index
    session_id:      str,
    conv_id:         int,
    turn_id:         int,
    retrieval_count: int,        # init = 1
}
```

Context nodes do **not** have `domain_label`.

### 2.2 Memory Node (`m`)

```python
MemoryNode = {
    id:              str,
    type:            "m",
    content:         str,        # LLM-generated episodic summary
    keywords:        List[str],  # max 5
    domain_label:    List[str],  # 3-5 labels
    scope:           str,        # BROAD | NARROW
    embedding:       vector,
    created_at:      int,
    session_id:      str,
    conv_id:         int,
    turn_id:         int,
    retrieval_count: int,        # init = 1
}
```

### 2.3 State Node (`s`)

```python
StateNode = {
    id:                       str,
    type:                     "s",
    content:                  str,        # 1-sentence persona signal
    keywords:                 List[str],  # max 5
    domain_label:             List[str],  # 3-5 labels
    scope:                    str,        # BROAD | NARROW
    current_decision_impact:  str,        # HIGH | LOW
    embedding:                vector,
    created_at:               int,
    session_id:               str,
    conv_id:                  int,
    turn_id:                  int,
    retrieval_count:          int,        # init = 1
}
```

### 2.4 Trait Node (`t`)

```python
TraitNode = {
    id:              str,
    type:            "t",
    content:         str,        # up to 3-sentence persona signal
    keywords:        List[str],  # max 5
    domain_label:    List[str],  # 3-5 labels
    scope:           str,        # BROAD | NARROW
    embedding:       vector,
    created_at:      int,
    session_id:      str,
    conv_id:         int,        # latest conv_id in the 2-chunk window
    turn_id:         int,        # last turn_id in the 2-chunk window
    retrieval_count: int,        # init = 1
}
```

Traits do **not** have `current_decision_impact`.

---

## 3. Edge Families

### 3.1 SOURCE Edges

| Source → Target | Meaning |
|-----------------|---------|
| `c → s` | state extracted from context |
| `c → m` | memory created from contexts |
| `c → t` | contexts used as source material for trait extraction |
| `s → t` | states used as trait-extraction input |
| `m → t` | memories used as trait-extraction input |

```python
SourceEdge = {
    id:         str,
    type:       "SOURCE",
    source_id:  str,
    target_id:  str,
    created_at: int,
    session_id: str,
}
```

### 3.2 EVIDENCE Edges

```python
EvidenceEdge = {
    id:          str,
    type:        "EVIDENCE",
    source_id:   str,
    target_id:   str,
    subtype:     str,    # depends on pair family
    session_id:  str,
    created_at:  int,
}
```

### 3.3 Allowed Subtypes by Pair Family

| Pair Family | Allowed Subtypes |
|-------------|------------------|
| `s ↔ s` | `SUPPORT`, `CONTRADICT`, `SHIFT_TO`, `IRRELEVANT` |
| `t ↔ t` | `SUPPORT`, `CONTRADICT`, `SHIFT_TO`, `IRRELEVANT` |
| `m ↔ m` | `SUPPORT`, `CONTRADICT`, `IRRELEVANT` |
| `m ↔ s` | `SUPPORT`, `CONTRADICT`, `IRRELEVANT` |
| `s → t` | `SUPPORT`, `CONTRADICT`, `IRRELEVANT` |
| `m → t` | `SUPPORT`, `CONTRADICT`, `IRRELEVANT` |

Notes:
- `SHIFT_TO` is **not** allowed for cross-type pairs.
- `SUPPORT`, `CONTRADICT`, and `IRRELEVANT` are **semantically symmetric**.
- `SHIFT_TO` is **strictly directional**.
- **edge absence means the pair has not been judged yet.** `IRRELEVANT`
  judgments DO produce edges (since gmem5) so that future ⑤b/⑤c/⑤d candidate
  selection can skip already-judged pairs via `has_direct_edge`. Sign
  propagation and node score counters explicitly exclude IRRELEVANT — these
  edges are neutral with respect to support/contradict accounting.
- `IRRELEVANT` means the pair was directly judged and found unrelated.

### 3.4 Evidence Subtype Definitions

| Subtype | Definition |
|---------|------------|
| `SUPPORT` | two nodes are consistent or mutually reinforcing |
| `CONTRADICT` | two nodes are in tension, but both may still hold |
| `SHIFT_TO` | an older state or trait has changed into a newer state or trait |
| `IRRELEVANT` | pair was judged and found unrelated |

### 3.5 `SHIFT_TO` Rule

`SHIFT_TO` is strictly directional:

```text
old_node --SHIFT_TO--> new_node
```

Constraints:
- allowed only for `s ↔ s` and `t ↔ t`;
- stored **only once** in the `old → new` direction;
- no reverse edge is automatically created.

This makes `SHIFT_TO` a temporal-transition edge rather than a symmetric conflict edge.

### 3.6 Canonical Storage Direction

Some pair families are semantically symmetric but stored canonically:

| Pair Family | Semantic Family | Canonical Storage Convention |
|-------------|-----------------|------------------------------|
| `s ↔ s` | symmetric except `SHIFT_TO` | store the directed judgment as returned; `SHIFT_TO` only `old → new` |
| `t ↔ t` | symmetric except `SHIFT_TO` | store the directed judgment as returned; `SHIFT_TO` only `old → new` |
| `m ↔ m` | semantically symmetric | in ③, the new memory is the source and the previous memory is the target |
| `m ↔ s` | semantically symmetric | always store canonically as `m → s` |
| `s → t` | directional by design | store as `s → t` |
| `m → t` | directional by design | store as `m → t` |

Retrieval may inspect incoming and outgoing edges as needed, but canonical storage remains fixed.

### 3.7 Per-Judgment Output Fields

Each relation-extraction call (②b, ③b, ⑤a, ⑤b, ⑤c, ⑤d) produces per judgment:

```json
{
  "source_id": <integer index into the prompt's node list>,
  "target_id": <integer index into the prompt's node list>,
  "relation": "SUPPORT|CONTRADICT|SHIFT_TO|IRRELEVANT"
}
```

(⑤d uses `state_id` / `memory_id` with separate index spaces; see §9.5.)

Node IDs are presented as **integer indices** in every relation-extraction prompt. The apply layer maps indices back to real graph IDs. See `gmem5_prompt.md` §3.2 for the node-listing format convention.

`reasoning` and `evidence_quote` fields are **not** produced. Relation labeling is performed directly without an output chain-of-thought field.

---

## 4. Labels

### 4.1 `scope` (State)

| Label | Meaning |
|-------|---------|
| `BROAD` | a persistent personal attribute, value, health condition, or lifestyle constraint that applies regardless of the current topic — relevant even if the conversation shifts to a completely different subject |
| `NARROW` | a preference or condition tied to the current task or topic that does not transfer meaningfully to an unrelated conversation |

### 4.2 `current_decision_impact` (State)

| Label | Meaning |
|-------|---------|
| `HIGH` | use only when BOTH are true: (1) the user would reasonably expect this to be remembered without re-stating it, and (2) ignoring it would cause a response that is clearly wrong, unsafe, or would noticeably frustrate the user; typical cases: hard constraint just stated (allergy, refusal, deadline), safety-relevant condition, explicit expectation in the current exchange |
| `LOW` | useful persona context, but the response would still be appropriate and acceptable without it |

### 4.3 `scope` (Trait)

| Label | Meaning |
|-------|---------|
| `BROAD` | the trait applies across all domains of the user's life — it shapes the user's approach regardless of the subject being discussed |
| `NARROW` | a domain-specific tendency or preference mainly expressed within a particular area (e.g., a consistent style in one hobby or activity) |

### 4.4 `scope` (Memory)

| Label | Meaning |
|-------|---------|
| `BROAD` | the episode reveals or confirms a cross-topic user characteristic (e.g., a health event, a major life decision, a value-revealing exchange, or a standing constraint the user reaffirmed) |
| `NARROW` | the episode is self-contained within the current topic or task — its implications do not extend beyond the current conversation thread |

### 4.5 `domain_label`

For `s`, `m`, and `t`:
- 3-5 short labels;
- 1-3 words each;
- mix broader and narrower topic labels.

Domain labels are not fed back into extraction prompts; no reuse guidance block is included.

### 4.6 `keywords` vs `domain_label`: Disjointness Contract

`keywords` and `domain_label` are intended to capture **complementary** signals:

| Field | Meaning |
|-------|---------|
| `keywords` | Surface-level tokens that appear (or near-appear) directly in the source text. Concrete entities, named items, specific actions, or particular phrases. |
| `domain_label` | Abstract topical or categorical labels at a higher level of abstraction. Broader subject areas the node belongs to. |

Hard constraint: **`keywords` and `domain_label` MUST be disjoint sets** (case-insensitive comparison). The disjointness is enforced two ways:

1. **Prompt-side (soft contract)**: every extraction prompt that produces these fields (②, ③, ④) includes a definition block with concrete examples and the disjointness rule. See `gmem5_prompt.md` §2.3.
2. **Storage-side (hard enforcement)**: post-LLM, a `deduplicate_labels(keywords, domain_label)` helper removes any label from `domain_label` that also appears in `keywords` (case-insensitive); `keywords` win. Applied to every node-creation path (state, memory, trait).

Retrieval-side `label_set(n) = keywords(n) ∪ domain_label(n)` and `overlap_norm` are unchanged. Count parameters (`MAX_KEYWORDS = 5`, `MAX_DOMAIN_LABELS = 5`, `MIN_DOMAIN_LABELS = 3`) are unchanged.

---

## 5. Extraction Schedule

| Target | Trigger | Count per call | LLM calls |
|--------|---------|----------------|-----------|
| State | every user turn (`STATE_EXTRACTION_H = 1`) | 0-1 | 1 (② extraction-only) + 0 or 1 (②b new-state relations: new↔new + new↔prev, chained when at least one judgeable pair exists) |
| Memory | chunk boundary (`conv_id` change) | 1 | 1 (③ extraction-only) + 0 or 1 (③b new-memory relations: new\_memory↔chunk\_states + new\_memory↔previous\_memory, chained when at least one judgeable pair exists) |
| Trait | every `TRAIT_EXTRACTION_CHUNKS` chunks | 0-1 | 1 (④ extraction) + 0–2 subcalls (⑤a/⑤b, new trait only) |
| Extra relations | every `TRAIT_EXTRACTION_CHUNKS` chunks | reservoir top-k | 0–2 subcalls (⑤c/⑤d, ENABLE_EXTRA_RELATION_EXTRACTION only; regardless of new trait) |

---

## 6. ② State Extraction (extraction-only)

**Trigger**: every user turn (`STATE_EXTRACTION_H = 1`).

### Input
- the **current turn** `(user_utterance, gt_response)` pair, labeled `[CURRENT TURN — extract a state from this turn only]`;
- the most recent `STATE_REF_CONTEXT_TURNS` prior `(user, assistant)` pairs labeled `[PRIOR CONTEXT — for disambiguation only. DO NOT extract states from these turns. They have already been processed.]`;
- prior context **ignores `conv_id` boundaries**: the most recent `STATE_REF_CONTEXT_TURNS` pairs are shown regardless of whether they fall under a different `conv_id` (i.e., even if a 12-hour gap precedes the current turn);
- if fewer than `STATE_REF_CONTEXT_TURNS` prior pairs exist (e.g., session start), include only what is available — no padding;
- no previous-state block is included in the prompt.

### Output
- `states`: 0-1 `StateNode`s (capped by `STATE_MAX_COUNT = 1`; `maxItems` injected into schema).

② is **extraction-only**. Its schema has no `judgments` field; all new-state
relation judgments (new↔new and new↔previous) are produced by ②b. The
judgment-retry policy (see §13) skips ② accordingly.

### `current_decision_impact` assignment rule in ②
The extractor must be conservative when assigning `current_decision_impact = HIGH`.

Rules:
- use `HIGH` only when BOTH are true:
  (1) the user would reasonably expect this to be remembered without re-stating it;
  (2) ignoring it would cause a response that is clearly wrong, unsafe, or would noticeably frustrate the user;
- typical `HIGH` cases: a hard constraint just stated (allergy, refusal, deadline), a safety-relevant condition, an explicit expectation in the current exchange;
- do **not** use `HIGH` for generic biography, background preference, or mildly useful context;
- expect at most 1 `HIGH` per extraction call;
- if uncertain between `HIGH` and `LOW`, choose `LOW`.

### Post-call
1. create new state nodes and embeddings;
2. create `c → s` SOURCE edges.

EVIDENCE edges among new states (new↔new) are not created here — they are
produced by ②b together with new↔previous edges.

---

## 6b. ②b New-State Relation Judgments

**Trigger**: chained immediately after ② when there is at least one
judgeable pair involving the new state(s):

```text
trigger ⇔ |new_state_ids| ≥ 2  ∨  (|new_state_ids| ≥ 1 ∧ |previous_state_ids| ≥ 1)
```

Under `STATE_MAX_COUNT = 1`, the new↔new branch is dead and the trigger
reduces to `new ≥ 1 ∧ prev ≥ 1`; the general form is kept so that raising
`STATE_MAX_COUNT` later requires no code change.

### Input
- the freshly created new state nodes (`new_state_ids`);
- the `STATE_NEW_REL_PREV_WINDOW` most-recent existing state nodes (by `created_at` desc), excluding the newly extracted ones (`previous_state_ids`).

### Output
- `judgments`: one entry per
  - unordered `new ↔ new` pair (each pair exactly once), and
  - `(new, previous)` pair.

Allowed subtypes:
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

`expected_judgment_count = C(|new_state_ids|, 2) + |new_state_ids| × |previous_state_ids|`.

### Post-call
- create `s ↔ s` EVIDENCE edges for every judgment, including `IRRELEVANT`.
- `SHIFT_TO` direction normalization:
  - if exactly one endpoint is in `new_state_ids` → store `prev → new` (old → new);
  - if both endpoints are new → fall back to `created_at` ordering;
  - if both are previous-batch (out of pair space; should not occur) → fall back to `created_at`.

---

## 7. ③ Memory Extraction (extraction-only)

**Trigger**: chunk boundary. Processes the previous chunk.

### Input
- all `(user_utterance, gt_response)` pairs in the previous chunk.

The chunk states are **not** included in the ③ prompt — they are produced by
a separate per-turn pipeline and are passed only to ③b for relation judgment.
No previous-memory block is included either.

### Output
- exactly 1 `MemoryNode`.

③ is **extraction-only**. Its schema has no `judgments` field; all new-memory
relation judgments (memory↔chunk_states + memory↔previous_memory) are produced
by ③b. The judgment-retry policy (see §13) skips ③ accordingly.

### Post-call
1. create memory node and embedding;
2. create `c → m` SOURCE edges from chunk contexts.

EVIDENCE edges from the new memory to chunk states or to the previous memory
are not created here — they are produced by ③b together.

---

## 7b. ③b New-Memory Relation Judgments

**Trigger**: chained immediately after ③ when there is at least one
judgeable pair involving the new memory:

```text
trigger ⇔ |chunk_state_ids| ≥ 1  ∨  previous_memory exists
```

In practice ③b fires on virtually every chunk boundary because chunks
almost always contain ≥ 1 state.

### Input
- the newly created memory node;
- the previous memory node (`_previous_memory_id`), if any;
- the chunk's state nodes (the same chunk just summarized by ③).

### Output
- `judgments`: one entry per
  - `(new_memory, previous_memory)` pair (when a previous memory exists), and
  - `(new_memory, chunk_state_i)` pair, in the listed chunk-state order.

Allowed subtypes:
```text
SUPPORT | CONTRADICT | IRRELEVANT
```

`expected_judgment_count = (1 if previous_memory exists else 0) + |chunk_state_ids|`.

Direction is **fixed by the prompt**:
- `(new_memory, previous_memory)` → stored as `new_memory → previous_memory` (canonical `m → m`).
- `(new_memory, chunk_state_i)` → stored as `new_memory → chunk_state_i` (canonical `m → s`).

### Post-call
- For every non-IRRELEVANT judgment, create the corresponding EVIDENCE edge:
  - `m → m` for `(new_memory, previous_memory)` SUPPORT/CONTRADICT;
  - `m → s` for `(new_memory, chunk_state_i)` SUPPORT/CONTRADICT.

---

## 8. ④ Trait Extraction

**Trigger**: every `TRAIT_EXTRACTION_CHUNKS` chunks. Runs after ③.

### Input
- recent 2-chunk conversation only.

The chunk states block, chunk memories block, and most-recent-existing-trait
block are **not** exposed to ④. The trait is inferred directly from raw
conversation as a likely persistent pattern. ⑤a still receives all those
nodes for relation judgment afterwards.

### Output
- `traits`: 0-1 `TraitNode`.

### Post-call
1. create trait node and embedding;
2. create `c → t`, `s → t`, and `m → t` SOURCE edges.

Note: `s → t` and `m → t` SOURCE edges are still created against the
recent-2-chunk states and memories at apply time (the apply layer knows
those node ids from `_completed_chunks`), even though they were not part
of the LLM-facing prompt. SOURCE edges record provenance, not evidence.

---

## 9. ⑤ Evidence Expansion Around the New Trait

⑤ is composed of internal subcalls.

### 9.1 ⑤a Local Trait Evidence Judgment

**Input**
- the new trait;
- recent 2-chunk states;
- recent 2-chunk memories;
- previous trait, if any.

**Judged pairs**
- `state ↔ new_trait`
- `memory ↔ new_trait`
- `previous_trait ↔ new_trait`

Allowed subtypes:
- `state ↔ trait`: `SUPPORT | CONTRADICT | IRRELEVANT`
- `memory ↔ trait`: `SUPPORT | CONTRADICT | IRRELEVANT`
- `trait ↔ trait`: `SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT`

`SHIFT_TO` may appear only as:
```text
old_trait --SHIFT_TO--> new_trait
```

### 9.2 Pair Similarity for Additional Candidate Selection

For compatible nodes `x` and `y`, define:

```text
node_overlap(x, y) = |(keywords(x) ∪ domain_label(x)) ∩ (keywords(y) ∪ domain_label(y))|

pair_score(x, y) = w_pair_sem · sem(x, y)
                 + w_pair_lex · log(1 + node_overlap(x, y))
```

This score is used for:
- post-union reranking in ⑤b (cap = `TRAIT_EXTRA_REL_TOPK_STATE` = 7),
- global unconnected candidate reservoirs in ⑤c and ⑤d.

### 9.3 ⑤b Trait-Centered Additional Relation Extraction

For the new trait, retrieve candidate nodes from the existing graph using the union of:
- semantic top-k compatible states,
- lexical top-k compatible states,
- semantic top-k compatible memories,
- lexical top-k compatible memories.

Only unconnected pairs are eligible.

Important:
- the config values `TRAIT_EXTRA_REL_TOPK_STATE` and `TRAIT_EXTRA_REL_TOPK_MEMORY` are **post-union caps**;
- semantic and lexical candidate lists are merged, deduplicated, reranked by `pair_score`, then capped.

Judged pairs:
- extra `state ↔ new_trait`
- extra `memory ↔ new_trait`

Allowed subtypes:
```text
SUPPORT | CONTRADICT | IRRELEVANT
```

### 9.4 ⑤c Additional State-State Relation Extraction

GraphMem maintains a global reservoir of high-scoring **unconnected** `state-state` pairs.

Update rule:
- whenever a new state is added, compare it against existing states not already directly connected;
- compute `pair_score(state_i, state_j)`;
- insert the pair into the global unconnected `state-state` reservoir;
- if the reservoir exceeds `STATE_STATE_EXTRA_REL_TOPK` (= 5), evict the lowest-scoring pair.

At ⑤c:
- consume the current global top-k pairs from this reservoir;
- pairs that became connected between insertion and consumption (e.g. via ②b) are **skipped** at build time via a `has_direct_edge` re-check;
- after judgment, all consumed pairs (including those skipped above) are removed from the reservoir regardless of the judgment result (including `IRRELEVANT`);
- the reservoir rebuilds incrementally as new states arrive before the next ⑤c trigger.

This makes ⑤c a **global top-k pair mining** step rather than an anchor-local step.

Allowed subtypes:
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

`SHIFT_TO` is allowed only when the pair is a true `old_state → new_state` transition.

### 9.5 ⑤d Additional State-Memory Relation Extraction

GraphMem maintains a global reservoir of high-scoring **unconnected** `state-memory` pairs.

Update rule:
- whenever a new state or memory is added, compare it against compatible existing nodes not already directly connected;
- compute `pair_score(state_i, memory_j)`;
- insert the pair into the global unconnected `state-memory` reservoir;
- if the reservoir exceeds `STATE_MEMORY_EXTRA_REL_TOPK` (= 2), evict the lowest-scoring pair.

At ⑤d:
- consume the current global top-k pairs from this reservoir;
- pairs that became connected between insertion and consumption (e.g. via ③b) are **skipped** at build time via a `has_direct_edge` re-check;
- after judgment, all consumed pairs (including those skipped above) are removed from the reservoir regardless of the judgment result (including `IRRELEVANT`);
- the reservoir rebuilds incrementally as new states or memories arrive before the next ⑤d trigger.

This is also a **global top-k pair mining** step.

Allowed subtypes:
```text
SUPPORT | CONTRADICT | IRRELEVANT
```

Stored canonically as:
```text
memory → state
```

The prompt returns `{state_id, memory_id, relation}` without directional ordering. The storage layer converts each judgment into a canonical `m → s` EVIDENCE edge.

### 9.6 Post-⑤ Edge Materialization

Create all newly judged `EVIDENCE` edges from ⑤a-⑤d.

No transitive closure is written back.

Retrieval-time note:
- expansion-reached memories compete with seed memories for the final relevant-memory set `m_f` through uniform evidence-based scoring (see `gmem4_retrieval.md` Section 4.6);
- this scoring is controlled entirely at retrieval time and does not change storage semantics.

---

## 10. Edge Creation Summary by Call

| Call | Pair Family | Canonical Storage |
|------|-------------|-------------------|
| ② | (extraction-only — no edges) | — |
| ②b | `s ↔ s` | `prev → new` for cross-batch; `created_at`-ordered for new↔new; `SHIFT_TO` only `old → new` |
| ③ | (extraction-only — only `c → m` SOURCE edges) | — |
| ③b | `m ↔ s`, `m ↔ m` | `m → s`; `new_memory → previous_memory` |
| ⑤a | `s → t`, `m → t`, `t ↔ t` | canonical cross-type direction; `SHIFT_TO` only for `t ↔ t` |
| ⑤b | `state ↔ trait`, `memory ↔ trait` | `s → t`, `m → t` |
| ⑤c | `s ↔ s` | directed as judged (older → newer); `SHIFT_TO` only `old → new` |
| ⑤d | `m ↔ s` | always stored canonically as `m → s` |

---

## 11. Cross-Temporal Reasoning Note

Cross-temporal consistency is supported by two mechanisms:

1. **Local mandatory extraction**
   - `state ↔ previous state`
   - `new memory ↔ previous memory`
   - `previous trait ↔ new trait`

2. **Sparse global candidate mining**
   - `⑤c` global top-k unconnected `state-state` pairs
   - `⑤d` global top-k unconnected `state-memory` pairs

Multi-hop reasoning is still deferred to retrieval time.

---

## 12. Minimal Execution Order Example

```text
[Chunk 0]
Turn 1: Context → ①
Turn 2: Context → ① → ② State
Turn 3: Context → ①
Turn 4: Context → ① → ② State
...

[Chunk 0 → 1 boundary]
③ Memory(chunk 0)

[Chunk 1]
...

[Chunk 1 → 2 boundary]
③ Memory(chunk 1)
④ Trait(chunks 0-1)
  → if new trait:
      ⑤a Local trait evidence
      ⑤b Trait-centered extra relations   (ENABLE_EXTRA_RELATION_EXTRACTION only)
⑤c Extra state-state relations            (ENABLE_EXTRA_RELATION_EXTRACTION only; runs regardless of new trait)
⑤d Extra state-memory relations           (ENABLE_EXTRA_RELATION_EXTRACTION only; runs regardless of new trait)

[Chunk 3 → 4 boundary: example with no new trait]
③ Memory(chunk 3)
④ Trait(chunks 2-3)
  → no new trait: ⑤a/⑤b skipped
⑤c Extra state-state relations            (ENABLE_EXTRA_RELATION_EXTRACTION only)
⑤d Extra state-memory relations           (ENABLE_EXTRA_RELATION_EXTRACTION only)
```
