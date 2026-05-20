# GraphMem: Storage and Extraction

> **Scope**: This document defines what the graph stores and how nodes and edges are created. It covers node and edge schema, label systems, extraction triggers, extraction I/O, and evidence-creation rules. Retrieval-time scoring, graph expansion, and final-set assembly are defined separately in `gmem6_retrieval.md`.

---

## 1. Design Principles

GraphMem stores only **direct judgments** and keeps construction sparse.

Core principles:

1. **Direct-only storage**
   - only direct LLM judgments are stored;
   - no transitive edges are materialized.

2. **Unified relation enum**
   - every relation-extraction call uses the same four-relation enum:
     `SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT`;
   - `SHIFT_TO` is the **natural** form only for same-type temporal transitions (`s ↔ s`, `t ↔ t`);
   - for cross-type pairs (`s ↔ t`, `e ↔ t`, `e ↔ s`, `e ↔ e`) the prompt discourages `SHIFT_TO`, but the apply layer stores it verbatim if the LLM emits it (no filtering).

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

### 2.2 Episode Node (`e`)

```python
EpisodeNode = {
    id:              str,
    type:            "e",
    content:         str,        # LLM-generated episode summary
    keywords:        List[str],  # max MAX_KEYWORDS (= 7)
    domain_label:    List[str],  # MIN_DOMAIN_LABELS..MAX_DOMAIN_LABELS (= 5..7); filled with "general"/"general_N" if under min
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
    keywords:                 List[str],  # max MAX_KEYWORDS (= 7)
    domain_label:             List[str],  # MIN_DOMAIN_LABELS..MAX_DOMAIN_LABELS (= 5..7); filled with "general"/"general_N" if under min
    scope:                    str,        # BROAD | NARROW
    recall_priority:          str,        # HIGH | LOW
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
    keywords:        List[str],  # max MAX_KEYWORDS (= 7)
    domain_label:    List[str],  # MIN_DOMAIN_LABELS..MAX_DOMAIN_LABELS (= 5..7); filled with "general"/"general_N" if under min
    scope:           str,        # BROAD | NARROW
    embedding:       vector,
    created_at:      int,
    session_id:      str,
    conv_id:         int,        # latest conv_id in the 2-chunk window
    turn_id:         int,        # last turn_id in the 2-chunk window
    retrieval_count: int,        # init = 1
}
```

Traits do **not** have `recall_priority`.

---

## 3. Edge Families

Edges are stored in `HeterogeneousGraph` as two pairs of adjacency maps —
`_src_out` / `_src_in` for SOURCE edges (set of neighbor node-ids per node)
and `_evid_out` / `_evid_in` for EVIDENCE edges (dict from neighbor node-id
to subtype). There is **no per-edge object** carrying `id`, `created_at`, or
`session_id`; those fields below describe the **logical** edge contract used
in this document (and the per-judgment LLM I/O contract), not the in-memory
representation. The persisted snapshot writes only `_src_out` and
`_evid_out` (see `HeterogeneousGraph.save_snapshot`).

### 3.1 SOURCE Edges

| Source → Target | Meaning |
|-----------------|---------|
| `c → s` | state extracted from context |
| `c → e` | episode created from contexts |
| `c → t` | contexts used as source material for trait extraction |
| `s → t` | states used as trait-extraction input |
| `e → t` | episodes used as trait-extraction input |

Logical schema (representation in memory is a `Set[neighbor_id]` per node):

```python
SourceEdge = {
    type:       "SOURCE",
    source_id:  str,
    target_id:  str,
}
```

### 3.2 EVIDENCE Edges

Logical schema (representation in memory is a `Dict[neighbor_id, subtype]`
per node; the subtype is the only per-edge value actually stored):

```python
EvidenceEdge = {
    type:        "EVIDENCE",
    source_id:   str,
    target_id:   str,
    subtype:     str,    # SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
}
```

### 3.3 Allowed Subtypes by Pair Family

Every pair family accepts the same unified enum
`SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT`. `SHIFT_TO` is the natural
temporal-transition form for same-type pairs (`s ↔ s`, `t ↔ t`); the prompt
discourages it for cross-type pairs but the apply layer stores it verbatim.

Notes:
- `SUPPORT`, `CONTRADICT`, and `IRRELEVANT` are **semantically symmetric**.
- `SHIFT_TO` is **strictly directional** (`old → new`).
- **Edge absence means the pair has not been judged yet.** `IRRELEVANT`
  judgments DO produce edges so future ⑤b/⑤c/⑤d candidate selection can skip
  already-judged pairs via `has_direct_edge` (gated by `EXTRA_REL_ONLY_IF_UNCONNECTED`).
  `signed_cache` carries `IRRELEVANT` as a third sign distinct from `SUP`/`CON`:
  the IRRELEVANT edge stops further propagation just like CON, but it does **not**
  count toward `support_ratio` (its presence keeps the pair in the
  "no evidence → default 0.5" branch when no other SUP/CON path exists).

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
- stored **only once** in the `old → new` direction; no reverse edge.
- natural usage is restricted to same-type pairs (`s ↔ s`, `t ↔ t`); cross-type
  emissions, if any, are stored verbatim but discouraged by prompt calibration.

This makes `SHIFT_TO` a temporal-transition edge rather than a symmetric conflict edge.

### 3.6 Canonical Storage Direction

Some pair families are semantically symmetric but stored canonically:

| Pair Family | Semantic Family | Canonical Storage Convention |
|-------------|-----------------|------------------------------|
| `s ↔ s` | symmetric except `SHIFT_TO` | store the directed judgment as returned; `SHIFT_TO` only `old → new` |
| `t ↔ t` | symmetric except `SHIFT_TO` | store the directed judgment as returned; `SHIFT_TO` only `old → new` |
| `e ↔ e` | semantically symmetric | in ③, the new episode is the source and the previous episode is the target |
| `e ↔ s` | semantically symmetric | always store canonically as `e → s` |
| `s → t` | directional by design | store as `s → t` |
| `e → t` | directional by design | store as `e → t` |

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

(⑤d uses `state_id` / `episode_id` with separate index spaces; see §9.5.)

Node IDs are presented as **integer indices** in every relation-extraction prompt. The apply layer maps indices back to real graph IDs. See `gmem6_prompt.md` §3.2 for the node-listing format convention.

`reasoning` and `evidence_quote` fields are **not** produced. Relation labeling is performed directly without an output chain-of-thought field.

---

## 4. Labels

### 4.1 `scope` (State)

| Label | Meaning |
|-------|---------|
| `BROAD` | a persistent personal attribute, value, health condition, or lifestyle constraint that applies regardless of the current topic — relevant even if the conversation shifts to a completely different subject |
| `NARROW` | a preference or condition tied to the current task or topic that does not transfer meaningfully to an unrelated conversation |

### 4.2 `recall_priority` (State)

| Label | Meaning |
|-------|---------|
| `HIGH` | use only when BOTH are true: (1) the user would reasonably expect this to be remembered without re-stating it, and (2) ignoring it would cause a response that is clearly wrong, unsafe, or would noticeably frustrate the user; typical cases: hard constraint just stated (allergy, refusal, deadline), safety-relevant condition, explicit expectation in the current exchange |
| `LOW` | useful persona context, but the response would still be appropriate and acceptable without it |

### 4.3 `scope` (Trait)

| Label | Meaning |
|-------|---------|
| `BROAD` | the trait applies across all domains of the user's life — it shapes the user's approach regardless of the subject being discussed |
| `NARROW` | a domain-specific tendency or preference mainly expressed within a particular area (e.g., a consistent style in one hobby or activity) |

### 4.4 `scope` (Episode)

| Label | Meaning |
|-------|---------|
| `BROAD` | the episode reveals or confirms a cross-topic user characteristic (e.g., a health event, a major life decision, a value-revealing exchange, or a standing constraint the user reaffirmed) |
| `NARROW` | the episode is self-contained within the current topic or task — its implications do not extend beyond the current conversation thread |

### 4.5 `domain_label`

For `s`, `e`, and `t`:
- 3-5 short labels;
- **single token each** (no whitespace, no multi-word phrases — see §4.6, Change 3 / gmem6);
- mix broader and narrower topic labels.

Domain labels are not fed back into extraction prompts; no reuse guidance block is included.

### 4.6 `keywords` vs `domain_label`: Disjointness + Single-Token Contract

`keywords` and `domain_label` are intended to capture **complementary** signals:

| Field | Meaning |
|-------|---------|
| `keywords` | Surface-level tokens that appear (or near-appear) directly in the source text. Concrete entities, named items, specific actions, or particular phrases. |
| `domain_label` | Abstract topical or categorical labels at a higher level of abstraction. Broader subject areas the node belongs to. |

Two hard constraints:

1. **Disjointness (gmem5 Change 2):** `keywords` and `domain_label` MUST be disjoint sets (case-insensitive comparison). Enforced (a) prompt-side via the disjointness rule in every node-emitting prompt (see `gmem6_prompt.md` §2.3), and (b) storage-side via `deduplicate_labels(keywords, domain_label)`, which strips any `domain_label` token that case-insensitively matches a keyword (`keywords` win). Applied to every node-creation path (state, episode, trait).

2. **Single token (Change 3 / gmem6):** every entry in `keywords` and every entry in `domain_label` MUST be a single token (no whitespace, no multi-word phrases). Rationale: overlap scoring `|q_kw ∩ label_set(n)|` is a token-level set intersection. Multi-word entries break this — a query token `"machine"` cannot match a node phrase `"machine learning"` (and vice versa). Enforced (a) prompt-side via the per-field rule in `gmem6_prompt.md` §2.3, and (b) storage-side via `_validate_keywords` / `_validate_domain_labels`, which silently drop any item containing whitespace.

Retrieval-side `label_set(n) = keywords(n) ∪ domain_label(n)` and `overlap_norm` are unchanged. Count parameters under gmem6 are `MAX_KEYWORDS = 7`, `MAX_DOMAIN_LABELS = 7`, `MIN_DOMAIN_LABELS = 5`. When fewer than `MIN_DOMAIN_LABELS` domain labels survive validation (whitespace drops, duplicate drops, and keyword-overlap drops from `deduplicate_labels`), `_validate_domain_labels` appends generic `"general"`, `"general_2"`, `"general_3"`, ... fillers up to `MIN_DOMAIN_LABELS`. Keyword backfill is intentionally **not** used because `HeterogeneousGraph.add_node` re-runs `deduplicate_labels` at storage time and would strip duplicated tokens again.

Caveat: compound terms (e.g., `"data science"`, `"machine learning"`) lose their phrase identity. False matches on common constituent tokens are possible (`"machine"` matching a `"machine learning"` node, or `"food"` matching an unrelated food-themed node). For ImplexConv — where keywords are mostly user-situation nouns rather than technical jargon — this risk is acceptable, but it is worth monitoring.

---

## 5. Extraction Schedule

| Target | Trigger | Count per call | LLM calls |
|--------|---------|----------------|-----------|
| State | every user turn (`STATE_EXTRACTION_H = 1`) | 0-1 | 1 (② extraction-only) + 0 or 1 (②b new-state relations: new↔new + new↔prev, chained when at least one judgeable pair exists) |
| Episode | chunk boundary (`conv_id` change) | 1 | 1 (③ extraction-only) + 0 or 1 (③b new-episode relations: new\_episode↔chunk\_states + new\_episode↔previous\_episode, chained when at least one judgeable pair exists) |
| Trait | every `TRAIT_EXTRACTION_CHUNKS` chunks | 0-1 | 1 (④ extraction) + 0–2 subcalls (⑤a/⑤b, new trait only) |
| Extra relations | every `TRAIT_EXTRACTION_CHUNKS` chunks | pair-level top-k over pending pool | 0–2 subcalls (⑤c/⑤d, ENABLE_EXTRA_RELATION_EXTRACTION only; regardless of new trait) |

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

### `recall_priority` assignment rule in ②
The extractor must be conservative when assigning `recall_priority = HIGH`.

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

## 7. ③ Episode Extraction (extraction-only)

**Trigger**: chunk boundary. Processes the previous chunk.

### Input
- all `(user_utterance, gt_response)` pairs in the previous chunk.

The chunk states are **not** included in the ③ prompt — they are produced by
a separate per-turn pipeline and are passed only to ③b for relation judgment.
No previous-episode block is included either.

### Output
- exactly 1 `EpisodeNode`.

③ is **extraction-only**. Its schema has no `judgments` field; all new-episode
relation judgments (episode↔chunk_states + episode↔previous_episode) are produced
by ③b. The judgment-retry policy (see §13) skips ③ accordingly.

### Post-call
1. create episode node and embedding;
2. create `c → e` SOURCE edges from chunk contexts.

EVIDENCE edges from the new episode to chunk states or to the previous episode
are not created here — they are produced by ③b together.

---

## 7b. ③b New-Episode Relation Judgments

**Trigger**: chained immediately after ③ when there is at least one
judgeable pair involving the new episode:

```text
trigger ⇔ |chunk_state_ids| ≥ 1  ∨  previous_episode exists
```

In practice ③b fires on virtually every chunk boundary because chunks
almost always contain ≥ 1 state.

### Input
- the newly created episode node;
- the previous episode node (`_previous_episode_id`), if any;
- the chunk's state nodes (the same chunk just summarized by ③).

### Output
- `judgments`: one entry per
  - `(new_episode, previous_episode)` pair (when a previous episode exists), and
  - `(new_episode, chunk_state_i)` pair, in the listed chunk-state order.

Allowed subtypes (unified):
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

`SHIFT_TO` is rarely appropriate here (the pairs are cross-type or
new→previous episode) and is stored verbatim if emitted; calibration text in
the user prompt discourages its use.

`expected_judgment_count = (1 if previous_episode exists else 0) + |chunk_state_ids|`.

Direction is **fixed by the prompt**:
- `(new_episode, previous_episode)` → stored as `new_episode → previous_episode` (canonical `e → e`).
- `(new_episode, chunk_state_i)` → stored as `new_episode → chunk_state_i` (canonical `e → s`).

### Post-call
- For every judgment (including `IRRELEVANT`), create the corresponding EVIDENCE edge in canonical direction:
  - `e → e` for `(new_episode, previous_episode)` SUPPORT / CONTRADICT / IRRELEVANT;
  - `e → s` for `(new_episode, chunk_state_i)` SUPPORT / CONTRADICT / IRRELEVANT;
  - `SHIFT_TO` is stored verbatim with the apply-layer direction rule (cross-type is discouraged by prompt calibration but not filtered).

---

## 8. ④ Trait Extraction

**Trigger**: every `TRAIT_EXTRACTION_CHUNKS` chunks. Runs after ③.

### Input
- recent 2-chunk conversation only.

The chunk states block, chunk episodes block, and most-recent-existing-trait
block are **not** exposed to ④. The trait is inferred directly from raw
conversation as a likely persistent pattern. ⑤a still receives all those
nodes for relation judgment afterwards.

### Output
- `traits`: 0-1 `TraitNode`.

### Post-call
1. create trait node and embedding;
2. create `c → t`, `s → t`, and `e → t` SOURCE edges.

Note: `s → t` and `e → t` SOURCE edges are still created against the
recent-2-chunk states and episodes at apply time (the apply layer knows
those node ids from `_completed_chunks`), even though they were not part
of the LLM-facing prompt. SOURCE edges record provenance, not evidence.

---

## 9. ⑤ Evidence Expansion Around the New Trait

⑤ is composed of internal subcalls.

### 9.1 ⑤a Local Trait Evidence Judgment

**Input**
- the new trait;
- recent 2-chunk states;
- recent 2-chunk episodes;
- previous trait, if any.

**Judged pairs**
- `state ↔ new_trait`
- `episode ↔ new_trait`
- `previous_trait ↔ new_trait`

Allowed subtypes (unified across all pair families):
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

For same-type `trait ↔ trait`, `SHIFT_TO` is the natural temporal
replacement form and is canonicalized as:
```text
old_trait --SHIFT_TO--> new_trait
```
For cross-type pairs (`state ↔ trait`, `episode ↔ trait`) `SHIFT_TO` is
stored verbatim if emitted by the LLM; calibration text in the user prompt
discourages its use for cross-type pairs.

### 9.2 Candidate Selection for Additional Relation Extraction

For compatible nodes `x` and `y`, define two independent rankings:

```text
sem_score(x, y)  = x.embedding · y.embedding             # cosine similarity
node_overlap(x, y) = |(keywords(x) ∪ domain_label(x)) ∩ (keywords(y) ∪ domain_label(y))|
lex_score(x, y)  = node_overlap(x, y)
```

Two selection rules apply depending on whether the call has a single anchor
or a pool of candidate pairs:

**Single-anchor top-k (⑤b).** For each anchor node, the candidate set is

```text
candidates = sem_topk(anchor) ∪ lex_topk(anchor)   # dedup by node_id
```

— i.e. the union of the semantic top-k and lexical top-k against unconnected
compatible nodes, deduplicated. There is no rerank and no post-union cap.

**Pair-level top-k (⑤c / ⑤d).** Both calls defer scoring until flush time:

```text
pair_sem(x, y) = sem_score(x, y)
pair_lex(x, y) = lex_score(x, y)
selected_pairs = sem_topk(pool) ∪ lex_topk(pool)   # dedup by node-id pair
```

where `pool` is the set of unconnected `(x, y)` pairs in which at least one
endpoint was newly added between the previous and current ⑤c / ⑤d trigger.
No rerank, no post-union cap. The union has at most `2k` pairs.

The per-list `k` comes from the existing config keys
(`TRAIT_EXTRA_REL_TOPK_STATE`, `TRAIT_EXTRA_REL_TOPK_EPISODE`,
`STATE_STATE_EXTRA_REL_TOPK`, `STATE_EPISODE_EXTRA_REL_TOPK`) — the same
value is used for the semantic and the lexical fetch. The exact union size
is non-deterministic in the overlap between the two rankings.

### 9.3 ⑤b Trait-Centered Additional Relation Extraction

For the new trait, retrieve candidate states and episodes per §9.2:

- `state_candidates   = sem_topk(trait, NODE_S) ∪ lex_topk(trait, NODE_S)` (dedup)
- `episode_candidates = sem_topk(trait, NODE_E) ∪ lex_topk(trait, NODE_E)` (dedup)

with `k = TRAIT_EXTRA_REL_TOPK_STATE` (= 7) for states and
`k = TRAIT_EXTRA_REL_TOPK_EPISODE` (= 3) for episodes. Only unconnected pairs
are eligible. No rerank, no cap.

Judged pairs:
- extra `state ↔ new_trait`
- extra `episode ↔ new_trait`

Allowed subtypes:
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

The full enum is shared across all relation-extraction calls. `SHIFT_TO`
applied to cross-type pairs is stored verbatim; calibration text in the
user prompt continues to warn against overusing it.

### 9.4 ⑤c Additional State-State Relation Extraction

GraphMem tracks only the **IDs of newly added states** between ⑤c triggers;
candidate pairs are not materialized at ingestion time.

Ingestion (between triggers):
- whenever a new state is added, append its ID to the ⑤c pending set;
- no scoring, no candidate-pair construction, no per-anchor fetch.

At ⑤c:
- build the candidate pair pool by enumerating, for each pending new state,
  every other state with no direct edge to it (dedup pairs by node-id pair);
- apply pair-level `sem_topK ∪ lex_topK` from §9.2 with
  `k = STATE_STATE_EXTRA_REL_TOPK` (= 5), yielding at most `2k = 10` pairs;
- clear the pending set;
- the pending set refills incrementally as new states arrive before the next ⑤c trigger.

Allowed subtypes:
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

`SHIFT_TO` is allowed only when the pair is a true `old_state → new_state` transition.

### 9.5 ⑤d Additional State-Episode Relation Extraction

GraphMem tracks only the **IDs of newly added states and episodes** between
⑤d triggers; candidate pairs are not materialized at ingestion time.

Ingestion (between triggers):
- whenever a new state is added, append its ID to the ⑤d pending-state set;
- whenever a new episode is added, append its ID to the ⑤d pending-episode set;
- no scoring, no candidate-pair construction.

At ⑤d:
- build the candidate pair pool by enumerating, for each pending new state, every episode with no direct edge to it, and for each pending new episode, every state with no direct edge to it (dedup pairs by `(state_id, episode_id)`);
- apply pair-level `sem_topK ∪ lex_topK` from §9.2 with
  `k = STATE_EPISODE_EXTRA_REL_TOPK` (= 3), yielding at most `2k = 6` pairs;
- clear both pending sets;
- the pending sets refill incrementally as new states or episodes arrive before the next ⑤d trigger.

Allowed subtypes:
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

The full enum is shared across all relation-extraction calls. `SHIFT_TO`
applied to cross-type state↔episode pairs is stored verbatim; calibration text
in the user prompt continues to warn against overusing it.

Stored canonically as:
```text
episode → state
```

The prompt returns `{state_id, episode_id, relation}` without directional ordering. The storage layer converts each judgment into a canonical `e → s` EVIDENCE edge.

### 9.6 Post-⑤ Edge Materialization

Create all newly judged `EVIDENCE` edges from ⑤a-⑤d.

No transitive closure is written back.

Retrieval-time note:
- expansion-reached episodes compete with seed episodes for the final relevant-episode set `e_f` through uniform evidence-based scoring (see `gmem6_retrieval.md` §4.5);
- this scoring is controlled entirely at retrieval time and does not change storage semantics.

---

## 10. Edge Creation Summary by Call

| Call | Pair Family | Canonical Storage |
|------|-------------|-------------------|
| ② | (extraction-only — no edges) | — |
| ②b | `s ↔ s` | `prev → new` for cross-batch; `created_at`-ordered for new↔new; `SHIFT_TO` only `old → new` |
| ③ | (extraction-only — only `c → e` SOURCE edges) | — |
| ③b | `e ↔ s`, `e ↔ e` | `e → s`; `new_episode → previous_episode` |
| ⑤a | `s → t`, `e → t`, `t ↔ t` | canonical cross-type direction; for `t ↔ t` SHIFT_TO, `old → new` by `created_at` |
| ⑤b | `state ↔ trait`, `episode ↔ trait` | `s → t`, `e → t` |
| ⑤c | `s ↔ s` | directed as judged (older → newer); `SHIFT_TO` always `old → new` |
| ⑤d | `e ↔ s` | always stored canonically as `e → s` |

---

## 11. Cross-Temporal Reasoning Note

Cross-temporal consistency is supported by two mechanisms:

1. **Local mandatory extraction** (every pair listed below is judged by the corresponding call — `IRRELEVANT` is a valid outcome, but each pair is always judged)
   - `new state ↔ previous state` and `new state ↔ new state` (②b)
   - `new episode ↔ previous episode` and `new episode ↔ each chunk_state_i` (③b)
   - `new trait ↔ previous trait`, `new trait ↔ each recent-2-chunk state`, `new trait ↔ each recent-2-chunk episode` (⑤a)

2. **Sparse global candidate mining**
   - `⑤c` global top-k unconnected `state-state` pairs
   - `⑤d` global top-k unconnected `state-episode` pairs

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
③ Episode(chunk 0)

[Chunk 1]
...

[Chunk 1 → 2 boundary]
③ Episode(chunk 1)
④ Trait(chunks 0-1)
  → if new trait:
      ⑤a Local trait evidence
      ⑤b Trait-centered extra relations   (ENABLE_EXTRA_RELATION_EXTRACTION only)
⑤c Extra state-state relations            (ENABLE_EXTRA_RELATION_EXTRACTION only; runs regardless of new trait)
⑤d Extra state-episode relations          (ENABLE_EXTRA_RELATION_EXTRACTION only; runs regardless of new trait)

[Chunk 3 → 4 boundary: example with no new trait]
③ Episode(chunk 3)
④ Trait(chunks 2-3)
  → no new trait: ⑤a/⑤b skipped
⑤c Extra state-state relations            (ENABLE_EXTRA_RELATION_EXTRACTION only)
⑤d Extra state-episode relations          (ENABLE_EXTRA_RELATION_EXTRACTION only)
```
