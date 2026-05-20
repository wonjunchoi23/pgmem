# GraphMem: Differences from Base Specification

> **Scope**: This document tracks approved modifications to the base GraphMem specification (`gmem4_bigflow.md`, `gmem4_storage_extraction.md`, `gmem4_retrieval.md`, `gmem4_config.md`, `gmem4_implementation.md`). When implementing or refactoring code, read this document **alongside** the base spec; this document overrides the base spec wherever they conflict.

> **Convention**: Each change entry has the same structure — *Original Behavior*, *New Behavior*, *Affected Documents*, *Implementation Notes*. New entries are appended at the bottom, never reordered, so old change numbers remain stable.

---

## Change Log

### Change 1: Empty Judgment Retry with IRRELEVANT Fallback

**Motivation**
The existing `JSON_RETRY = 3` only handles JSON parsing failures. It does not catch the case where the LLM returns valid JSON but silently omits all judgments. This silent omission is distinguishable from a legitimate empty case (e.g., only one new state was extracted, so no `new ↔ new` pair exists).

**Original Behavior**
- Judgments array can be empty for two indistinguishable reasons:
  1. legitimately no pairs to judge (e.g., 0 or 1 new node), OR
  2. LLM was supposed to produce judgments but skipped them.
- Both cases are silently accepted; no retry occurs.

**New Behavior**
- Distinguish *expected* empty from *unexpected* empty using `expected_judgment_count`:
  - if `expected_judgment_count == 0` → empty is correct, no retry;
  - if `expected_judgment_count > 0` AND `returned_judgment_count == 0` → retry up to `JUDGMENT_RETRY` times.
- `IRRELEVANT` is treated as an **explicit valid judgment**, not an empty case. A judgments array fully populated with `IRRELEVANT` entries is not retried.
- After `JUDGMENT_RETRY` exhausted with no judgments, **fallback**: treat all expected pairs as `IRRELEVANT`. No edges are created. Nodes themselves are preserved.

**Expected-count formula per call**

| Call | `expected_judgment_count` |
|------|---------------------------|
| ② state extraction | `C(|new_states|, 2)` (pairs among new states) |
| ②b new ↔ previous state | `|new_states| × |previous_state_ids|` |
| ③ memory extraction | `|chunk_states|` (new_memory ↔ each chunk state) |
| ③b new ↔ previous memory | `1` if previous memory exists else `0` |
| ⑤a local trait evidence | `|local_state_pool| + |local_memory_pool| + |existing_traits|` (only unconnected pairs) |
| ⑤b trait-centered extra relation | post-cap candidate count |
| ⑤c global state-state | `min(reservoir_size, STATE_STATE_EXTRA_REL_TOPK)` |
| ⑤d global state-memory | `min(reservoir_size, STATE_MEMORY_EXTRA_REL_TOPK)` |

**New Parameter**

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `JUDGMENT_RETRY` | 3 | retry count when judgments array is empty but was expected to be non-empty |

**Affected Documents**
- `gmem4_config.md` §13 — add `JUDGMENT_RETRY = 3`
- `gmem4_implementation.md` §4 — add retry policy line to each relation-extraction prompt section (②, ②b, ③, ③b, ⑤a, ⑤b, ⑤c, ⑤d)
- `gmem4_implementation.md` §7 — add bullet on retry policy and IRRELEVANT fallback

**Implementation Notes**

```python
def call_with_judgment_retry(prompt_fn, parser_fn, expected_count, max_retry=JUDGMENT_RETRY):
    """
    Wrapper for any relation-extraction LLM call.

    Returns: parsed judgments list (possibly empty after fallback).
    """
    for attempt in range(max_retry + 1):
        raw = prompt_fn()
        parsed = parser_fn(raw)               # JSON_RETRY already wraps this
        if expected_count == 0:
            return parsed                     # legitimate empty case
        if len(parsed.get("judgments", [])) > 0:
            return parsed                     # success
        # else: retry
    # fallback after all retries exhausted
    return {"judgments": []}                  # caller treats all expected pairs as IRRELEVANT
```

- `JUDGMENT_RETRY` is independent from `JSON_RETRY`. The two stack: each `JUDGMENT_RETRY` attempt internally allows up to `JSON_RETRY` JSON parse retries.
- When fallback returns empty judgments, the caller must NOT mark this as a hard failure; the call is considered complete with all pairs implicitly `IRRELEVANT`.
- Logging: record each retry trigger with call name, `expected_count`, attempt number, and the raw output that triggered the retry. This is useful for tuning prompt quality.

---

### Change 2: Sharpened keyword vs domain_label Distinction

**Motivation**
The base spec defines `label_set(n) = keywords(n) ∪ domain_label(n)` and uses the union for overlap scoring. In practice the LLM tends to confuse the two fields and produce overlapping content (e.g., putting "fitness" in both keywords and domain_label). This dilutes the implicit lexical-vs-topical signal.

**Decision Scope**
- Retrieval-side change: **none** (label_set union remains).
- Extraction-side change: tighten the prompt-level definitions and provide concrete examples so the LLM produces **complementary**, not overlapping, fields.
- Query-side: spaCy-based extraction remains as-is.
- Count parameters (`MAX_KEYWORDS = 5`, `MAX_DOMAIN_LABELS = 5`, `MIN_DOMAIN_LABELS = 3`): unchanged.

**Original Behavior**
- Extraction prompts ask for `keywords` and `domain_label` without sharply distinguishing them. The LLM may produce:
  - `keywords: ["marathon", "fitness", "running"]`
  - `domain_label: ["fitness", "health", "running"]`
- Substantial overlap; both fields collapse to roughly the same surface set.

**New Behavior**
- All extraction prompts that produce these two fields (state ②, memory ③, trait ④) must include the following definition block:

```text
keyword:
  Surface-level tokens that appear (or near-appear) directly in the source text.
  Concrete entities, named items, specific actions, or particular phrases.
  Example for "user runs marathons every quarter":
    keywords = ["marathon", "quarter", "running"]

domain_label:
  Abstract topical or categorical labels at a higher level of abstraction.
  Broader subject areas the node belongs to. Should NOT duplicate keywords.
  Example for "user runs marathons every quarter":
    domain_label = ["fitness", "endurance_sport", "health"]

Hard constraint:
  keywords and domain_label MUST be disjoint sets.
  Do not place the same string in both.
```

- The disjointness constraint is a soft contract enforced in the prompt; on output, if the LLM still produces overlap, the implementation removes overlap from `domain_label` (keywords win) before storing.

**Affected Documents**
- `gmem4_storage_extraction.md` — update label-field definitions (location depends on current structure of that document)
- `gmem4_implementation.md` §4.1 (②), §4.2 (③), §4.3 (④) — add the definition block above to each prompt template

**Implementation Notes**

```python
def deduplicate_labels(keywords: list[str], domain_label: list[str]) -> tuple[list[str], list[str]]:
    """
    Enforce disjointness post-hoc: if a token appears in both, keep it in keywords only.
    Case-insensitive comparison, original case preserved in keywords.
    """
    kw_lower = {k.lower() for k in keywords}
    domain_label = [d for d in domain_label if d.lower() not in kw_lower]
    return keywords, domain_label
```

- Apply `deduplicate_labels` to every node creation path (state, memory, trait).
- No change to retrieval-side `label_set`, `overlap_norm`, or scoring weights.

---

### Change 3: Symmetric SHIFT_TO Scoring (Bidirectional Sign)

**Motivation**
The base spec penalizes the old node A in `A → SHIFT_TO → B` but provides no positive signal to the new node B. The intent of SHIFT_TO is precisely "B is the current valid version", but in scoring B has no advantage over an isolated state with no evidence. This breaks the desired ranking when query embedding favors A's surface form.

**Original Behavior**
For `SHIFT_TO(A → B)` with both A and B in the pool:
- A: `con_w += 1`
- B: no effect (described as "support implicit via absence of penalty")

**New Behavior**
For `SHIFT_TO(A → B)` with both A and B in the pool:
- A (outgoing source): `con_w += 1` *(unchanged)*
- B (incoming target): `sup_w += 1` *(NEW)*

**Worked Example**
Chain `A → SHIFT_TO → B → SHIFT_TO → C`, all three in the pool, no other evidence:

| Node | sup_w | con_w | support_ratio | Interpretation |
|------|-------|-------|---------------|----------------|
| A    | 0     | 1 (out→B) | 0.0       | fully outdated |
| B    | 1 (in←A) | 1 (out→C) | 0.5    | once new, now superseded |
| C    | 1 (in←B) | 0     | 1.0           | currently valid |

**Affected Documents**
- `gmem4_retrieval.md` §3.5 — replace the "Storage / Sign reading" block:

  Old:
  > Sign reading:  B →CON→ A         (B invalidates A)
  > A's support_ratio: receives con_w += 1 (penalized as outdated)
  > B's support_ratio: unaffected (no incoming CON from this edge)

  New:
  > Sign reading: bidirectional
  > A's support_ratio: receives con_w += 1 (penalized as outdated)
  > B's support_ratio: receives sup_w += 1 (boosted as current)

- `gmem4_retrieval.md` §4.3 — update trait selection pseudocode:

  Old:
  ```text
  if ∃ SHIFT_TO(z → t) where z in pool:   # t is new — no effect on t
      pass  # support signal is implicit via absence of penalty
  ```
  New:
  ```text
  if ∃ SHIFT_TO(z → t) where z in pool:   # t is new
      sup_w += 1                            # boosted as current
  ```

- `gmem4_retrieval.md` §4.4 — same change for state selection.
- `gmem4_retrieval.md` §4.5 — no change (memory has no SHIFT_TO).
- `gmem4_retrieval.md` §7 — update summary bullet on "reversed-direction SHIFT_TO" → "bidirectional SHIFT_TO".
- `gmem4_bigflow.md` "Core Design Principles" — update the "Reversed SHIFT_TO in scoring" row:

  Old:
  > `SHIFT_TO(A→B)` is read as `B→CON→A` during support-ratio computation; old node penalized, new node unaffected

  New:
  > `SHIFT_TO(A→B)` contributes both signs during support-ratio computation: A receives `con_w += 1` (penalized as outdated), B receives `sup_w += 1` (boosted as current)

- `gmem4_implementation.md` §7 — update the matching bullet:

  Old:
  > during support-ratio computation, `SHIFT_TO(A → B)` is read as reversed-direction CON: `B →CON→ A`; this penalizes the old node and leaves the new node unaffected;

  New:
  > during support-ratio computation, `SHIFT_TO(A → B)` contributes bidirectionally: A receives `con_w += 1` (old, penalized), B receives `sup_w += 1` (new, boosted);

**Implementation Notes**
- Expansion-time behavior of SHIFT_TO is unchanged: still forward-only traversal (Rule B), still excluded from the sign table during expansion.
- The bidirectional contribution applies **only at support-ratio computation time** (final-set assembly).
- Edge case: if A is in the pool but B is not (or vice versa), still no contribution from this edge — the rule requires *both endpoints in the pool*, same as before.
- Shift-chain collapse (§4.3, §4.4) interacts naturally: with the new scoring, B and C are more likely to outrank A in top-k, so collapse rules either drop A explicitly or A simply doesn't make it in.

---

### Change 4: Reasoning and Evidence Quote in Relation Extraction

**Motivation**
The current judgment output `{a, b, relation}` provides no introspection into *why* the LLM made the call. This hurts:
- post-hoc analysis of edge quality,
- debugging of bad judgments during prompt iteration,
- snapshot-based qualitative evaluation.
Adding a short rationale plus decisive quotes per judgment makes each edge auditable, with bounded token overhead.

**Original Behavior**
- Each judgment is `{a: id, b: id, relation: str}`.
- Edges store only the relation type and timestamp metadata (and any other fields already in the schema).
- No rationale is captured at any point.

**New Behavior**
- All relation-extraction LLM calls (②, ②b, ③, ③b, ⑤a, ⑤b, ⑤c, ⑤d) extend their per-judgment output schema to:
  ```json
  {
    "a": "<node_id>",
    "b": "<node_id>",
    "relation": "SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT",
    "reasoning": "<1-2 sentences explaining the judgment>",
    "evidence_quote": {
      "a": "<short decisive snippet from node a, ≤ ~15 words>",
      "b": "<short decisive snippet from node b, ≤ ~15 words>"
    }
  }
  ```
- **Storage policy**:
  - For `SUPPORT`, `CONTRADICT`, `SHIFT_TO` judgments: store `reasoning` and `evidence_quote` as edge attributes.
  - For `IRRELEVANT` judgments: no edge is created (existing behavior); rationale and quotes are discarded after parsing. They are not retained anywhere.
- **Prompt-side use**: rationale and quotes are **not** included in the serialization for response prompting (①), QA answering (⑥), or challenged-trait conflict evidence. They exist purely as edge metadata for analysis and snapshots.
- Applies to all eight relation-extraction calls.

**New Edge Schema Fields**

| Field | Type | Description |
|-------|------|-------------|
| `rationale` | `str` | 1–2 sentence reasoning produced by the LLM at edge-creation time |
| `evidence_quote` | `{a: str, b: str}` | decisive text snippets from each endpoint node |

These are added to the existing edge attributes (relation type, timestamp, etc.).

**Affected Documents**
- `gmem4_storage_extraction.md` — extend the edge schema definition with `rationale` and `evidence_quote` fields. Specify they are populated only for non-IRRELEVANT judgments.
- `gmem4_implementation.md` §4.1, §4.1b, §4.2, §4.2b, §4.4, §4.5, §4.6, §4.7 — extend the per-judgment output schema in each prompt template; add the storage-policy bullet (IRRELEVANT excluded).
- `gmem4_implementation.md` §7 — add bullet: rationale and evidence_quote are stored on non-IRRELEVANT edges only and are excluded from any serialization that reaches ①/⑥.

**Implementation Notes**

```python
@dataclass
class JudgmentRecord:
    a: str
    b: str
    relation: Literal["SUPPORT", "CONTRADICT", "SHIFT_TO", "IRRELEVANT"]
    reasoning: str                          # always parsed
    evidence_quote: dict[str, str]          # always parsed, keys = {"a", "b"}

def apply_judgment(graph, j: JudgmentRecord):
    if j.relation == "IRRELEVANT":
        return                              # no edge, rationale discarded
    edge_attrs = {
        "relation": j.relation,
        "rationale": j.reasoning,
        "evidence_quote_a": j.evidence_quote["a"],
        "evidence_quote_b": j.evidence_quote["b"],
        # ... plus existing timestamp / metadata fields
    }
    graph.add_edge(j.a, j.b, **edge_attrs)
```

- Prompt-side: each relation-extraction prompt must instruct the LLM to produce all four fields (`relation`, `reasoning`, `evidence_quote.a`, `evidence_quote.b`) per judgment, even for `IRRELEVANT`. Discarding happens on the apply side, not the prompt side, to keep prompt instructions uniform across relation types.
- Quote length cap: keep `evidence_quote` snippets short. Recommended soft cap: ~15 words each. The prompt should include this guidance.
- Token cost estimate: ~10–15% increase in internal-output tokens across ②③④⑤. No change to external/QA tokens.
- Snapshot serialization: include the new fields in `save_snapshot()` output for offline analysis.

---

### Change 5: State Extraction — Per-Turn Frequency, Single State Cap

**Motivation**
Under `STATE_EXTRACTION_H = 2`, the LLM sees two consecutive user turns at once. When the two turns carry different persona signals, the LLM tends to either conflate them into one merged state or silently drop one of them. Moving to per-turn extraction (H=1) eliminates this conflation. With per-turn granularity, capping output at one state per call is sufficient and forces the LLM to focus on the single most decision-relevant signal.

**Original Behavior**
- `STATE_EXTRACTION_H = 2` (extract every 2 user turns).
- `STATE_MAX_COUNT = 2` (up to 2 states per call).
- A turn with no persona signal still produces a call; 0-state output is allowed but not actively guided.

**New Behavior**
- `STATE_EXTRACTION_H = 1` (extract every user turn).
- `STATE_MAX_COUNT = 1` (at most 1 state per call).
- 0-state output is **explicitly endorsed** in the prompt via the NOT-a-state examples introduced in Change 6. The LLM is instructed to skip extraction when no clear persona signal exists in the current turn.

**Affected Documents**
- `gmem4_config.md` §1 — change `STATE_EXTRACTION_H` from 2 to 1, change `STATE_MAX_COUNT` from 2 to 1.
- `gmem4_bigflow.md` "process_turn() Overview" — the comment "every h_state user turns" still applies (H is now 1, but the structural trigger doesn't change).
- `gmem4_implementation.md` §1 — same.

**Implementation Notes**
- Cost impact: number of ② calls roughly doubles. Token impact per call drops slightly (single turn input vs two turns), so net internal-token increase is below 2×.
- ②b (new ↔ previous state) trigger condition is unchanged: it runs whenever ② produces at least one new state and previous_state_ids exist.
- With `STATE_MAX_COUNT = 1`, ② will never produce more than one new state, so the `new ↔ new` pair count is always 0. The judgments array for ② is always empty by construction. Change 1's retry logic must skip ② (treat as `expected_judgment_count = 0`).
- ⑤c (state-state extra relation) becomes more important as the primary cross-state evidence path, since ② no longer generates intra-call pairs.

---

### Change 6: State Extraction — Definition and Examples in Prompt

**Motivation**
The base ② prompt defines a state in one line ("a user-specific condition that is currently or recently valid"). This is too thin to consistently distinguish states from traits, transient behaviors, and query-side noise. Adding concrete positive and negative examples sharpens the LLM's judgment and supports the per-turn 0-state behavior introduced in Change 5.

**Original Behavior**
- Definition only; no examples.
- No explicit guidance to skip non-persona turns.

**New Behavior**
- The ② prompt includes the following block (placed near the node-type description):

```text
A state is a user-specific condition that is currently or recently
valid and may change over time. Extract a state ONLY when the current
turn clearly reveals such a condition about the user.

Good state examples:
- "user is vegetarian"
- "user is planning a trip to Japan next month"
- "user works as a software engineer at a fintech startup"

NOT a state — skip extraction:
- "user said hello today"          → behavior, not a condition about the user
- "user is asking about weather"   → about the query, not the user
- "user enjoys traveling"          → this is a trait, not a state

If the current turn does not clearly reveal a state, return an empty
state list. Do not invent or speculate.
```

**Affected Documents**
- `gmem4_implementation.md` §4.1 — insert the block above into the ② prompt template.

**Implementation Notes**
- Empty state output is now an expected, normal outcome. The downstream pipeline must not treat 0 states as a failure.
- The state-vs-trait line (`"user enjoys traveling"`) is the most important boundary signal in this block; keep it intact when iterating on the prompt.

---

### Change 7: State Extraction — Reference Context Window

**Motivation**
With per-turn extraction (Change 5), single turns are often too short to disambiguate (e.g., user replying "Yeah, that's me"). Without prior context, the LLM either over-extracts on shallow tokens or fails to extract a real signal. Showing recent turns as **read-only reference**, with explicit labeling separating "current turn (extract from)" from "prior context (for understanding only)", recovers the disambiguation without inviting the LLM to mine prior turns repeatedly.

**Original Behavior**
- ② prompt contains only the recent conversation block; no explicit primary/reference distinction.
- Reference window size is implicit in the existing prompt structure.

**New Behavior**
- ② prompt is split into two clearly labeled blocks:

```text
[CURRENT TURN — extract a state from this turn only]
User: <user_utterance>
Assistant: <gt_response>

[PRIOR CONTEXT — for disambiguation only.
 DO NOT extract states from these turns. They have already been processed.]
User: <prior turn -1 user>
Assistant: <prior turn -1 assistant>
User: <prior turn -2 user>
Assistant: <prior turn -2 assistant>
User: <prior turn -3 user>
Assistant: <prior turn -3 assistant>
```

- Window size: most recent `STATE_REF_CONTEXT_TURNS` (user, assistant) pairs prior to the current turn.
- **Conversation-boundary policy**: prior context **ignores conv_id boundaries**. The most recent 3 prior pairs are shown regardless of whether they fall under a different `conv_id` (i.e., even if there is a 12-hour gap before the current turn).
- If fewer than `STATE_REF_CONTEXT_TURNS` prior pairs exist (e.g., session start), include only what is available. No padding, no placeholder turns.

**New Parameter**

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `STATE_REF_CONTEXT_TURNS` | 3 | number of prior (user, assistant) pairs shown as read-only reference in ② |

**Affected Documents**
- `gmem4_config.md` §1 — add `STATE_REF_CONTEXT_TURNS = 3`.
- `gmem4_implementation.md` §4.1 — replace the existing prompt-content note ("Prompt contains only the recent conversation block") with the CURRENT TURN / PRIOR CONTEXT split shown above.

**Implementation Notes**
- The reference window is sourced from the same context cache used elsewhere (the `k0`-sized recent-pairs buffer). No new storage.
- The `[PRIOR CONTEXT — DO NOT extract from these]` label is essential. In practice, LLMs comply with such explicit labeling, but if drift is observed during evaluation, strengthen the line further (e.g., repeat the instruction at the end of the prompt).
- This change applies **only to ②**. ③ (memory) and ④ (trait) already operate on richer chunked input and need no reference window.

---

### Change 8: Trait Extraction — Definition and Examples in Prompt

**Motivation**
Trait extraction (④) suffers from the same definitional thinness as ②: the boundary between trait, state, and one-off behavior is fuzzy without examples. Adding a sharper definition centered on the "in-general" test, plus a small set of positive examples and direct state/trait comparisons, improves the produced traits' generality and reduces accidental state extraction in the trait slot.

**Original Behavior**
- Trait defined as "a generalized user characteristic that persists across situations and time."
- No examples; no operational test.

**New Behavior**
- The ④ prompt includes the following block:

```text
A trait is a generalized user characteristic that persists across
situations and time. Use the "in general" test: a trait should still
be true if you asked the user about themselves "in general" with no
specific time, place, or context attached.

Good trait examples:
- "user is detail-oriented and prefers thorough explanations"
- "user enjoys outdoor adventure activities"
- "user is risk-averse in financial decisions"
- "user values directness over politeness"

State vs Trait — these pairs show the boundary:
- "user is currently in Tokyo"             → state (changes when they leave)
- "user enjoys traveling internationally"  → trait (persistent disposition)

- "user has a project deadline next Friday" → state
- "user works hard under pressure"          → trait
```

**Affected Documents**
- `gmem4_implementation.md` §4.3 — insert the block above into the ④ prompt template.

**Implementation Notes**
- The "in general" test is the single most useful operational anchor in this block; surface it before the examples for best effect.
- `TRAIT_MAX_COUNT = 1` remains unchanged — at most one trait per 2-chunk boundary.
- No structural change to ④ (still triggered every 2 chunks; still followed by ⑤a/⑤b when extra-relation extraction is enabled).

---

### Change 9: APS Definition Refactor — Impact-Only, Top-k by Seed Score, Disjoint from Seed States

**Motivation**
The base APS definition has two structural issues. First, requiring `scope = BROAD` *and* `current_decision_impact = HIGH` together creates a false negative for NARROW + HIGH states (e.g., "user is recovering from knee surgery this month") that are time-bounded but still critical for the next response. Decision-relevance is what `current_decision_impact` was designed to measure; layering `scope` on top of it conflates two orthogonal axes. Second, unconditional inclusion (with no query relevance signal) injects high-impact states into prompts even when the query is unrelated, polluting the response context.

The new definition treats `current_decision_impact` as the only membership criterion and adds query-relevance via simple top-k ranking on the existing `seed_score`. This avoids threshold tuning while still letting truly irrelevant HIGH states drop out (when many HIGH candidates exist and the query favors others).

**Original Behavior** (`gmem4_retrieval.md` §2.7)
- APS candidates: `scope = BROAD` AND `current_decision_impact = HIGH`.
- Included unconditionally, regardless of query.
- SHIFT_TO sources excluded.
- If candidate count > `k_aps`, keep most recent `k_aps`.
- APS members may also appear in `s_seed` (no exclusion).

**New Behavior**
- APS candidates: `current_decision_impact = HIGH` only. `scope` is no longer a membership criterion.
- SHIFT_TO sources excluded *(unchanged)*.
- Candidates ranked by `seed_score(s, query)` — the same scope-dependent score used elsewhere.
- Top `k_aps` by seed score selected as APS.
- **Disjointness with `s_seed`**: APS members are excluded from the `s_seed` candidate pool.
- HIGH states that were ranked outside the APS top-k are **not discarded**: they fall through to the `s_seed` pool and compete normally there. Every state still has a chance to be retrieved, just not in two places at once.

**Affected Documents**
- `gmem4_retrieval.md` §2.5 — State seed retrieval: add a note that APS members are excluded from the candidate pool prior to top-`k_s` selection.
- `gmem4_retrieval.md` §2.7 — Replace the APS construction description:

  Old:
  > `aps` is constructed from states satisfying:
  > - `scope = BROAD`
  > - `current_decision_impact = HIGH`
  > These states are included **unconditionally**, regardless of seed score.

  New:
  > `aps` is constructed by:
  > 1. taking all states with `current_decision_impact = HIGH`,
  > 2. excluding states that are SHIFT_TO sources,
  > 3. ranking the remaining candidates by `seed_score(s, query)`,
  > 4. selecting the top `k_aps`.
  >
  > APS members are excluded from the `s_seed` candidate pool. HIGH states that are not selected for APS (because more than `k_aps` HIGH candidates exist) remain in the `s_seed` pool and compete there by normal seed scoring.

- `gmem4_bigflow.md` "Core Design Principles" — update the APS-related lines:

  Old:
  > **APS for implicit persona reasoning**: broad, strictly assigned high-impact states bypass ordinary seed scoring

  New:
  > **APS for implicit persona reasoning**: high-impact states are surfaced via a dedicated top-`k_aps` slot ranked by seed score; APS and `s_seed` are disjoint partitions of the state pool

- `gmem4_implementation.md` §7 — update the APS-related bullet:

  Old:
  > APS is filtered once, at construction time: any state with an outgoing `SHIFT_TO` edge is excluded. Because expansion does not create new edges, surviving APS members remain `SHIFT_TO`-terminal throughout the retrieval call, so no post-expansion re-filter is performed.

  New:
  > APS is constructed at retrieval time as the top-`k_aps` HIGH-impact, non-SHIFT_TO-source states ranked by seed score. APS members are excluded from `s_seed`. HIGH states beyond rank `k_aps` fall through to `s_seed` and compete normally. APS construction does not look at `scope`; `scope` only affects seed-score weights.

- `gmem4_config.md` §6 — `APS_EXCLUDE_SHIFT_SOURCE` remains; the comment for `k_aps` should clarify that it is the size of a top-k slot, not a cap on a candidate set.

**Implementation Notes**

```python
def construct_aps(states, query, k_aps):
    """
    Returns top-k_aps HIGH-impact, non-SHIFT_TO-source states ranked by seed_score.
    """
    candidates = [s for s in states
                  if s.current_decision_impact == "HIGH"
                  and not has_outgoing_shift_to(s)]
    candidates.sort(key=lambda s: seed_score(s, query), reverse=True)
    return candidates[:k_aps]

def seed_retrieve_states(all_states, query, k_s, aps_set):
    """
    Standard top-k_s state retrieval, excluding states already in APS.
    """
    aps_ids = {s.id for s in aps_set}
    pool = [s for s in all_states if s.id not in aps_ids]
    pool.sort(key=lambda s: seed_score(s, query), reverse=True)
    return pool[:k_s]
```

- The seed-score function used here is the same one defined in `gmem4_retrieval.md` §2.5, including its scope-dependent weights. NARROW + HIGH and BROAD + HIGH candidates are scored with their respective weights and ranked together.
- Edge case: if no HIGH candidates exist after SHIFT_TO filtering, APS is empty. This is correct and intended.
- Edge case: if HIGH candidates exist but the top-ranked one has very low `seed_score` (no other competitors), it still enters APS. This is the implicit-persona-reasoning case the design is meant to support.
- Tooling/snapshot: log both APS contents and the HIGH candidates that fell through to `s_seed`, for offline analysis of the partition.

---

### Change 10: Remove scope/impact Labels from Node Listing Format in Prompts

**Motivation**
The current node-listing format in extraction and judgment prompts inlines `(scope, current_decision_impact)` (or `(scope)` for memory and trait) next to each node:

```text
[id] [40 minutes ago] (NARROW, LOW): The user is interested in ...
```

Two problems with this:
1. The labels are system-internal annotations. The LLM is not asked to *use* them when reasoning about pairs; they take up context length without changing the LLM's task.
2. For default-valued labels (`NARROW`, `LOW`), the annotation is especially noisy — it tells the LLM nothing actionable.

Removing the inline labels entirely simplifies the prompt and reduces token usage with no expected loss in judgment quality. The labels themselves are still produced and stored on the nodes; they just are not surfaced into prompts that consume node listings.

**Original Behavior** (`gmem4_prompt_updated.md` §3.2, §3.3)
- §3.2 node format:
  ```text
  State:  [{nid}] [{elapsed}] ({scope}, {current_decision_impact}): {content}
  Memory: [{nid}] [{elapsed}] ({scope}): {content}
  Trait:  [{nid}] [{elapsed}] ({scope}): {content}
  ```
- §3.3 ⑤c pair format includes `({scope_a}, {impact_a})` per side.
- §3.3 ⑤d pair format includes `({scope_s})` and `({scope_m})`.

**New Behavior**
- §3.2 node format becomes label-free across all three node types:
  ```text
  State:  [{nid}] [{elapsed}]: {content}
  Memory: [{nid}] [{elapsed}]: {content}
  Trait:  [{nid}] [{elapsed}]: {content}
  ```
- §3.3 ⑤c pair format:
  ```text
  Pair {idx}
  - source: [{id_a}] [{elapsed_a}]: {content_a}
  - target: [{id_b}] [{elapsed_b}]: {content_b}
  ```
- §3.3 ⑤d pair format:
  ```text
  Pair {idx}
  - state:  [{s_id}] [{elapsed_s}]: {content_s}
  - memory: [{m_id}] [{elapsed_m}]: {content_m}
  ```

This affects every prompt that lists nodes via `_format_*_list`: ②b, ⑤a, ⑤b, ⑤c, ⑤d, plus any listing block in ④ (recent states/memories/previous trait).

The `Relevant States` section of the QA serialization (§3.4) is already label-free in the base spec; no change there.

**Schema and storage are not affected.** Nodes still carry `scope` and `current_decision_impact` fields. Only the prompt-side rendering is changed.

**Affected Documents**
- `gmem4_prompt_updated.md` §3.2 — replace the format spec as shown above.
- `gmem4_prompt_updated.md` §3.3 — replace both pair format specs as shown above.

**Implementation Notes**
- Single point of change: `_format_node_for_prompt` (or whatever the rendering helper is named) drops the `(scope, ...)` portion.
- LLM extraction prompts that *produce* `scope` and `current_decision_impact` values (②, ③, ④) are unchanged — those fields are still extracted and stored.
- Token saving: small per node (≈10 chars) but compounds across long listings (⑤a, ⑤c with 10+ nodes).

---

### Change 11: Replace "chunk" with Natural-Language Equivalents in LLM-Facing Prompts

**Motivation**
"Chunk" is a system-internal abstraction (a `conv_id`-aligned grouping unit). It appears in several LLM-facing prompts as if it were a known concept, forcing the LLM to either ignore it or guess what it means. Since one chunk corresponds to one conversation under the current chunking scheme (`CHUNK_SIZE_CONV = 1`), "conversation" is a precise and familiar substitute.

**Original Behavior**
"chunk" appears in the following LLM-facing locations (`gmem4_prompt_updated.md`):

| Location | Original text |
|----------|---------------|
| §2.1 `_NODE_TYPE_DESC` Memory definition | "An episodic summary of what happened during a conversation chunk." |
| §6 ③ user prompt header | `[Conversation Chunk]` |
| §6 ③ user prompt header | `[Chunk States]` |
| §6 ③ user prompt body | "summarizes what happened in this chunk" |
| §8 ④ user prompt header | `[Recent 2-Chunk Conversation]` |
| §8 ④ user prompt header | `[States from Recent 2 Chunks]` |
| §8 ④ user prompt header | `[Memories from Recent 2 Chunks]` |

**New Behavior**

| Location | New text |
|----------|----------|
| §2.1 Memory definition | "An episodic summary of what happened during a recent conversation." |
| §6 ③ header | `[Recent Conversation]` |
| §6 ③ header | `[States from this Conversation]` |
| §6 ③ body | "summarizes what happened in this conversation" |
| §8 ④ header | `[Recent Two Conversations]` |
| §8 ④ header | `[States from the Recent Two Conversations]` |
| §8 ④ header | `[Memories from the Recent Two Conversations]` |

**System-internal documentation** (spec docs `gmem4_*.md` other than `gmem4_prompt_updated.md`) may continue to use "chunk" as a system term. The replacement applies only to text that reaches the LLM.

**Affected Documents**
- `gmem4_prompt_updated.md` §2.1 — Memory definition line.
- `gmem4_prompt_updated.md` §6 — ③ user prompt headers and body.
- `gmem4_prompt_updated.md` §8 — ④ user prompt headers.

**Implementation Notes**
- Naming overlap with §3.4: the QA serialization has a `[Recent Conversation]` section as well, which renders the context cache. The ③ prompt's `[Recent Conversation]` header refers to the chunk being summarized. These two sections live in different prompts (③ is an extraction call, the QA serialization feeds ⑥) and never appear together, so the name reuse is acceptable. If readability across the codebase is a concern, ③ can use `[Conversation to Summarize]` instead. This document leaves it as `[Recent Conversation]` for now; flag if a rename is preferred.
- Implementation should grep the actual prompt template strings for any remaining occurrence of "chunk" beyond the locations cataloged above and apply the same replacement principle (system term → natural-language equivalent).

---

### Change 12: QA Prompt — Strengthened Task Description, Neutral Tone, Negative Instruction

**Motivation**
The base QA prompts (`gmem4_prompt_updated.md` §13) communicate the retrieval-section semantics but do not explicitly tell the LLM (i) that the user information is meant to *personalize* the answer, (ii) that the answer should be concise, or (iii) that meta-commentary about the memory itself ("based on your past conversations…") should be avoided. Adding these instructions, while keeping the tone neutral and not changing the output schemas, raises QA quality without affecting the evaluation pipeline.

**Original Behavior**

`SYS_QA_OPPOSED`:
```text
You are a helpful assistant answering a question about a user based on their memory.
{node_type_desc}
Traits without contradicting evidence are listed under [Traits]; those with contradicting evidence are under [Challenged Traits] with the conflicting evidence shown.
Current Constraints are broad, high-impact states that should be prioritized.
When Challenged traits conflict with listed evidence, weigh the evidence carefully.
Respond in JSON format.
```

`QA_PROMPT_OPPOSED` ends with: `Answer concisely (maximum 100 words).`

`SYS_QA_SUPPORTIVE` is structurally similar; user prompt enforces yes/no enum.

**New Behavior**

**`SYS_QA_OPPOSED`** (replace):
```text
You are a helpful assistant who has been talking with this user across multiple sessions.
The information below describes what is known about the user from past conversations.
{node_type_desc}
Traits without contradicting evidence are listed under [Traits]; those with contradicting evidence are under [Challenged Traits] with the conflicting evidence shown.
Current Constraints are high-impact user states that should be prioritized in the response.
When Challenged Traits conflict with listed evidence, weigh the evidence carefully.

Task:
- Use the user information when it is relevant to the question.
- Give an answer that fits this specific user when relevant; otherwise answer normally.
- Answer naturally. Do not explicitly reference the memory (e.g., do not say "based on what I remember" or "according to your past conversations").

Respond in JSON format.
```

**`QA_PROMPT_OPPOSED`** (user prompt) — replace the trailing length instruction:
```text
[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

Answer in 1-2 sentences.
```
*(replaces "Answer concisely (maximum 100 words).")*

**`SYS_QA_SUPPORTIVE`** (replace):
```text
You are a helpful assistant answering a yes/no question about a user based on past conversations with them.
The information below describes what is known about the user.
{node_type_desc}
Traits without contradicting evidence are listed under [Traits]; those with contradicting evidence are under [Challenged Traits] with the conflicting evidence shown.
Current Constraints are high-impact user states that should be prioritized.
When Challenged Traits conflict with listed evidence, weigh the evidence carefully.

Task:
- Use the user information when it is relevant.
- Answer based on the user information; if irrelevant, fall back to general reasoning.

Respond in JSON format.
```

**`QA_PROMPT_SUPPORTIVE`** (user prompt) — unchanged:
```text
[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

Answer the yes/no question based only on the information in memory.
You MUST answer with exactly one of: yes or no.
```

**Output schemas are unchanged** for both subsets:
- opposed: `{ "answer": "..." }` (free string)
- supportive: `{ "answer": "yes" | "no" }` (enum)

**Affected Documents**
- `gmem4_prompt_updated.md` §13.1 — replace `SYS_QA_OPPOSED` and modify the trailing line of `QA_PROMPT_OPPOSED`.
- `gmem4_prompt_updated.md` §13.2 — replace `SYS_QA_SUPPORTIVE`. `QA_PROMPT_SUPPORTIVE` stays.
- `gmem4_implementation.md` §4.8 — note that ⑥ uses two distinct system prompts keyed by `subset`.

**Implementation Notes**
- The phrasing "Current Constraints are high-impact user states" replaces the older "broad, high-impact states" wording, consistent with Change 9 (APS no longer requires BROAD).
- "Answer in 1-2 sentences" is more restrictive than "maximum 100 words"; the new instruction supersedes the old one. If the existing post-processing expects a 100-word window, it should still pass — the new constraint is strictly tighter.
- Negative instruction wording ("do not say 'based on what I remember' …") is a known-effective pattern; if the LLM still inserts meta-commentary in evaluation, strengthen further by repeating the instruction at the end of the user prompt.
- No change to retry logic, schema, or post-processing.

---

### Change 13: Prompt Deduplication — Remove System/User Overlap and Unify Shared Wording

**Motivation**
Across all LLM calls, several blocks were duplicated between system and user
prompts (most notably `_NODE_TYPE_DESC` and `_EVID_DESC_FULL`'s calibration
rules) and self-redundant sections appeared inside individual user prompts
(e.g., a `Final instruction:` block restating earlier task constraints).
Also, semantically identical guidance across the four ⑤ calls and ②b was
worded inconsistently. The goal is to make each prompt section express its
content exactly once.

**Original Behavior**
- `_NODE_TYPE_DESC` was duplicated verbatim in `updater.py` and
  `generator.py` and contained a `Quick test:` block restating the four-line
  definition.
- `_EVID_DESC_FULL` carried a `Calibration rules:` block whose items also
  appeared in each call's user prompt.
- `_LABEL_DISCIPLINE_BLOCK` contained a five-line `Self-check:` paragraph
  that restated the `domain_label` definition.
- ② / ②b / ③ / ④ user prompts ended with `Final instruction:` blocks
  restating earlier task constraints. ② additionally restated the State
  definition and a 3-question `Self-check:`.
- ⑤a body and final instruction used `for each required pair` while
  ⑤b/⑤c/⑤d used different wording. ②b had `is not SUPPORT` (lowercase)
  while ⑤a/⑤b/⑤c/⑤d had `is NOT SUPPORT` (uppercase).
- ②b vs ⑤c SHIFT_TO calibration items differed in wording.

**New Behavior**
- `_NODE_TYPE_DESC` is defined once in `generator.py`. `updater.py` imports
  it (`from generator import _NODE_TYPE_DESC`). The `Quick test:` block is
  removed.
- `_EVID_DESC_FULL` carries only the four label definitions; the
  `Calibration rules:` block is removed.
- `_LABEL_DISCIPLINE_BLOCK` is shortened: the `Self-check:` paragraph and
  the casing/singular-plural Hard-constraint line (already enforced by
  `deduplicate_labels`) are removed. The keywords/domain_label definition
  lines are tightened.
- ② user prompt removes the State definition first paragraph, the
  `Self-check:` 3-question block, the mid-prompt `Assign placeholder ids:`
  line, the `describe a currently or recently valid user-specific persona
  signal` line, and the `Final instruction:` block. scope BROAD/NARROW are
  collapsed to a single line each.
- ②b user prompt is restructured: the previous separate
  `For previous↔new:` / `For new↔new:` / `Relation calibration` /
  `SHIFT_TO caution` blocks are consolidated into `Direction rules:` +
  `new↔new pair caution:` + a single `Calibration:` block; the
  `Final instruction:` block is removed.
- ③ user prompt removes the `Final instruction:` block. scope BROAD/NARROW
  collapse to single lines each.
- ④ user prompt removes the first sentence of the Trait definition and the
  two redundant trait-criterion lines that restate `_NODE_TYPE_DESC`'s Trait
  definition; the `Final instruction:` block is removed.
- ⑤a/⑤b/⑤c/⑤d shared phrasings are aligned: similarity-mining preamble,
  `Topical similarity alone is NOT SUPPORT.` (uppercase NOT, also applied to
  ②b), CONTRADICT calibration meta line, output-count phrasing
  (`Output exactly N judgments — one for each listed {pair|candidate}, in
  the listed order.`), and ②b/⑤c SHIFT_TO calibration wording.
- ⑤a/⑤b/⑤c/⑤d Final instruction blocks are kept (per-call output-count
  reinforcement is intentional).

**Affected Documents**
- `gmem5_prompt.md` §2.1, §2.2a, §2.3 — replace component bodies.
- `gmem5_prompt.md` §4 (②), §5 (②b), §6 (③), §8 (④) — replace user
  prompts.
- `gmem5_prompt.md` §9 (⑤a), §10 (⑤b), §11 (⑤c), §12 (⑤d) — apply shared
  wording.

**Implementation Notes**
- All apply-side parsers (`_validate_relation`, `_validate_relation_reduced`,
  schema enforcement) are unchanged; this is a prompt-only refactor.
- `_NODE_TYPE_DESC` is now imported in `updater.py` from `generator.py` to
  prevent future divergence between QA-side and extraction-side wording.

---

### Change 14: QA Prompt + Retrieval Serialization Cleanup; Trait↓ State↑ Rebalance

**Motivation**
Refinements after Change 13:
- (a) the multichoice QA system prompt carried a 4-line Challenged Trait
  policy that could be compressed to a single sentence;
- (b) the retrieval serialization always emitted every section header
  (including a `(none)` body for empty sections), and the
  `[Relevant States]` / `[Relevant Memories]` headers had no description
  while only `[Current Constraints]` did;
- (c) state pool was undersized vs. `K_APS + K_SF`.

**Original Behavior**
- `SYS_QA_MULTICHOICE` carried a 4-line Challenged Trait policy block.
- `_serialize` emitted every section header even when empty, with a
  `(none)` body line.
- `[Relevant States]` and `[Relevant Memories]` headers had no header_note.
- `K_SF = 8`, `K_STATE = 10`. The state pool `K_STATE` was smaller than
  `K_APS + K_SF = 11`, marginally insufficient.

**New Behavior**
- The 4-line Challenged Trait policy in `SYS_QA_MULTICHOICE` is compressed
  to a single sentence:
  `[Traits] are currently reliable; [Challenged Traits] may be outdated,
  replaced, or contradicted. For a challenged trait, prefer any "shifted to"
  entry and weigh listed conflicting evidence before using the trait.`
  (Note: personamem has only the multichoice QA system prompt — there is no
  separate `_QA_COMMON_RETRIEVAL_GUIDE` constant as in the implexconv
  variant, since there is no second QA system prompt to share with.)
- `_serialize`'s inner `add_lines(title, lines, header_note)` now skips
  emitting the section entirely when `lines` is empty — no header, no
  `(none)` line, no blank-line separator.
- `[Relevant States]` and `[Relevant Memories]` carry header_note lines:
  - `[Relevant States]`: *Additional retrieved user states pertaining to the
    current question; not necessarily high-impact.*
  - `[Relevant Memories]`: *Episodic summaries of past conversations
    relevant to the current question.*
- Retrieval count parameters in `config_0.py` are rebalanced:
  - `K_SF`: `8 → 12`
  - `K_STATE`: `10 → 17` (pool now ≥ `K_APS + K_SF = 15`, restoring
    sufficiency)
- `K_T_FINAL` is NOT introduced. Personamem keeps its stability/τ-based
  trait classification (`traits_stable` / `traits_tentative` /
  `traits_challenged`) and does not use a unified `W_SR` final scoring.

---

### Change 15: Per-Call Output Caps and Per-Session LLM Call Logging Filter

**Motivation**
Two operational fixes:
- (a) every internal call (②, ②b, ③, ③b, ④, ⑤a, ⑤b, ⑤c, ⑤d) shared a
  single output cap `MAX_TOKENS_INTERNAL = 10000`, far above what most calls
  ever produce;
- (b) per-call prompt/output logging was guarded by a hardcoded
  `context_index in (0, 1, 2)` check inside `run_experiment.py`, with no
  config hook.

**Original Behavior**
- All 9 internal calls used `cfg.MAX_TOKENS_INTERNAL` (= 10000).
- LLM call logging gated by
  `if cfg.ENABLE_LLM_CALL_LOGGING and context.context_index in (0, 1, 2)`.

**New Behavior**
- `MAX_TOKENS_INTERNAL` is removed. Each of the 9 internal calls now uses
  its own per-call cap:

  | Constant | Value | Used by |
  |----------|-------|---------|
  | `MAX_TOKENS_STATE` | 1000 | ② state extraction |
  | `MAX_TOKENS_STATE_NEW_REL` | 800 | ②b state new-relation judgments |
  | `MAX_TOKENS_MEMORY` | 1000 | ③ memory extraction |
  | `MAX_TOKENS_MEMORY_NEW_REL` | 2000 | ③b memory new-relation judgments |
  | `MAX_TOKENS_TRAIT` | 1200 | ④ trait extraction |
  | `MAX_TOKENS_TRAIT_EVIDENCE_5A` | 2500 | ⑤a local trait evidence |
  | `MAX_TOKENS_TRAIT_EXTRA_REL_5B` | 1500 | ⑤b trait extra relations |
  | `MAX_TOKENS_STATE_STATE_5C` | 1000 | ⑤c state-state relations |
  | `MAX_TOKENS_STATE_MEMORY_5D` | 800 | ⑤d state-memory relations |

- A new config parameter `LLM_CALL_LOG_FIRST_N_SESSIONS` (default `10`)
  replaces the hardcoded session filter. Per-call prompt/output logging is
  active iff `cfg.ENABLE_LLM_CALL_LOGGING` is `True` AND
  `context.context_index < cfg.LLM_CALL_LOG_FIRST_N_SESSIONS`.

---

### Change 16: Scope NARROW Prior, IRRELEVANT-as-Edge, Trait in c-Seed Expansion, Batched JUDGMENT_RETRY, Evidence-Quote Rollback

**Motivation**
A bundle of corrections plus a rollback of Change 4's auditability fields
(personamem-specific decision):
- (a) `scope` was nearly 100% `BROAD` because the calls had no
  `If uncertain → NARROW` prior.
- (b) ⑤b/⑤c/⑤d candidate selection (gated by `has_direct_edge`) re-judged
  pairs that had already been classified `IRRELEVANT`, because IRRELEVANT
  judgments produced no edge.
- (c) traits with no direct seed but a `SOURCE` connection from a
  c-seed could be silently missed during graph expansion: Step 1 of
  expansion only admitted `m` and `s` from `get_source_children`, even
  though ④ creates `SOURCE` edges into traits as well.
- (d) the batched runner (`run_experiment.py:_drain_pending_calls`)
  bypassed `JUDGMENT_RETRY` entirely; only the sequential path
  (`updater.py:execute_call`) honored it.
- (e) Change 4's `reasoning` and `evidence_quote.{a,b}` fields are dropped
  from all relation-extraction outputs (see "Evidence-quote rollback"
  below).

**Original Behavior**
- ②/③/④ scope blocks contained no `If uncertain → NARROW` prior.
- Apply-side `_store_evidence_edge` returned early on IRRELEVANT, so no edge
  was stored.
- `retriever.py` Step 1 c-seed → SOURCE expansion filtered children to
  `{NODE_M, NODE_S}`, dropping any `NODE_T`.
- `_drain_pending_calls` ran one batched generation per call_type group
  with no empty-judgments retry.
- `_JUDGMENT_OBJECT` and `_STATE_MEMORY_JUDGMENT_OBJECT` required
  `reasoning` and `evidence_quote.{a,b}` fields; `add_evidence_edge`
  required keyword-only `rationale`/`evidence_quote_a`/`evidence_quote_b`
  arguments and stored them in `_evid_metadata`.

**New Behavior**
- A single line `If uncertain between BROAD and NARROW, choose NARROW.` is
  appended after the `scope:` definition block in ②, ③, and ④ user
  prompts.
- `add_evidence_edge` no longer drops IRRELEVANT. All `_apply_*_result`
  methods route through the `_store_evidence_edge` helper, which now stores
  the edge unconditionally. `out_sup/out_con/in_sup/in_con` counters and
  sign propagation continue to exclude IRRELEVANT, so support/contradict
  accounting is unchanged.
- `retriever.py` Step 1 c-seed → SOURCE expansion now admits
  `{NODE_M, NODE_S, NODE_T}`. Steps 2a/2b/2c are unchanged.
- `PendingLLMCall` gains `expected_pairs: List[Tuple[str, str]]`. Each
  relation-extraction builder (②b/③b/⑤a/⑤b/⑤c/⑤d) populates this list at
  construction time using the call's canonical direction:
  - ②b: `prev_state → new_state`, then `new(earlier) → new(later)`
  - ③b: `new_memory → previous_memory`, then `new_memory → chunk_state`
  - ⑤a: `state/memory/prev_trait → new_trait`
  - ⑤b: `candidate → new_trait`
  - ⑤c: `older_state → newer_state` (chronological)
  - ⑤d: `m → s` (storage canonical)
- `GraphUpdater.apply_irrelevant_fallback(call)` (and its module-level
  passthrough on `GraphMemModule`) is added. It iterates
  `call.expected_pairs` and stores an `IRRELEVANT` edge for each, with
  defensive guards: skip if either endpoint is missing or if a direct edge
  already exists.
- `_drain_pending_calls` mirrors the sequential `JUDGMENT_RETRY` policy:
  after each batched generation, identify jobs with
  `expected_judgment_count > 0` AND `judgments == []`; re-batch with an
  appended hint
  `"Previous attempt returned empty judgments; you MUST output exactly N
  judgments."`; up to `cfg.JUDGMENT_RETRY` retries; retry token usage
  accumulates on top of the original; on exhaustion,
  `module.apply_irrelevant_fallback(call)` is invoked.
- The sequential path (`execute_call`) also appends the same retry hint and
  invokes `apply_irrelevant_fallback` on exhaustion (previously it only
  retried without a hint).
- **Evidence-quote rollback**: `reasoning` and `evidence_quote.{a,b}` are
  removed from `_JUDGMENT_OBJECT`, `_STATE_MEMORY_JUDGMENT_OBJECT`, and
  every relation-extraction prompt's "Output requirements" section.
  `add_evidence_edge` no longer accepts `rationale`/`evidence_quote_a`/
  `evidence_quote_b`; `_evid_metadata`, `get_evidence_metadata`, and the
  associated JSON I/O are removed from `graph_store.py`. The `_audit_fields`
  helper in `updater.py` is removed. Change 4's auditability schema is
  effectively rolled back; the IRRELEVANT-as-edge logic in this change
  supersedes Change 4's "no edge for IRRELEVANT" rule.

**Implementation Notes**
- `apply_irrelevant_fallback` stores IRRELEVANT only when neither node is
  missing AND no direct edge already exists. This avoids overwriting a
  SUPPORT/CONTRADICT/SHIFT_TO edge produced by a partial retry.
- Personamem keeps its string-id system; `_resolve_id` is unchanged.
  Implexconv's defensive integer-id normalization is N/A here.
- Storage volume increases — every previously-judged-IRRELEVANT pair now
  has an edge. In return, ⑤b/⑤c/⑤d LLM-call volume drops because candidate
  pools shrink to truly unjudged pairs.

---

## New Configuration Parameters Summary

| Parameter | Value | Source change | Description |
|-----------|-------|---------------|-------------|
| `JUDGMENT_RETRY` | 3 | Change 1 | retry count when judgments are missing but expected |
| `STATE_REF_CONTEXT_TURNS` | 3 | Change 7 | prior (user, assistant) pairs shown as read-only reference in ② |
| `MAX_TOKENS_STATE` / `_STATE_NEW_REL` / `_MEMORY` / `_MEMORY_NEW_REL` / `_TRAIT` / `_TRAIT_EVIDENCE_5A` / `_TRAIT_EXTRA_REL_5B` / `_STATE_STATE_5C` / `_STATE_MEMORY_5D` | 1000 / 800 / 1000 / 2000 / 1200 / 2500 / 1500 / 1000 / 800 | Change 15 | per-call output caps replacing the shared `MAX_TOKENS_INTERNAL` |
| `LLM_CALL_LOG_FIRST_N_SESSIONS` | 10 | Change 15 | log per-call prompts/outputs only for `context_index < this` |

## Modified Configuration Parameters Summary

| Parameter | Before | After | Source change |
|-----------|--------|-------|---------------|
| `STATE_EXTRACTION_H` | 2 | 1 | Change 5 |
| `STATE_MAX_COUNT` | 2 | 1 | Change 5 |
| `K_SF` | 8 | 12 | Change 14 |
| `K_STATE` | 10 | 17 | Change 14 |
| `MAX_TOKENS_INTERNAL` | 10000 | (removed; per-call caps) | Change 15 |

## Edge Schema Fields Summary

Change 4 introduced `rationale` and `evidence_quote.{a,b}` on EVIDENCE edges
for SUPPORT / CONTRADICT / SHIFT_TO. **These fields were rolled back in
Change 16**: relation-extraction outputs no longer carry them, and
`graph_store.HeterogeneousGraph` no longer stores `_evid_metadata`. The
edge subtype is the only persisted classification; IRRELEVANT edges are
now stored too (Change 16) so candidate selection can skip them.

## Modified Behaviors Summary

| Behavior | Before | After | Source change |
|----------|--------|-------|---------------|
| Empty judgments handling | silently accepted | retry then IRRELEVANT fallback | Change 1 |
| keyword / domain_label fields | may overlap | enforced disjoint via prompt + post-hoc | Change 2 |
| SHIFT_TO scoring | one-sided (only A penalized) | bidirectional (A penalized, B boosted) | Change 3 |
| Relation-extraction output | `{a, b, relation}` | `{a, b, relation, reasoning, evidence_quote}` | Change 4 |
| State extraction frequency | every 2 user turns | every user turn | Change 5 |
| State count per call | up to 2 | exactly up to 1 | Change 5 |
| State prompt definition | single line | definition + good/not examples + skip rule | Change 6 |
| State prompt context | flat recent block | CURRENT TURN / PRIOR CONTEXT split | Change 7 |
| Trait prompt definition | single line | definition + "in general" test + examples + state/trait pairs | Change 8 |
| APS membership criterion | `BROAD AND HIGH`, unconditional | `HIGH` only, top-k by seed_score | Change 9 |
| APS / s_seed overlap | may overlap | mutually disjoint | Change 9 |
| Node listing format in prompts | `[id] [elapsed] (scope, impact): content` | `[id] [elapsed]: content` | Change 10 |
| "chunk" in LLM prompts | system term exposed | replaced with "conversation" / natural language | Change 11 |
| QA opposed system prompt | minimal task description | personalization task + neutral tone + negative instruction | Change 12 |
| QA opposed length instruction | "maximum 100 words" | "1-2 sentences" | Change 12 |
| QA supportive system prompt | minimal task description | personalization task + neutral tone | Change 12 |
| `_NODE_TYPE_DESC` source | duplicated in `updater.py` and `generator.py` (with `Quick test:` block) | defined in `generator.py`; `updater.py` imports; no `Quick test:` block | Change 13 |
| `_EVID_DESC_FULL` content | label definitions + `Calibration rules:` block | label definitions only (calibration moved to per-call user prompts) | Change 13 |
| `_LABEL_DISCIPLINE_BLOCK` size | ~22 lines (with `Self-check:` paragraph) | ~13 lines (no `Self-check:`) | Change 13 |
| Per-call `Final instruction:` blocks | ②, ②b, ③, ④ ended with redundant restatement | removed in ②, ②b, ③, ④; kept in ⑤a/⑤b/⑤c/⑤d for output-count reinforcement | Change 13 |
| ②b user-prompt structure | separate `For previous↔new:` / `For new↔new:` / `Relation calibration` / `SHIFT_TO caution` blocks | unified `Direction rules:` + `new↔new pair caution:` + single `Calibration:` block | Change 13 |
| Cross-call shared phrasings | inconsistent (case, wording, line wrap) across ②b/⑤a/⑤b/⑤c/⑤d | identical wording for similarity-mining preamble, `is NOT SUPPORT`, CONTRADICT calibration, output-count phrasing, and ②b/⑤c SHIFT_TO calibration | Change 13 |
| QA Challenged Trait policy | 4 lines hardcoded in `SYS_QA_MULTICHOICE` | compressed to one sentence | Change 14 |
| Empty serialization sections | header emitted with `(none)` body line | section omitted entirely | Change 14 |
| `[Relevant States]` / `[Relevant Memories]` headers | no description note | one-line header_note disambiguating from `[Current Constraints]` / `[Stable Traits]` | Change 14 |
| Internal-call output cap | single `MAX_TOKENS_INTERNAL = 10000` shared by ②/②b/③/③b/④/⑤a/⑤b/⑤c/⑤d | 9 per-call caps (800-2500), each ~3-5× expected output | Change 15 |
| LLM call logging session filter | hardcoded `context_index ∈ (0,1,2)` in runner | `cfg.LLM_CALL_LOG_FIRST_N_SESSIONS` (default 10), strict `<` | Change 15 |
| `scope` BROAD/NARROW prior | none (LLM defaulted to BROAD ~100%) | one-line `If uncertain → NARROW` added to ②/③/④ user prompts | Change 16 |
| IRRELEVANT judgments | dropped at `_store_evidence_edge` (no edge) | stored as edges so `has_direct_edge` skip works in ⑤b/⑤c/⑤d | Change 16 |
| c-seed → SOURCE expansion | `{NODE_M, NODE_S}` only | `{NODE_M, NODE_S, NODE_T}` (traits are valid SOURCE-children of c) | Change 16 |
| Batched JUDGMENT_RETRY | bypassed (only sequential path retried) | mirrored in `_drain_pending_calls` with hint append + token accumulation | Change 16 |
| JUDGMENT_RETRY exhaustion fallback | no edges created | `apply_irrelevant_fallback(call)` writes IRRELEVANT edges for `call.expected_pairs` | Change 16 |
| Relation-extraction `reasoning` / `evidence_quote` fields (Change 4) | required in schema, prompt, and `add_evidence_edge` | removed from schema, prompts, `add_evidence_edge`, and `_evid_metadata` storage | Change 16 (rollback of Change 4) |

---

### Change 17: PersonaMem Realignment to ImplexConv Codebase

**Original Behavior (PersonaMem-specific divergence after Change 16)**

PersonaMem accumulated several extensions absent from the ImplexConv variant:
- (a) State `stability` enum (`TRANSIENT` / `SHORT_TERM` / `LONG_TERM`) plus
  per-state `_score_state` recency-decay multiplier;
- (b) Trait `effective_stability` computed from per-Node `in_sup`/`out_sup`/
  `in_con`/`out_con` counters, used in `_score_trait`;
- (c) Per-node-type seed scoring (`_score_context`, `_score_memory`,
  `_score_state`, `_score_trait`) with `W_REC` / `LAMBDA_BASE`;
- (d) τ-based trait validation producing three buckets
  (`traits_stable` / `traits_tentative` / `traits_challenged`);
- (e) Multi-hop signed-reachable BFS using cumulative-sign accumulation
  (`CON × SUP → CON`, `CON × CON → stop`);
- (f) `JUDGMENT_RETRY` triggered on `len(judgments) != expected` (partial
  arrays retried, not only empty);
- (g) Final state-set scoring used `DELTA · support_ratio` additive term;
  final memory scoring used `DELTA_M · support_ratio` additive term.

**New Behavior (aligned to ImplexConv `gmem5/`)**

- `stability` field removed from `Node`; `_score_state` no longer uses a
  recency-decay multiplier (no recency term in seed scoring at all).
- `in_sup` / `out_sup` / `in_con` / `out_con` counters removed from `Node`;
  `effective_stability` and `direct_ratio` methods removed from
  `HeterogeneousGraph`.
- Single unified `_seed_score(node, q_emb, q_kw_set)` replaces the four
  per-type scorers; `_weights_for(node)` picks `(W_SEM_C, W_OV_C)` for
  context, else scope-dependent `(W_SEM_NARROW, W_OV_NARROW)` /
  `(W_SEM_BROAD, W_OV_BROAD)`.
- Trait classification: top-`K_T_FINAL` traits selected by
  `W_SR · support_ratio + (1 - W_SR) · seed_score`; within that top-k a
  trait is `challenged` iff (CONTRADICT ratio ≤ τ) ∨ (outgoing SHIFT_TO
  into pooled trait). `traits_tentative` bucket removed.
- Final state and memory scoring share the same
  `W_SR · ratio + (1 - W_SR) · seed_score` formula. `DELTA` /
  `DELTA_M` removed.
- `get_ordinary_signed_reachable` rewritten as level-based BFS with
  SUP-priority (SUP × SUP → SUP continue; SUP × CON → CON record-but-stop;
  CON nodes never expand).
- `_graph_expansion` Step 1 SOURCE-derived expansion includes `NODE_T`
  (Change 16 already specified this; consolidated under a new
  `_expand_from` helper).
- `JUDGMENT_RETRY` triggers only when `len(judgments) == 0`; non-empty
  partial arrays are accepted as-is. The retry hint is updated to read
  `"Previous attempt returned empty judgments; ..."` on both the sequential
  and batched paths.
- ②b user prompt no longer carries a trailing `Final instruction:` block
  (it was retained in PersonaMem after Change 13).
- ② state-extraction user prompt moves `Assign placeholder ids:` to the
  end of the prompt (after `[PRIOR CONTEXT]`).
- ⑤d user prompt corrects `"is not SUPPORT"` → `"is NOT SUPPORT"`
  (capitalization, missed in Change 13's cross-call alignment).
- `_EVID_DESC_FULL` gains blank lines between bullet items (cosmetic
  parity with ImplexConv).
- `_NODE_TYPE_DESC` (in `generator.py`) and `_SYS_QA_MULTICHOICE` retain
  the PersonaMem-specific verbose definitions and four-option (a/b/c/d)
  schema — the multichoice QA prompt is *not* replaced by ImplexConv's
  opposed/supportive prompts.
- `CHUNK_FACTOR`, `TIME_PER_CONV_ID_HOURS` runtime override, and
  `INCLUDE_RECENT_CONVERSATION_FOR_QA = False` (PersonaMem-specific) are
  preserved.

**Affected Documents**
- `gmem5_bigflow.md`, `gmem5_config.md`, `gmem5_implementation.md`,
  `gmem5_prompt.md`, `gmem5_retrieval.md`, `gmem5_storage_extraction.md`
  — replaced with the ImplexConv versions and then re-patched for the
  PersonaMem multichoice QA, `INCLUDE_RECENT_CONVERSATION_FOR_QA = False`,
  and `CHUNK_FACTOR`.
- This entry is the only record of Change 17; older entries (Change 1 –
  Change 16) describe the pre-realignment trajectory and remain accurate
  for that history.

**Implementation Notes**
- `_resolve_id` was already accepting defensive string forms in PersonaMem
  prior to Change 17; no change needed there.
- Snapshots written before Change 17 contain `stability`,
  `in_sup`/`out_sup`/`in_con`/`out_con` keys per node; the new
  `load_snapshot` silently drops them (unknown keys are ignored by the
  current `Node(...)` constructor signature). New snapshots written after
  Change 17 do not include those keys.
- Older `traits_tentative` retrieval-log entries (`module_specific
  .num_by_slot.traits_tentative`) are not emitted in new runs; downstream
  analysis scripts should treat the key as optional.
