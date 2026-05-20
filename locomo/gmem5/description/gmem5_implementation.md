# GraphMem: Implementation Notes

> **Scope**: Runtime and implementation conventions only. Schema and extraction logic are defined in `gmem5_storage_extraction.md`; retrieval semantics are defined in `gmem5_retrieval.md`; prompt templates and node-listing formats are defined in `gmem5_prompt.md`; tunable parameters are defined in `gmem5_config.md`.

---

## 1. Execution Flow

```text
process_turn(user_utterance, gt_response, conv_id, turn_id, session_id)

  if conv_id changed and previous chunk exists:
      ③ Memory extraction (extraction-only)
      if (|chunk_state_ids| ≥ 1) or (previous_memory exists):
          ③b New-memory relation judgments (new↔chunk_states + new↔prev_memory)
      if 2-chunk boundary:
          ④ Trait extraction
          if new trait exists:
              ⑤a Local trait evidence judgment
              ⑤b Trait-centered extra relation extraction  [ENABLE_EXTRA_RELATION_EXTRACTION only]
          ⑤c Extra state-state relation extraction         [ENABLE_EXTRA_RELATION_EXTRACTION only; always, regardless of new trait]
          ⑤d Extra state-memory relation extraction        [ENABLE_EXTRA_RELATION_EXTRACTION only; always, regardless of new trait]

  create ContextNode
  retrieval → serialization
  ① response prompt construction (logged only in QA-only variant)
  update context cache

  if user_turn_count % STATE_EXTRACTION_H == 0:    # STATE_EXTRACTION_H = 1: every user turn
      ② State extraction (extraction-only, ≤ STATE_MAX_COUNT states)
      if (|new_state_ids| ≥ 2) or (|new_state_ids| ≥ 1 and |previous_state_ids| ≥ 1):
          ②b New-state relation judgments (new↔new + new↔prev)
```

`finalize_chunk()` must be called before Phase 2 so the last chunk runs ③ and, when applicable, ④-⑤d.

---

## 2. Serialization

The same retrieval object is used for:
- ① response prompt construction,
- ⑥ QA answering.

Sections:
```text
[Current Constraints]
[Traits]
[Challenged Traits]
[Relevant States]
[Relevant Memories]
[Recent Conversation]
```

Rules:
- `[Recent Conversation]` is included for response prompting;
- for QA it is included only if `INCLUDE_RECENT_CONVERSATION_FOR_QA = True`;
- default QA behavior is to omit it;
- `[Traits]` contains stable traits from the top-k trait selection;
- `[Challenged Traits]` contains traits with support ratio ≤ τ, with nested conflict evidence;
- traits with no evidence are classified as stable.

---

## 3. Common System Descriptions

### 3.1 Node Type Description

```text
State:  A user-specific condition that is currently or recently valid and may change over time.
Trait:  A generalized user characteristic that persists across situations and time.
Memory: An episodic summary of what happened during a recent conversation.
```

(LLM-facing prompts replace the system-internal term "chunk" with natural-language equivalents — see `gmem5_prompt.md` §2.1 / §6 / §8.)

### 3.2 Full Evidence Relationship Description

Use this when `SHIFT_TO` is allowed:

```text
SUPPORT:    The two pieces of information are consistent or mutually reinforcing.
CONTRADICT: The two pieces of information are in tension, but both may still hold.
SHIFT_TO:   An older state or trait has changed into a newer one.
            Use only for same-type temporal transitions and only in the old → new direction.
IRRELEVANT: The pair was judged and found unrelated.
```

### 3.3 Reduced Evidence Relationship Description

Use this when `SHIFT_TO` is not allowed:

```text
SUPPORT:    The two pieces of information are consistent or mutually reinforcing.
CONTRADICT: The two pieces of information are in tension or conflict.
IRRELEVANT: The pair was judged and found unrelated.
```

---

## 4. Prompt Templates

### 4.1 ② State Extraction (extraction-only)

Uses:
- node type description,
- (no evidence-relationship description — ② emits no judgments).

Prompt structure (see `gmem5_prompt.md` §4 for the full template):
- **State definition + good/NOT examples**: tells the LLM when to skip (no clear persona signal in the current turn → return empty state list).
- **`[CURRENT TURN]` block**: the single `(user_utterance, gt_response)` pair to extract from.
- **`[PRIOR CONTEXT]` block**: most recent `STATE_REF_CONTEXT_TURNS` `(user, assistant)` pairs as **read-only** reference for disambiguation, with an explicit "DO NOT extract states from these turns" label. Ignores `conv_id` boundaries; if fewer pairs are available, include only what exists (no padding).
- **keyword/domain_label disjointness block**: see §4.x note below. Output is post-processed via `deduplicate_labels`.
- No previous-state block.

`STATE_MAX_COUNT = 1` (`maxItems = 1` injected into schema). 0-state output is an expected, normal outcome.

② is **extraction-only**: its schema has no `judgments` field. `expected_judgment_count = 0`; the judgment-retry wrapper (§7) skips it. All new-state relation judgments (new↔new and new↔previous) are produced by ②b (§4.1b).

Prompt-side label rule for `current_decision_impact` (included when `STRICT_HIGH_DEFAULT_LOW = True`):
- Use `HIGH` only if the state should directly influence the assistant's very next response.
- Assign `HIGH` only when omitting the state would likely cause a materially worse, misleading, unsafe, or clearly non-personalized next answer.
- Do not assign `HIGH` to generic biography, weak preferences, broad background facts, or information that is merely relevant in a loose sense.
- If uncertain between `HIGH` and `LOW`, assign `LOW`.

### 4.1b ②b New-State Relation Judgments

Uses:
- node type description,
- **full** evidence relationship description.

Chained after ②. Input: newly created state nodes (`new_state_ids`) and the `STATE_NEW_REL_PREV_WINDOW` (= 3) most-recent existing state nodes (excluding newly extracted, by `created_at` desc) as the previous-state set. Judges:

- every unordered `new ↔ new` pair (when `|new_state_ids| ≥ 2`); and
- every `(new, previous)` pair (when `|previous_state_ids| ≥ 1`).

Trigger condition (apply-side dispatch):

```
trigger ⇔ |new_state_ids| ≥ 2  ∨  (|new_state_ids| ≥ 1 ∧ |previous_state_ids| ≥ 1)
```

Under `STATE_MAX_COUNT = 1`, the new↔new branch is dead and the trigger reduces to "new ≥ 1 ∧ prev ≥ 1"; the general form is kept so that raising `STATE_MAX_COUNT` later requires no code change.

`expected_judgment_count = C(|new_state_ids|, 2) + |new_state_ids| × |previous_state_ids|`. The judgment-retry wrapper (§7) applies: if `expected > 0` but the LLM returns an empty `judgments` array, retry up to `JUDGMENT_RETRY` times; on exhaustion, `apply_irrelevant_fallback` writes IRRELEVANT edges for every entry in `call.expected_pairs` and the call completes without hard failure.

`SHIFT_TO` direction is normalized on the apply side using `new_state_ids` membership:
- exactly one endpoint in `new_state_ids` → store `prev → new` (old → new);
- both endpoints in `new_state_ids` → fall back to `created_at` ordering;
- both in previous-batch (out of pair space; should not occur) → fall back to `created_at`.

### 4.2 ③ Memory Extraction (extraction-only)

Uses:
- node type description (the LLM-facing wording uses "recent conversation"; see `gmem5_prompt.md` §2.1),
- (no evidence-relationship description — ③ emits no judgments).

Prompt structure (see `gmem5_prompt.md` §6 for the full template):
- task description (concise episodic summary, "begins with The user", scope rules, label-discipline block),
- `[Recent Conversation]` data block at the bottom.

The chunk-states block has been **removed** (Change: ③ summarizes the
conversation only; chunk states are passed to ③b for relation judgment). No
previous-memory block.

③ is **extraction-only**: its schema has no `judgments` field. `expected_judgment_count = 0`; the judgment-retry wrapper (§7) skips it. All new-memory relation judgments live in ③b.

### 4.2b ③b New-Memory Relation Judgments

Uses:
- node type description,
- **reduced** evidence relationship description.

Chained after ③. Input: the newly created memory node, the previous memory node (if any), and the chunk's state nodes. Judges, in this fixed order:

1. the `(new_memory, previous_memory)` pair, when a previous memory exists;
2. one `(new_memory, chunk_state_i)` pair for each chunk state, in the listed extraction order.

Trigger condition (apply-side dispatch):

```
trigger ⇔ |chunk_state_ids| ≥ 1  ∨  previous_memory exists
```

In practice ③b fires on virtually every chunk boundary because chunks almost always contain ≥ 1 state.

`expected_judgment_count = (1 if previous_memory exists else 0) + |chunk_state_ids|`. The judgment-retry wrapper (§7) applies normally.

**Direction is fixed by the prompt**: `source = new_memory`; `target = previous_memory or chunk_state`. Canonical storage matches the prompt direction (`new_memory → previous_memory` for `m ↔ m`; `new_memory → chunk_state` for `m ↔ s`), so no flip is needed in the common path. The apply layer keeps a defensive flip as a safety net for non-conforming outputs.

The prompt also surfaces two domain-specific bias guards:
- `(new_memory, chunk_state_i)`: same-conversation co-extraction alone is not SUPPORT — the memory must add an episodic fact that would still ground the state if read in isolation;
- `(new_memory, previous_memory)`: temporal adjacency alone is not SUPPORT — only continues / confirms / concretely reinforces.

### 4.3 ④ Trait Extraction

No evidence judgment.

Prompt structure (see `gmem5_prompt.md` §8 for the full template):
- **Trait definition + "in general" test**: sharpens the trait-vs-state boundary. Concrete trait examples and state/trait comparison pairs have been **removed** to avoid canonical-bias mimicry.
- **Inferable-pattern criterion** (replacing the older "supported by repeated or converging evidence"): "be inferable from the recent conversations as a likely persistent pattern, not just a one-time event or momentary feeling." This avoids strict-evidence over-blocking at first ④ run, when no trait↔state/memory edges exist yet.
- **0-trait skip rule**: "If the recent conversations do not reveal a clear new persistent pattern, output 0 traits." (replaces the previous-trait paraphrase guard, which was vestigial after the input simplification below.)
- **Inputs**: only `[Recent Two Conversations]`. The chunk states block, chunk memories block, and most-recent-existing-trait block are **not** exposed to ④. The trait is inferred directly from raw conversation as a likely persistent pattern; relating it to existing nodes is the responsibility of ⑤a (which still receives all those nodes).
- Keyword/domain_label disjointness block.
- The system term "chunk" is not exposed to the LLM (header reads "Recent Two Conversations").

`TRAIT_MAX_COUNT = 1` (unchanged).

### 4.4 ⑤a Local Trait Evidence Judgment

Uses:
- node type description,
- **full** evidence relationship description.

Pair-specific rules inside the prompt:
- `state ↔ trait`: no `SHIFT_TO`
- `memory ↔ trait`: no `SHIFT_TO`
- `trait ↔ trait`: `SHIFT_TO` allowed only for `old_trait → new_trait`

`expected_judgment_count = |local_state_pool| + |local_memory_pool| + |existing_traits|` (only unconnected pairs). The user prompt substitutes this count as `{NUM_REQUIRED_PAIRS}` and instructs the LLM to output exactly that many judgments in the listed order. Judgment-retry policy applies (§7).

### 4.5 ⑤b Trait-Centered Extra Relation Extraction

Uses:
- node type description,
- **reduced** evidence relationship description.

Candidate construction:
1. retrieve provisional semantic top-k states and memories;
2. retrieve provisional lexical top-k states and memories;
3. merge and deduplicate;
4. rerank by `pair_score`;
5. cap to:
   - `TRAIT_EXTRA_REL_TOPK_STATE`
   - `TRAIT_EXTRA_REL_TOPK_MEMORY`

Only unconnected pairs are judged.

`expected_judgment_count = post-cap candidate count` (the number of pairs surviving the merge/dedup/rerank/cap pipeline). The user prompt substitutes this count as `{NUM_CANDIDATES}` and instructs the LLM to output exactly that many judgments. Judgment-retry policy applies (§7).

Output relations:
```text
SUPPORT | CONTRADICT | IRRELEVANT
```

### 4.6 ⑤c Additional State-State Relation Extraction

Uses:
- node type description,
- **full** evidence relationship description.

Selection policy:
- GraphMem maintains a global reservoir of high-scoring unconnected state-state pairs;
- whenever a new state is created, candidate scores against unconnected existing states are computed and the reservoir is updated;
- ⑤c consumes the current reservoir top-k.

This is **global top-k pair mining**, not anchor-local retrieval.

`expected_judgment_count = min(reservoir_size, STATE_STATE_EXTRA_REL_TOPK)`. The user prompt substitutes this count as `{NUM_PAIRS}` and instructs the LLM to output exactly that many judgments. Judgment-retry policy applies (§7).

Output relations:
```text
SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT
```

Constraint:
- `SHIFT_TO` is allowed only for true `old_state → new_state` transitions.

### 4.7 ⑤d Additional State-Memory Relation Extraction

Uses:
- node type description,
- **reduced** evidence relationship description.

Selection policy:
- GraphMem maintains a global reservoir of high-scoring unconnected state-memory pairs;
- whenever a new state or memory is created, candidate scores against compatible unconnected nodes are computed and the reservoir is updated;
- ⑤d consumes the current reservoir top-k.

This is also **global top-k pair mining**.

`expected_judgment_count = min(reservoir_size, STATE_MEMORY_EXTRA_REL_TOPK)`. The user prompt substitutes this count as `{NUM_PAIRS}` and instructs the LLM to output exactly that many judgments. Judgment-retry policy applies (§7).

Output relations:
```text
SUPPORT | CONTRADICT | IRRELEVANT
```

Stored canonically as:
```text
memory → state
```

### 4.x Cross-Cutting Notes for All Relation-Extraction Calls

These rules apply uniformly to ②b, ③b, ⑤a, ⑤b, ⑤c, ⑤d (every relation-extraction call). ② and ③ are extraction-only and exempt:

**Per-judgment output schema**:
```json
{
  "a": <integer index into id_map>,
  "b": <integer index into id_map>,
  "relation": "SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT"
}
```

Each call builds a local `id_map: List[str]` (index → node_id). The LLM outputs integer indices; the apply layer resolves them back to node IDs via `id_map`. ⑤d uses separate `id_map` (states) and `id_map_b` (memories). See `gmem5_storage_extraction.md` §3.7 and `gmem5_prompt.md` §1.9.

**Storage policy on judgments**:
- `SUPPORT | CONTRADICT | SHIFT_TO`: edge created;
- `IRRELEVANT`: no edge.

**Keyword/domain_label disjointness** (for ②, ③, ④ — the calls that produce node-level labels; ②/③ are extraction-only but still emit labels):
- prompt includes a definition block sharply distinguishing the two fields with a hard disjointness constraint;
- post-LLM, `deduplicate_labels(keywords, domain_label)` is applied on every node creation path: any `domain_label` token that also appears in `keywords` (case-insensitive) is removed; `keywords` win.

**Node-listing format in prompts** (applies to every call that lists existing nodes — ②b, ③b, ④, ⑤a, ⑤b, ⑤c, ⑤d):
- Format: `[id] [elapsed]: content` (no inline `(scope, current_decision_impact)` annotation).
- The `scope` and `current_decision_impact` fields remain in the node schema; they are simply not surfaced to the LLM.
- See `gmem5_prompt.md` §3.2 / §3.3.

### 4.8 ⑥ QA Answering

Uses the serialized retrieval object and does not mutate memory. Two distinct system prompts are used, keyed by `subset`:
- `subset = "opposed"`: `SYS_QA_OPPOSED` + `QA_PROMPT_OPPOSED` (free-text answer, ≤ 1-2 sentences, personalized when relevant);
- `subset = "supportive"`: `SYS_QA_SUPPORTIVE` + `QA_PROMPT_SUPPORTIVE` (yes/no enum). The supportive prompt is **memory-only** — if the retrieved memory is insufficient, irrelevant, or ambiguous, the model must answer `"no"`. There is no fallback to general reasoning.

Both prompts:
- describe `[Traits]` vs `[Challenged Traits]` semantics, including the `↳ shifted to:` and `↳ conflicting evidence:` sub-bullets that the retrieval serialization (§3.4) emits;
- describe `[Current Constraints]` as states to honor in the response **unless the current question explicitly overrides them**;
- include a personalization task description and (for opposed) a negative instruction prohibiting meta-commentary about the memory itself ("based on what I remember", "according to your past conversations").

Output schemas (both subsets emit `reasoning` first, then `answer`, to elicit chain-of-thought before the final reply):
- opposed: `{ "reasoning": "...", "answer": "..." }` (`answer` is free string)
- supportive: `{ "reasoning": "...", "answer": "yes" | "no" }`

`reasoning` is capped at one short sentence and is used only for offline analysis / debugging; the existing post-processor reads only the `answer` field, so evaluation pipelines that key on `answer` are unaffected.

---

## 5. Pair Similarity Utility

The same pair scoring utility is used for ⑤b reranking and ⑤c/⑤d reservoir maintenance:

```text
node_overlap(x, y) = |(keywords(x) ∪ domain_label(x)) ∩ (keywords(y) ∪ domain_label(y))|

pair_score(x, y) = w_pair_sem · sem(x, y)
                 + w_pair_lex · log(1 + node_overlap(x, y))
```

### 5.1 Global Reservoir Maintenance

For ⑤c and ⑤d:
- new candidate pairs are inserted into the relevant global reservoir;
- if the reservoir exceeds its size cap, the lowest-scoring pair is evicted.

Consume behavior at ⑤c/⑤d execution:
- the current top-k pairs are consumed from the reservoir and judged;
- after judgment, all consumed pairs are removed from the reservoir regardless of the judgment result (including `IRRELEVANT`);
- the reservoir then rebuilds incrementally as new nodes arrive before the next ⑤ trigger.

---

## 6. Token / Call Accounting

| # | Call | Trigger | Token category | Executed? |
|---|------|---------|----------------|-----------|
| ① | Response prompt construction | every turn | response input only | NO in QA-only variant |
| ② | State extraction (extraction-only, no judgments) | every user turn (`STATE_EXTRACTION_H = 1`) | internal | YES |
| ③ | Memory extraction (extraction-only, no judgments) | chunk boundary | internal | YES |
| ③b | New-memory relations (m↔chunk_states + m↔prev_memory) | chained after ③ when judgeable pair exists | internal | YES |
| ④ | Trait extraction | 2-chunk boundary | internal | YES |
| ⑤a | Local trait evidence | after ④, new trait only | internal | YES |
| ⑤b | Trait-centered extra relation extraction | after ⑤a, new trait only, ENABLE_EXTRA_RELATION_EXTRACTION | internal | YES |
| ⑤c | Extra state-state | every 2-chunk boundary, ENABLE_EXTRA_RELATION_EXTRACTION (regardless of new trait) | internal | YES |
| ⑤d | Extra state-memory | after ⑤c, ENABLE_EXTRA_RELATION_EXTRACTION | internal | YES |
| ⑥ | QA answering | per QA question | qa | YES |

---

## 7. Runtime Conventions

- `SHIFT_TO` is stored only in the `old → new` direction;
- no reverse `SHIFT_TO` edge is created;
- during expansion, `SHIFT_TO` is traversed forward only (old → new) for node discovery; reverse traversal is not performed;
- during support-ratio computation, `SHIFT_TO(A → B)` contributes **bidirectionally**: A receives `con_w += 1` (old, penalized), B receives `sup_w += 1` (new, boosted). The contribution requires both endpoints to be in the pool;
- CON terminates sign propagation immediately; there is no multi-hop CON propagation;
- if multiple paths exist between two nodes, shortest path takes priority; among equal-length paths, SUP takes priority over CON;
- `m ↔ s` is semantically symmetric but always stored canonically as `m → s`;
- `current_decision_impact = HIGH` is assigned conservatively and defaults to `LOW` under uncertainty;
- all node types serve as evidence for all other node types during support-ratio computation;
- if a node has no evidence, `support_ratio` defaults to 0.5;
- trait, state, and memory final scoring uses the same formula: `w_sr · support_ratio + (1 - w_sr) · seed_score`;
- shift-chain collapse is applied to `t_final`, `s_final`, and `s_aps`;
- ⑤a/⑤b run only when a new trait was extracted at the current 2-chunk boundary;
- ⑤c/⑤d run at every 2-chunk boundary regardless of whether a new trait was extracted;
- ⑤c/⑤d reservoir pairs are removed after consumption; the reservoir rebuilds incrementally before the next ⑤ trigger;
- when `ENABLE_EXTRA_RELATION_EXTRACTION = False`, ⑤b/⑤c/⑤d are all skipped and reservoir maintenance is also disabled;
- APS is constructed at retrieval time as the top-`k_aps` HIGH-impact, non-`SHIFT_TO`-source states ranked by `seed_score`. APS members are excluded from `s_seed`. HIGH states beyond rank `k_aps` fall through to `s_seed` and compete normally. APS construction does not look at `scope`; `scope` only affects seed-score weights;
- ② runs every user turn with `STATE_MAX_COUNT = 1` and is extraction-only (no `judgments` field in its schema). All new-state relation judgments (new↔new + new↔previous) are produced by ②b. The judgment-retry wrapper (below) treats ② as `expected_judgment_count = 0` and skips it;
- ②b is triggered when there is at least one judgeable pair: `|new_state_ids| ≥ 2 ∨ (|new_state_ids| ≥ 1 ∧ |prev_state_ids| ≥ 1)`, where `prev_state_ids` is the `STATE_NEW_REL_PREV_WINDOW` (= 3) most-recent existing states excluding the newly extracted ones;
- ③ is extraction-only (no `judgments` field). Its prompt contains only the recent-conversation block; chunk states are not exposed. All new-memory relation judgments live in ③b. The retry wrapper skips ③ as `expected_judgment_count = 0`;
- ③b is triggered when there is at least one judgeable pair: `|chunk_state_ids| ≥ 1 ∨ previous_memory exists`. It judges, in order: the `(new_memory, previous_memory)` pair (if a previous memory exists), then one `(new_memory, chunk_state_i)` pair per chunk state in listed order. Direction is fixed (`source = new_memory` in all pairs), so the apply layer's flip logic is a defensive safety net only;
- **Judgment retry policy** (`JUDGMENT_RETRY`): for every relation-extraction call, the wrapper computes `expected_judgment_count` AND populates `call.expected_pairs: List[Tuple[str, str]]` at build time (see `gmem5_storage_extraction.md` §6/§9 and §4.x of this document for per-call formulas and canonical directions). If `expected > 0` and the LLM returns an empty `judgments` array, retry up to `JUDGMENT_RETRY` times; on each retry attempt (2+), the hint `"Previous attempt returned empty judgments; you MUST output exactly N judgments."` is appended to the user prompt. **Both** the sequential path (`updater.py:_run_call`) and the batched path (`run_experiment.py:_drain_pending_calls`) implement this policy — the batched path was previously bypassing retry entirely. After exhaustion, `apply_irrelevant_fallback(call)` writes `IRRELEVANT` edges for every `(src, dst)` in `call.expected_pairs` (skipping missing nodes and pairs that already have a direct edge). Storing IRRELEVANT lets future ⑤b/⑤c/⑤d candidate selection skip these pairs via `has_direct_edge` instead of re-judging them. The call is **complete**, not a hard failure. `IRRELEVANT` is a valid judgment, not an empty case — a fully `IRRELEVANT` array is **not** retried. `JSON_RETRY` and `JUDGMENT_RETRY` stack: each judgment-retry attempt internally allows up to `JSON_RETRY` JSON-parse retries. Logging records each retry trigger with call name, `expected_count`, attempt number, and the raw output;
- **`keywords` / `domain_label` disjointness**: `deduplicate_labels(keywords, domain_label)` is applied on every node-creation path (state, memory, trait). Case-insensitive comparison; `keywords` win on overlap. The retrieval-side `label_set(n) = keywords(n) ∪ domain_label(n)` is unchanged.

---

## 7.1 Reference Helpers (sketches)

```python
def call_with_judgment_retry(prompt_fn, parser_fn, expected_count,
                             max_retry=JUDGMENT_RETRY, call_name=""):
    """
    Wrapper for any relation-extraction LLM call.
    Returns: parsed dict (possibly with empty 'judgments' after fallback).
    """
    for attempt in range(max_retry + 1):
        hint = (f"Previous attempt returned empty judgments; "
                f"you MUST output exactly {expected_count} judgments."
                if attempt > 0 else "")
        raw = prompt_fn(hint=hint)
        parsed = parser_fn(raw)               # JSON_RETRY already wraps this
        if expected_count == 0:
            return parsed                     # legitimate empty case (e.g., ②)
        if len(parsed.get("judgments", [])) > 0:
            return parsed                     # success (incl. all-IRRELEVANT)
        log_retry(call_name, expected_count, attempt, raw)
    return {"judgments": []}                  # caller treats expected pairs as IRRELEVANT


def deduplicate_labels(keywords, domain_label):
    """
    Enforce keyword/domain_label disjointness post-hoc; keywords win.
    Case-insensitive comparison, original case preserved in keywords.
    """
    kw_lower = {k.lower() for k in keywords}
    domain_label = [d for d in domain_label if d.lower() not in kw_lower]
    return keywords, domain_label


def apply_judgment(graph, j, id_map, id_map_b=None):
    """
    j has fields: a (int), b (int), relation.
    a indexes into id_map; b indexes into id_map_b if provided, else id_map.
    IRRELEVANT yields no edge.
    """
    if j.relation == "IRRELEVANT":
        return
    node_a = id_map[j.a]
    node_b = (id_map_b if id_map_b is not None else id_map)[j.b]
    graph.add_edge(node_a, node_b,
        relation=j.relation,
        # plus existing timestamp / metadata fields
    )


def construct_aps(states, query, k_aps):
    """
    Top-k_aps HIGH-impact, non-SHIFT_TO-source states ranked by seed_score.
    """
    candidates = [s for s in states
                  if s.current_decision_impact == "HIGH"
                  and not has_outgoing_shift_to(s)]
    candidates.sort(key=lambda s: seed_score(s, query), reverse=True)
    return candidates[:k_aps]


def seed_retrieve_states(all_states, query, k_s, aps_set):
    """
    Standard top-k_s state retrieval, excluding APS members.
    HIGH states beyond k_aps naturally fall through here.
    """
    aps_ids = {s.id for s in aps_set}
    pool = [s for s in all_states if s.id not in aps_ids]
    pool.sort(key=lambda s: seed_score(s, query), reverse=True)
    return pool[:k_s]
```

---

## 8. Interface Reminder

```python
result = module.process_turn(...)
module.finalize_chunk(...)
qa_result = module.get_qa_answer(question)
module.clear()
module.save_snapshot(directory)
module.load_snapshot(directory)
```

No generated response is written into memory. Ground-truth responses are always used for storage.
