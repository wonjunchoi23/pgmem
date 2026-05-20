# GraphMem v5 — Prompt Templates (PersonaMem variant)

> **Scope**: Centralizes every LLM prompt emitted by the PersonaMem
> implementation in `gmem5/`. Reflects what should appear in
> [generator.py](../generator.py), [updater.py](../updater.py), and
> [retriever.py](../retriever.py) after applying every change recorded in
> [`difference.md`](difference.md) on top of the gmem4 baseline.
>
> Calls covered:
> - ② State extraction (per-turn; extraction only, no judgments) → `call_2_state`
> - ②b New-state relations (new↔new + new↔prev judgments)        → `call_2b_state_new_rel`
> - ③ Memory extraction (extraction only, no judgments)          → `call_3_memory`
> - ③b New-memory relations (new_memory↔chunk_states + new_memory↔prev_memory) → `call_3b_memory_new_rel`
> - ④ Trait extraction                                            → `call_4_trait`
> - ⑤a Local trait evidence judgment                              → `call_5a_trait_evidence`
> - ⑤b Trait-centered extra relations                             → `call_5b_trait_extra_rel`
> - ⑤c Extra state-state relations                                → `call_5c_state_state_rel`
> - ⑤d Extra state-memory relations                               → `call_5d_state_memory_rel`
> - ⑥ QA answering (multichoice a/b/c/d)                          → `call_6_qa`
>
> ⚠ **Variant notes**:
> - There is **no ① Response prompt** call.
> - States and Traits **do not carry `stability` metadata** in this variant
>   (only `scope` and, for states, `current_decision_impact`).
> - State extraction is **per-turn** and emits at most one state per call
>   (`STATE_EXTRACTION_H = 1`, `STATE_MAX_COUNT = 1`). ② is extraction-only —
>   it does not emit a `judgments` array. All relation judgments involving
>   newly extracted states (both new↔new and new↔previous) are produced by
>   `call_2b_state_new_rel`.
> - Memory uses the same extraction-only / relations split. ③ is
>   extraction-only and emits no `judgments`. ③b
>   (`call_3b_memory_new_rel`) judges every (new_memory, chunk_state_i)
>   pair AND the (new_memory, previous_memory) pair, in that order.
> - Retrieval has no `Tentative Traits` section — only `Traits` (stable)
>   and `Challenged Traits`.
> - User prompts do **not** include a `[Recent Domain Labels]` block.
> - QA is **multichoice** (PersonaMem): four options labeled `a`/`b`/`c`/`d`.
>   The schema emits a short `reasoning` field before `answer` to elicit a
>   chain-of-thought; the post-processor uses only `answer`.
> - Every relation-extraction call's per-judgment output schema is minimal:
>   integer `source_id`/`target_id` (or `state_id`/`memory_id` in ⑤d) plus
>   `relation`. No `reasoning` or `evidence_quote` fields are produced
>   (see §1.9).
> - Every node-listing format drops `(scope, current_decision_impact)` —
>   nodes still carry these fields, but they are not surfaced to the LLM
>   (see §3.2 / §3.3).
> - The system-internal term "chunk" is replaced with natural-language
>   wording in user-facing strings (see §2.1, §6, §8, §11).
>
> Retrieval logic, storage schema, and execution flow are defined in the
> companion gmem5 design documents (`gmem5_retrieval.md`,
> `gmem5_storage_extraction.md`, `gmem5_bigflow.md`,
> `gmem5_implementation.md`, `gmem5_config.md`).

---

## Table of Contents

1. [Global Prompt Conventions](#1-global-prompt-conventions) — incl. §1.9 per-judgment auditability schema, §1.10 empty-judgment retry
2. [Reusable Prompt Components](#2-reusable-prompt-components) — incl. §2.3 keyword/domain_label disjointness block
3. [Formatting Conventions](#3-formatting-conventions)
4. [② State Extraction (per-turn, ≤ 1 state)](#4-②-state-extraction-per-turn--1-state)
5. [②b New-State Relations](#5-②b-new-state-relations)
6. [③ Memory Extraction + Memory↔State Evidence](#6-③-memory-extraction--memorystate-evidence)
7. [③b Memory ↔ Previous-Memory Relation](#7-③b-memory--previous-memory-relation)
8. [④ Trait Extraction](#8-④-trait-extraction)
9. [⑤a Local Trait Evidence Judgment](#9-⑤a-local-trait-evidence-judgment)
10. [⑤b Trait-Centered Additional Relation Extraction](#10-⑤b-trait-centered-additional-relation-extraction)
11. [⑤c Additional State-State Relation Extraction](#11-⑤c-additional-state-state-relation-extraction)
12. [⑤d Additional State-Memory Relation Extraction](#12-⑤d-additional-state-memory-relation-extraction)
13. [⑥ QA Answering (Multichoice)](#13-⑥-qa-answering-multichoice)
14. [Output Schemas Summary](#14-output-schemas-summary)

---

## 1. Global Prompt Conventions

### 1.1 JSON-only rule
All ②–⑥ calls return **strict JSON only** (enforced via `guided_json` schemas).
No prose, no markdown fences, no explanations outside JSON. The QA prompt
emits both `reasoning` and `answer` fields (in that order) inside a single
JSON object.

### 1.2 Evidence judgment discipline
When the model is uncertain it should prefer:
- `IRRELEVANT` over speculative linkage,
- `CONTRADICT` only when there is real tension,
- `SHIFT_TO` only for genuine temporal replacement within same-type pairs.

### 1.3 Node-writing discipline
- `State.content` — single concise sentence beginning with `The user`.
- `Memory.content` — 1–2 sentences beginning with `The user`.
- `Trait.content` — 2–3 complete sentences beginning with `The user`.

### 1.4 Label discipline
- `keywords` — up to `MAX_KEYWORDS` specific nouns or noun phrases.
- `domain_label` — `MIN_DOMAIN_LABELS` to `MAX_DOMAIN_LABELS` short topical
  labels, each 1–3 words.
- The user prompt always inlines these limits via the `STATE_MAX_COUNT`,
  `MAX_KEYWORDS`, `MIN_DOMAIN_LABELS`, `MAX_DOMAIN_LABELS` config values.
- There is **no** `[Recent Domain Labels]` reuse hint block in this variant.

### 1.5 Pair-family constraint
Judgment prompts must obey the allowed subtype space of the pair family:

| Pair family       | Allowed relations                                 |
|-------------------|---------------------------------------------------|
| `state ↔ state`   | `SUPPORT`, `CONTRADICT`, `SHIFT_TO`, `IRRELEVANT` |
| `trait ↔ trait`   | `SUPPORT`, `CONTRADICT`, `SHIFT_TO`, `IRRELEVANT` |
| `memory ↔ memory` | `SUPPORT`, `CONTRADICT`, `IRRELEVANT`             |
| `memory ↔ state`  | `SUPPORT`, `CONTRADICT`, `IRRELEVANT`             |
| `state ↔ trait`   | `SUPPORT`, `CONTRADICT`, `IRRELEVANT`             |
| `memory ↔ trait`  | `SUPPORT`, `CONTRADICT`, `IRRELEVANT`             |

### 1.6 `SHIFT_TO` rule
`SHIFT_TO` is allowed **only** for same-type temporal transition:

```text
old_state --SHIFT_TO--> new_state
old_trait --SHIFT_TO--> new_trait
```

It should be used only when:
- the old node is no longer currently valid,
- the new node is its temporal replacement,
- the pair is not merely coexisting tension.

### 1.7 Placeholder IDs
Extraction calls emit nodes with placeholder IDs that the updater rewrites to
real graph IDs after parsing:

| Call | Placeholder pattern |
|------|---------------------|
| ② state extract  | `new_0`, `new_1`, … (≤ `STATE_MAX_COUNT`) |
| ③ memory extract | `new_memory` (single)                      |
| ④ trait extract  | `new_trait` (0 or 1)                       |

The follow-up calls ②b and ③b operate on already-stored nodes and use the
**real graph IDs** shown in the listed blocks for `source_id`/`target_id`.

### 1.8 Metadata fields actually emitted

Distinct from the PersonaMem variant, the implex variant **omits `stability`**
from both states and traits:

| Node  | Fields emitted in this variant                   |
|-------|--------------------------------------------------|
| State | `scope`, `current_decision_impact` (no stability)|
| Trait | `scope` (no stability)                           |
| Memory| `scope`                                          |

### 1.9 Per-judgment output schema

Every relation-extraction call (②b, ③b, ⑤a, ⑤b, ⑤c, ⑤d) uses the following
minimal per-judgment schema:

```json
{
  "source_id": <integer>,
  "target_id": <integer>,
  "relation": "SUPPORT|CONTRADICT|SHIFT_TO|IRRELEVANT"
}
```

`source_id` and `target_id` are **integer indices** into the call's node-listing
order (see §3.2). The apply layer maps each index back to the real graph ID.

(In ⑤d the keys are `state_id`/`memory_id` instead of `source_id`/`target_id`,
each indexing into their respective separate node lists — see §12.)

`reasoning` and `evidence_quote` fields are **not** produced. No chain-of-thought
output is included in the judgment schema.

### 1.9b Direction convention

A reusable canonical-direction map for relation-extraction calls. Each
call's user prompt may repeat its specific direction rules verbatim; this
section is the authoritative reference and the apply-side storage
canonicalization matches it 1:1.

```text
Direction convention:
- For memory ↔ state judgments:
    source_id = memory id, target_id = state id.
- For state ↔ trait judgments:
    source_id = state id, target_id = trait id.
- For memory ↔ trait judgments:
    source_id = memory id, target_id = trait id.
- For new_memory ↔ previous_memory judgments:
    source_id = new_memory id, target_id = previous_memory id.
- For same-type SHIFT_TO judgments (state↔state, trait↔trait):
    source_id = older node id, target_id = newer node id.

```

⑤d uses keyed identifiers `state_id` and `memory_id` (not source_id /
target_id) with separate integer index spaces. Storage is canonical `m → s`.

### 1.10 Empty-judgment retry

Each relation-extraction call computes an `expected_judgment_count` AND
populates `expected_pairs: List[Tuple[str, str]]` — the canonical
`(src_node_id, dst_node_id)` tuples it is supposed to judge. If
`expected_judgment_count > 0` but the parsed `judgments` array is empty, the
wrapper retries up to `JUDGMENT_RETRY` times. On each retry attempt (attempt
2 onward), the following hint is appended to the end of the user prompt
before resubmission:

```text
Previous attempt returned empty judgments; you MUST output exactly {N} judgments.
```

where `{N}` is `expected_judgment_count`. Both execution paths apply this
policy:

- **sequential** (`updater.py:_run_call`) — used for one-off testing and the
  PersonaMem-style synchronous loop;
- **batched** (`run_experiment.py:_drain_pending_calls`) — used by the
  production runner. The batched path was previously bypassing
  `JUDGMENT_RETRY` entirely; it now mirrors the sequential policy:
  empty-judgment jobs are re-batched with the hint, retry token usage is
  accumulated on top of the original, and the same `JUDGMENT_RETRY` cap
  applies.

After exhaustion, the call's `expected_pairs` are stored as **`IRRELEVANT`
edges** in the graph (via `apply_irrelevant_fallback`). This is the change
from the earlier "no edges created" fallback: storing IRRELEVANT edges means
that subsequent ⑤b/⑤c/⑤d candidate selection (gated by `has_direct_edge`)
skips these pairs instead of re-judging them. The call is considered
**complete**, not failed.

`IRRELEVANT` is a valid judgment, not an empty case — a fully `IRRELEVANT`
array is **not** retried. See `gmem5_implementation.md` §7.

`JSON_RETRY` and `JUDGMENT_RETRY` stack: each judgment-retry attempt
internally allows up to `JSON_RETRY` JSON-parse retries.

② is exempt: it is extraction-only and does not produce `judgments`. All new-state
relation judgments are handled by ②b.

---

## 2. Reusable Prompt Components

### 2.1 Node-type description (`_NODE_TYPE_DESC`)

Used in every ②–⑤ system prompt and in both QA system prompts (⑥).

```text
Node types:
  State:  A user-specific condition that is currently or recently valid and may change over time. It captures the user's present stance, ongoing goal, constraint, situation, or preference shift. States are time-bounded and context-sensitive, and they may affect upcoming decisions or responses. Exclude transient emotions unless they directly modify an active task constraint or decision.
  Trait:  A generalized user characteristic that persists across situations and time. It represents recurring dispositions, stable preferences, values, or habitual tendencies. Traits are cross-situational and relatively context-independent. Evidential support is checked separately at later stages; focus here on whether the content itself is trait-like.
  Memory: An episodic summary of what happened during a recent conversation. It captures concrete events, topics, and actions at a particular time, not generalized persona attributes.
```

This block is the single source of truth: defined in `generator.py` and imported
by `updater.py` so that QA-side and extraction-side prompts share identical
node-type wording.

The system-internal term "chunk" does not appear in any LLM-facing prompt
(replaced with "conversation" / "recent conversation" / "recent two
conversations"). Spec documents may continue to use "chunk" internally.

### 2.2a Full evidence relation description (`_EVID_DESC_FULL`)

Used by: ②b, ⑤a, ⑤c (calls where `SHIFT_TO` is allowed). ② is extraction-only and does not consume evidence-relation descriptions.

```text
Evidence relationships:
  SUPPORT: The two pieces of information are consistent or mutually reinforcing. Use this only when one piece provides independent evidential force for the other.

  CONTRADICT: The two pieces of information are in clear tension, but they do not necessarily form a temporal replacement. Both may still be meaningful evidence.

  SHIFT_TO: A same-type temporal transition where the older information has changed into newer information and is no longer currently valid. Use only for true old → new replacement in state-state or trait-trait pairs. When emitting SHIFT_TO, the earlier node is the source old node and the later node is the target new node.

  IRRELEVANT: The pair was judged and found unrelated, OR the pair is merely topically related without one piece providing independent evidential force for or against the other.
```

The system prompt carries **only the four label definitions**. All
calibration rules (e.g. "topical similarity alone is not SUPPORT", uncertainty
priors, SHIFT_TO cautions) are placed in the **user prompt** of each call so
that the rules can be tailored to that call's pair family.

### 2.2b Reduced evidence relation description (`_EVID_DESC_REDUCED`)

Used by: ③b, ⑤b, ⑤d (calls where `SHIFT_TO` is not applicable). ③ is
extraction-only and does not consume evidence-relation descriptions.

```text
Evidence relationships:
  SUPPORT: The two pieces of information are consistent or mutually reinforcing. Use this only when one piece provides independent evidential force for the other.
  CONTRADICT: The two pieces of information are in clear tension or conflict. Do not use this for merely different topics, weak associations, or facts that can naturally coexist.
  IRRELEVANT: The pair was judged and found unrelated, OR the pair is merely topically related without one piece providing independent evidential force for or against the other.
```

### 2.3 keywords vs domain_label discipline (`_LABEL_DISCIPLINE_BLOCK`)

Used in every node-emitting prompt: ② state extraction, ③ memory extraction,
④ trait extraction. Inserted near the per-node metadata description.

```text
keywords:
  Surface-level tokens from the source text, or close lexical variants.
  Use concrete entities, named items, specific actions, constraints, or particular phrases.
  Do not use broad topical categories here.

domain_label:
  Abstract topical or categorical labels at a higher level of abstraction.
  Use broader subject areas or mid-level categories that could group related memories across different surface wording.

Hard constraint:
  Each label string MUST appear in at most one of keywords or domain_label.
  A domain_label may overlap lexically with a keyword only when it expresses a clearly broader topical category, not a simple restatement.
  Drop any domain_label that merely restates or narrowly rephrases a keyword.
```

Even with this prompt-side rule, the apply side runs `deduplicate_labels`:
any `domain_label` token that case-insensitively matches a `keyword` is removed
(`keywords` win). Trivial lexical variants (singular/plural, spacing,
hyphenation, casing) collapse to the same surface form before comparison.

---

## 3. Formatting Conventions

### 3.1 Conversation formatting (`_format_turns`)

```text
[{elapsed}] User: {user_utterance}
[{elapsed}] Assistant: {gt_response}
```

Elapsed time is rendered as a compact natural-language relative time:

```text
now, 5min ago, 3h ago, 2d ago, 4mo ago, 1y ago
```

Empty lists are rendered as:

```text
(none)
```

### 3.2 Node formatting in extraction prompts (`_format_*_list`)

Default form (no metadata):

```text
State:  [{idx}] [{elapsed}]: {content}
Memory: [{idx}] [{elapsed}]: {content}
Trait:  [{idx}] [{elapsed}]: {content}
```

`{idx}` is a **zero-based integer index** assigned at build time for each
relation-extraction call. Each call builds its own local `id_map: List[str]`
(index → real graph node ID); the LLM outputs integer `source_id`/`target_id`
values referencing this list. The apply layer maps indices back to real graph IDs.
The integer index space is local to each call and not shared across calls.

Empty lists are rendered as `(none)` (consistent with §3.1).

By default, node-listing prompts hide `(scope, current_decision_impact)`.
The fields are still produced and stored on the nodes; they are simply not
surfaced into prompts that consume node listings.

**Exception**: state-state relation prompts that can emit `SHIFT_TO` —
②b and ⑤c — expose `scope` (but **not** `current_decision_impact`) so the
LLM can reason about whether a narrow topical preference is replacing a
broader lifestyle constraint. The state-listing variant for those calls
renders as:

```text
State:  [{idx}] [{elapsed}] (scope={scope}): {content}
```

This affects every prompt that lists nodes via `_format_*_list`: ②b, ③, ④,
⑤a, ⑤b, ⑤c, ⑤d. Of those, only ②b and ⑤c use the `scope`-exposed form
(and only for state nodes); all others use the default form.

### 3.3 Pair-list formatting for ⑤c and ⑤d

#### ⑤c state ↔ state

Pairs are rendered in **chronological order** (older first, newer second) so the
LLM has the SHIFT_TO direction encoded in the layout. `scope` is exposed
because broad lifestyle constraints should not be replaced by narrow topical
preferences; hiding `scope` invites false SHIFT_TO judgments.

```text
Pair {pair_idx}
- older_state: [{old_node_idx}] [{old_elapsed}] (scope={old_scope}): {old_content}
- newer_state: [{new_node_idx}] [{new_elapsed}] (scope={new_scope}): {new_content}
```

`{old_node_idx}` and `{new_node_idx}` are integer indices into the call's `id_map`.

Append to the ⑤c user prompt:

```text
The pair is shown in chronological order: older_state first, newer_state second.
Be conservative with SHIFT_TO. Do not assume that a new topical preference replaces a broader lifestyle constraint, value, or long-running condition. SHIFT_TO is only for true content-level replacement of the same kind of attribute.
```

#### ⑤d state ↔ memory

⑤d has no SHIFT_TO so `scope` exposure is not required. To reduce
same-conversation bias, append a "Relation context" hint with the
conversation gap and whether the two come from the same conversation:

```text
Pair {pair_idx}
- state:  [{s_idx}] [{elapsed_s}]: {content_s}
- memory: [{m_idx}] [{elapsed_m}]: {content_m}
  Relation context: {relation_context_str}
```

Where `{s_idx}` indexes into the call's state `id_map` and `{m_idx}` indexes
into the call's memory `id_map_b` (separate zero-based index spaces).

`{relation_context_str}` is:
- `"same conversation."` — when `s.conv_id == m.conv_id` (no "apart" suffix);
- `"different conversations, {gap} apart."` — otherwise, where `{gap}` is the
  compact-elapsed gap (§3.1).

Schema and storage are unaffected. Only the prompt-side rendering is changed.

### 3.4 Retrieval serialization for ⑥ QA

The retriever produces a single string with the following fixed section order
(see `GraphRetriever._serialize`). Each section header is rendered as
`[Title]` on its own line; some headers carry an inline parenthetical note.
Empty sections still emit the header with no body lines.

```text
[Current Constraints]
(These are high-impact user states the assistant should honor in the response, unless the user explicitly overrides them in the current message.)
[{elapsed}] {state.content}
...

[Traits]
[{elapsed}] {trait.content}
...

[Challenged Traits]
[{elapsed}] {trait.content}
  ↳ shifted to: [{elapsed}] {newer_trait.content}
  ↳ conflicting evidence: [{elapsed}] {state_or_memory_or_trait.content}
...

[Relevant States]
(Additional retrieved user states pertaining to the current question; not necessarily high-impact.)
[{elapsed}] {state.content}
...

[Relevant Memories]
(Episodic summaries of past conversations relevant to the current question.)
[{elapsed}] {memory.content}
...

[Recent Conversation]
{context_cache_str.splitlines()}
```

**Empty sections are omitted entirely** (the header is not emitted when there
are no listed lines). The `[Recent Conversation]` section is included for QA
when `INCLUDE_RECENT_CONVERSATION_FOR_QA` is `True` (default in current
config). For QA the cache is sliced to its **last `QA_CONTEXT_PAIRS` pairs**
(default 5 = 10 lines) so that the prompt carries only recent context, while
non-QA retrieval continues to see the full cache.

Notes:
- There is **no `Tentative Traits` section** in this variant. Stable traits go
  under `[Traits]`; traits with `SHIFT_TO`-out edges or active `CONTRADICT`
  evidence go under `[Challenged Traits]`.
- `Current Constraints` softens the directive to "honor unless overridden" —
  HIGH-impact states are not absolute rules; the user may explicitly override
  them in the current message.
- `Challenged Traits` body distinguishes two sub-bullets:
  - `↳ shifted to:` — newer trait that the challenged trait was replaced by
    (sourced from `SHIFT_TO`-out edges into the pool);
  - `↳ conflicting evidence:` — state, memory, or trait nodes contributing
    `CONTRADICT` signal toward the challenged trait.
- `[Relevant States]` and `[Relevant Memories]` carry a one-line header note
  to disambiguate them from `[Current Constraints]` (states) and `[Traits]`
  (persona summary).

#### Final serialization example

```text
[Current Constraints]
(These are high-impact user states the assistant should honor in the response, unless the user explicitly overrides them in the current message.)
[2d ago] The user recently injured their leg and should avoid strenuous physical activity.

[Traits]
[6mo ago] The user generally enjoys playing soccer and outdoor sports.

[Challenged Traits]
[1y ago] The user usually eats meat-heavy meals.
  ↳ shifted to: [2mo ago] The user recently became vegetarian.
  ↳ conflicting evidence: [1mo ago] The user avoided meat when choosing restaurants.

[Relevant States]
(Additional retrieved user states pertaining to the current question; not necessarily high-impact.)
[2mo ago] The user recently became vegetarian.

[Relevant Memories]
(Episodic summaries of past conversations relevant to the current question.)
[1mo ago] The user discussed choosing vegetarian-friendly restaurants with friends.
```

---

## 4. ② State Extraction (per-turn, ≤ 1 state)

> Extract at most one state node from the **current user turn**. Up to
> `STATE_REF_CONTEXT_TURNS` previous `(user, assistant)` pairs are shown as
> read-only reference. ② is extraction-only — it does not emit any
> relation judgments. All new-state relation judgments (new↔new when the
> call ever emits ≥2 states, plus new↔previous) are produced by the
> follow-up call ②b (`call_2b_state_new_rel`).
>
> Source: `GraphUpdater._build_state_call` ([updater.py](../updater.py)).

### System (`SYS_STATE_EXTRACT`)

```text
You are a persona state extraction assistant.
Extract user states from the current user turn.
Do not classify relationships in this call.
Respond in strict JSON.

{node_type_desc}
```

② is extraction-only and does **not** consume `_EVID_DESC_FULL`. All new-state
relation judgments live in ②b.

### User

```text
A state does not have to be permanently stable. If the current turn reveals
a useful persona-relevant signal but there is not yet enough evidence to call
it a long-term trait, extract it as a state. Later stages may generalize
repeated or stable states into traits.

Do not extract if the information is only:
  - a greeting, thanks, or conversational behavior,
  - a description of the current query rather than the user,
  - assistant-side information,
  - a transient emotion with no effect on an active task, decision, constraint, or safety issue.

Task:
Extract up to {STATE_MAX_COUNT} persona state(s) revealed in the CURRENT TURN.
If the current turn contains no new persona-relevant signal, return an empty states list.
Do not invent or speculate.

Use the assistant response only as read-only context for resolving references in the user's utterance.
Do NOT extract a state if the condition is stated only by the assistant and not expressed or clearly implied by the user.

Each state must:
- begin with "The user",
- be a single concise sentence,
- avoid raw episodic narration unless it directly functions as a current condition or useful persona signal.

State metadata:
scope:
  BROAD  : a currently valid persona signal, condition, value-driven stance, health-related limitation, lifestyle constraint, role, or preference that may affect decisions across unrelated topics.
  NARROW : a currently valid preference, goal, condition, or constraint tied mainly to the current task, topic, or short-term situation.

If uncertain between BROAD and NARROW, choose NARROW.

current_decision_impact:
  HIGH : the assistant must actively remember this right now.
         Use HIGH only when BOTH are true:
         (1) the user would reasonably expect this to be remembered without re-stating it,
         (2) ignoring it would cause a response that is clearly wrong, unsafe,
             or noticeably frustrating.
         Prefer HIGH for persistent constraints, hard restrictions, safety-relevant
         conditions, urgent deadlines, or explicit standing expectations.
         Be conservative with short-lived turn-specific preferences.
  LOW  : useful persona context, but the response would still be appropriate
         and acceptable without it.

If uncertain between HIGH and LOW, choose LOW.

{label_discipline_block}    # see §2.3

Also provide for each state:
- keywords: up to {MAX_KEYWORDS} specific nouns or noun phrases
- domain_label: {MIN_DOMAIN_LABELS} to {MAX_DOMAIN_LABELS} short topical labels

[CURRENT TURN — extract state(s) from this turn only]
{current_turn_block}

[PRIOR CONTEXT — for disambiguation only.
 These turns may be from earlier today or earlier sessions.
 Use them only to clarify what the CURRENT TURN refers to.
 DO NOT extract states from these turns.]
{prior_context_block}

Assign placeholder ids: new_0, new_1, ... in order of extraction.
```

Notes:
- `{current_turn_block}` follows §3.1 with the single current `(user, gt_response)` pair.
- `{prior_context_block}` follows §3.1 with up to `STATE_REF_CONTEXT_TURNS`
  previous `(user, assistant)` pairs. Prior context **ignores `conv_id`
  boundaries**: the most recent N pairs are shown regardless of whether they
  fall under a different `conv_id` (i.e., even if a 12-hour gap precedes the
  current turn). If fewer than `STATE_REF_CONTEXT_TURNS` prior pairs are
  available (e.g., session start), include only what exists.
- There is **no `[Previously Extracted States]` block** and **no
  `[Recent Domain Labels]` block** in this user prompt; previous-state
  relations are deferred to ②b.

### Output schema (`STATE_EXTRACT_SCHEMA`, capped at `STATE_MAX_COUNT`)

```json
{
  "states": [
    {
      "id": "new_0",
      "content": "The user ...",
      "keywords": ["..."],
      "domain_label": ["..."],
      "scope": "BROAD|NARROW",
      "current_decision_impact": "HIGH|LOW"
    }
  ]
}
```

`states.maxItems` is set dynamically to `STATE_MAX_COUNT` (= 1). The schema
has **no `judgments` field** — ② is extraction-only and the judgment-retry
wrapper skips it (see §1.10). All new-state relation judgments are produced
by ②b (§5).

---

## 5. ②b New-State Relations

> Run after ② whenever there is at least one judgeable pair involving the
> newly extracted state(s). Judges:
>
>   - every **new ↔ new** pair (when the same ② call produced ≥ 2 new states), and
>   - every **new ↔ previous** pair (when previous-state IDs exist),
>
> using real graph IDs. ② is extraction-only — all new-state relation
> judgments live here.
>
> Trigger condition (apply side):
>
>   `len(new_state_ids) ≥ 2`  OR  (`len(new_state_ids) ≥ 1` AND `len(previous_state_ids) ≥ 1`)
>
> Under the current `STATE_MAX_COUNT = 1`, the new↔new branch is dead and the
> trigger reduces to "new ≥ 1 AND previous ≥ 1". The general form is kept so
> that raising `STATE_MAX_COUNT` later does not require any code change here.
>
> Source: `GraphUpdater._build_state_new_rel_call` ([updater.py](../updater.py)).

### System (`SYS_STATE_NEW_REL`)

```text
You are an evidence classification assistant.
Classify direct relationships involving newly extracted states: pairs among the new states themselves, and pairs between each new state and each previously extracted state.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships involving newly extracted states.

Each node is identified by an integer index shown in brackets: [index].
Indices 0..{n_prev_valid - 1} are previous states. Indices {n_prev_valid}..{n_prev_valid + n_new_valid - 1} are new states.

Judge:
- every previous↔new pair; and
- every unordered pair of new states, only when more than one new state was extracted.
  Judge each new↔new pair exactly once; do not output both (A,B) and (B,A).

Direction rules:
- For previous↔new pairs: previous state = source_id, new state = target_id.
- For new↔new SHIFT_TO: source_id = older state index (as described in content), target_id = newer replacement state index.
- For new↔new SUPPORT, CONTRADICT, IRRELEVANT: earlier-listed new state = source_id, later-listed new state = target_id.

new↔new pair caution:
- The two states were extracted from the SAME current turn. Co-extracted states usually coexist.
- Use SHIFT_TO for a new↔new pair only if the CURRENT TURN explicitly states a temporal replacement between them.
- Do not infer SHIFT_TO merely because the two states differ, contrast, or appear emotionally opposed.

Calibration:
- Topical similarity alone is NOT SUPPORT.
- Temporal proximity alone does not create evidential force.
- A short-term or narrow preference does not replace a broader condition, value, role, lifestyle constraint, or safety condition unless one state explicitly invalidates the other.
- Do not use SHIFT_TO when the newer state only adds detail without invalidating the older state.
- Do not use SHIFT_TO for a topic change without explicit replacement.
- When uncertain between SHIFT_TO and CONTRADICT, choose CONTRADICT.
- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.

[Previous States — extracted before the current turn]
{previous_states_block_with_scope}

[New States — extracted from the current turn]
{new_states_block_with_scope}
```

`{new_states_block_with_scope}` and `{previous_states_block_with_scope}` use
the §3.2 **scope-exposed** listing variant (`(scope=BROAD|NARROW)`); see §3.2.
Both blocks use integer indices from a single combined `id_map` (prev states first,
new states after). If `{previous_states_block_with_scope}` is empty (session
start) it renders as `(none)` and only new↔new pairs are judged.

`expected_judgment_count = C(|new_states|, 2) + |new_states| × |previous_state_ids|`.
Judgment-retry policy applies (§1.10).

### Output schema

`JUDGMENTS_SCHEMA` with full relation enum, integer IDs:

```json
{
  "judgments": [
    {
      "source_id": 0,
      "target_id": 2,
      "relation": "SUPPORT|CONTRADICT|SHIFT_TO|IRRELEVANT"
    }
  ]
}
```

### Apply-side notes

- For `SHIFT_TO`, direction is normalized using `new_state_ids` membership
  (older node → newer node):
  - if exactly one endpoint is in `new_state_ids` → the previous-batch endpoint
    is the older one; store `prev → new`.
  - if both endpoints are new (only possible when `STATE_MAX_COUNT ≥ 2`) → fall
    back to `created_at` ordering.
  - if both are previous-batch (should not happen under ②b's allowed pair
    space) → fall back to `created_at`.
- `IRRELEVANT` judgments produce no edge (existing storage policy).

---

## 6. ③ Memory Extraction (extraction-only)

> Summarize the chunk into one episodic memory. ③ does **not** classify
> any relations; all new-memory relation judgments live in ③b.
>
> Source: `GraphUpdater._build_memory_call` ([updater.py](../updater.py)).

### System (`SYS_MEMORY_EXTRACT`)

```text
You are an episodic memory extraction assistant.
Summarize the recent conversation into one episodic memory.
Do not classify relationships in this call.
Respond in strict JSON.

{node_type_desc}
```

③ is extraction-only and does **not** consume `_EVID_DESC_REDUCED`. All
new-memory relation judgments (memory↔chunk_states + memory↔previous_memory)
live in ③b.

### User

```text
Create exactly one episodic memory node summarizing what happened in the recent conversation.
Each memory must:
- begin with "The user",
- be 1–2 sentences,
- summarize concrete events, discussed topics, actions, and developments,
- describe the episode itself, not generalized persona traits.

Do NOT include chunk-level state extractions or restate persona traits unless
they are part of the episode itself. The state nodes are extracted by a
separate pipeline; this call summarizes only the conversation.

Memory metadata:
scope:
  BROAD  : the episode reveals or confirms a cross-topic user characteristic (e.g., a health event, a major life decision, a value-revealing exchange, or a standing constraint the user reaffirmed).
  NARROW : the episode is self-contained within the current topic or task — its implications do not extend beyond the current conversation thread.

If uncertain between BROAD and NARROW, choose NARROW.

{label_discipline_block}    # see §2.3

Also provide:
- keywords: up to {MAX_KEYWORDS} specific nouns or noun phrases
- domain_label: {MIN_DOMAIN_LABELS} to {MAX_DOMAIN_LABELS} short topical labels

Assign the memory the placeholder id: new_memory.

[Recent Conversation]
{chunk_conversation_block}
```

The user prompt contains **only** the conversation block (Change: chunk
state listing has been removed; ③ is now purely an episodic-summary task).
There is no `[Previous Memory]` block, no `[Chunk States]` block, no
`[Recent Domain Labels]` block. The header `[Recent Conversation]` collides
with the QA-side serialization section name (§3.4); the two never appear in
the same prompt.

### Output schema (`MEMORY_EXTRACT_SCHEMA`)

```json
{
  "memory": {
    "id": "new_memory",
    "content": "The user ...",
    "keywords": ["..."],
    "domain_label": ["..."],
    "scope": "BROAD|NARROW"
  }
}
```

The schema has **no `judgments` field** — ③ is extraction-only and the
judgment-retry wrapper skips it (`expected_judgment_count = 0`). All
new-memory relation judgments are produced by ③b (§7).

---

## 7. ③b New-Memory Relations

> Run after ③ whenever there is at least one judgeable pair involving the
> newly extracted memory. Judges, in order:
>
>   - one (new_memory, previous_memory) pair, if a previous memory exists; and
>   - one (new_memory, chunk_state_i) pair for each chunk state, in the listed order.
>
> ③ is extraction-only — all new-memory relation judgments live here.
>
> Trigger condition (apply side):
>
>   `|chunk_state_ids| ≥ 1`  OR  previous_memory exists
>
> In practice ③b fires on virtually every chunk boundary because chunks
> almost always contain ≥ 1 state.
>
> Source: `GraphUpdater._build_memory_new_rel_call` ([updater.py](../updater.py)).

### System (`SYS_MEMORY_NEW_REL`)

```text
You are an evidence classification assistant.
Classify direct relationships involving a newly extracted memory: the pair (new_memory, previous_memory) when a previous memory exists, and one pair (new_memory, chunk_state_i) for each chunk state.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_reduced}
```

### User

```text
You will judge direct evidence relationships involving the newly extracted memory.

Each node is identified by an integer index shown in brackets: [index].
Index 0 is the new memory. Index {prev_idx} is the previous memory. Indices {state_start}..{state_start + n_states - 1} are chunk states.

Judge:
- the (new_memory, previous_memory) pair, when a previous memory exists; and
- one (new_memory, chunk_state_i) pair for each listed chunk state, in the listed order.

Direction rules (FIXED):
- source_id MUST be 0 (new memory index) for every judgment.
- target_id is the previous_memory index or chunk_state index.

Relation rules:

For (new_memory, previous_memory):
  Use SUPPORT | CONTRADICT | IRRELEVANT.
  Temporal adjacency alone is NOT SUPPORT.
  Use SUPPORT only when the new memory continues, confirms, or concretely reinforces the previous memory.

For (new_memory, chunk_state_i):
  Use SUPPORT | CONTRADICT | IRRELEVANT.
  Use SUPPORT only when the memory provides concrete episodic evidence that independently grounds or confirms the state.
  Same-conversation co-extraction alone is NOT sufficient evidential force.
  The memory must add an episodic fact that would still ground the state if read in isolation.

When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.

Output exactly {N} judgments in this order:
  1) the (new_memory, previous_memory) judgment, if a previous memory exists;
  2) one (new_memory, chunk_state_i) judgment for each listed chunk state, in the listed order.
Do not invent states, memories, or judgments.

[New Memory]
{new_memory_block}

[Previous Memory]
{previous_memory_block_or_none}

[Chunk States — listed in extraction order]
{chunk_states_block_or_none}

Final instruction:
Output exactly {N} judgments in the order specified above (previous_memory first if present, then chunk states in listed order).
Use the integer indices shown above for source_id and target_id.
Return strict JSON only.
```

`{new_memory_block}`, `{previous_memory_block_or_none}`, and `{chunk_states_block_or_none}`
use the §3.2 default listing format with integer indices from a single combined
`id_map` (new_memory first, then prev_memory if exists, then chunk states in order).
Either of the latter two blocks may render as `(none)`.

`{N}` is `(1 if previous_memory exists else 0) + |chunk_state_ids|`.

`expected_judgment_count = (1 if previous_memory exists else 0) + |chunk_state_ids|`.
Judgment-retry policy applies (§1.10).

### Output schema

`JUDGMENTS_SCHEMA` with reduced relation enum, integer IDs:

```json
{
  "judgments": [
    {
      "source_id": 0,
      "target_id": 1,
      "relation": "SUPPORT|CONTRADICT|IRRELEVANT"
    }
  ]
}
```

### Apply-side notes

- Direction is fixed by the prompt: `source = new_memory`, `target = prev_memory or chunk_state`. Storage canonicalization (`m → s`, `new_memory → previous_memory`) matches the prompt direction, so no flip is needed in the common path. The apply layer keeps a defensive flip as a safety net for non-conforming outputs.
- `IRRELEVANT` judgments produce no edge (existing storage policy).

---

## 8. ④ Trait Extraction

> Extract at most one new trait from the recent two-chunk window.
>
> Source: `GraphUpdater._build_trait_call` ([updater.py:860](../updater.py#L860)).

### System (`SYS_TRAIT_EXTRACT`)

```text
You are a persona trait extraction assistant.
Infer at most one new long-term trait from the accumulated recent evidence.
Respond in strict JSON.

{node_type_desc}
```

### User

```text
Use the "in general" test: a trait should still be true if you asked the user about themselves "in general" with no specific time, place, or context attached.

Extract 0 or 1 trait from the recent two conversations below.
A trait should:
- begin with "The user",
- be 2–3 complete sentences,
- be inferable from the recent conversations as a likely persistent pattern, not just a one-time event or momentary feeling.

If the recent conversations do not reveal a clear new persistent pattern, output 0 traits.

Trait metadata:
scope:
  BROAD  : the trait applies across all domains of the user's life — it would shape the user's approach regardless of the subject being discussed
  NARROW : a domain-specific tendency or preference mainly expressed within a particular area (e.g., a consistent style in one hobby or activity)

If uncertain between BROAD and NARROW, choose NARROW.

{label_discipline_block}    # see §2.3

Also provide:
- keywords: up to {MAX_KEYWORDS} specific nouns or noun phrases
- domain_label: {MIN_DOMAIN_LABELS} to {MAX_DOMAIN_LABELS} short topical labels

Assign the trait the placeholder id: new_trait.

[Recent Two Conversations]
{two_chunk_conversation_block}
```

The user prompt contains **only** the recent-two-conversations block.
Chunk states, chunk memories, and the previously extracted trait are **not**
exposed to ④ — the trait is inferred directly from the raw conversation as
a likely persistent pattern. The system-internal term "chunk" is not exposed —
headers use "recent two conversations" wording.

### Output schema (`TRAIT_EXTRACT_SCHEMA`, capped at `TRAIT_MAX_COUNT`)

```json
{
  "traits": [
    {
      "id": "new_trait",
      "content": "The user ...",
      "keywords": ["..."],
      "domain_label": ["..."],
      "scope": "BROAD|NARROW"
    }
  ]
}
```

`traits.maxItems` is set dynamically to `TRAIT_MAX_COUNT`. **No `stability`
property** in the schema.

---

## 9. ⑤a Local Trait Evidence Judgment

> Judge the new trait against local recent evidence and the previous trait.
>
> Source: `GraphUpdater._build_5a_call` ([updater.py:920](../updater.py#L920)).

### System (`SYS_TRAIT_EVIDENCE_5A`)

```text
You are an evidence classification assistant.
Classify the direct relationship between a newly extracted trait and nearby existing nodes.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships between a newly extracted trait and nearby existing nodes.

Judge:
- each (recent state, new_trait) pair, in the listed order;
- each (recent memory, new_trait) pair, in the listed order;
- the (previous_trait, new_trait) pair, when a previous trait exists.

Direction rule:
- For (recent_state, new_trait) and (recent_memory, new_trait), use the existing-node id as source_id and the new_trait id as target_id.
- For (previous_trait, new_trait) SHIFT_TO, source_id MUST be the previous (old) trait and target_id MUST be the new trait.

Relation rules:
- For state↔trait and memory↔trait: use only SUPPORT | CONTRADICT | IRRELEVANT.
- For previous_trait↔new_trait: use SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT. SHIFT_TO is allowed only for a true old_trait → new_trait replacement.

SUPPORT calibration:
- Topical similarity alone is NOT SUPPORT.
- For (state, new_trait): use SUPPORT only when the state describes a current/recent condition that the new trait would naturally generalize from or predict. Sharing a domain or keyword is not enough.
- For (memory, new_trait): use SUPPORT only when the memory captures concrete user behavior that the trait would predict; the memory must add evidential force beyond mere topic overlap.
- For (previous_trait, new_trait): use SUPPORT only when the two traits independently describe overlapping or compatible persistent characteristics.
- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.

CONTRADICT calibration:
- Use CONTRADICT only when the candidate clearly conflicts with the new trait at the persona level.
- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the trait.
- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.

SHIFT_TO caution (only for previous_trait↔new_trait):
- Use SHIFT_TO only when the previous trait is no longer applicable because the new trait replaces it at the same dimension.
- Do not use SHIFT_TO when the new trait only adds a related disposition without invalidating the previous one.
- When uncertain between SHIFT_TO and CONTRADICT, choose CONTRADICT.

Output requirements:
Output exactly {NUM_REQUIRED_PAIRS} judgments — one for each listed pair, in the listed order.
Do not skip or duplicate pairs.

[New Trait]
{new_trait_block}

[Recent States]
{recent_states_block}

[Recent Memories]
{recent_memories_block}

[Previous Trait]
{previous_trait_block_or_none}

Final instruction:
Output exactly {NUM_REQUIRED_PAIRS} judgments — one for each listed pair, in the listed order.
Use the integer indices shown in brackets above for source_id and target_id.
Return strict JSON only.
```

`{NUM_REQUIRED_PAIRS}` is `|recent_states| + |recent_memories| + (1 if previous_trait exists else 0)`.

All blocks use integer indices from a single combined `id_map` (states first, then
memories, then prev_trait if exists, then new_trait last). Blocks use the §3.2
default listing format (no scope/impact). The `previous_trait` block may render
as `(none)`.

`expected_judgment_count = |recent_states_pool| + |recent_memories_pool| + |existing_traits|`
(only unconnected pairs are listed). Judgment-retry policy applies (§1.10).

### Output schema (`JUDGMENTS_SCHEMA`)

```json
{
  "judgments": [
    {
      "source_id": 0,
      "target_id": 3,
      "relation": "SUPPORT|CONTRADICT|SHIFT_TO|IRRELEVANT"
    }
  ]
}
```

---

## 10. ⑤b Trait-Centered Additional Relation Extraction

> Judge extra unconnected state/trait and memory/trait candidates retrieved
> from outside the local two-chunk window.
>
> Source: `GraphUpdater._build_5b_call` ([updater.py:970](../updater.py#L970)).
> Skipped (returns `None`) when both candidate lists are empty.

### System (`SYS_TRAIT_EXTRA_REL_5B`)

```text
You are an evidence classification assistant.
Classify direct relationships between a new trait and additional candidate nodes retrieved from the graph.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_reduced}
```

### User

```text
You will judge direct evidence relationships between a newly extracted trait and additional candidate nodes that are currently unconnected to it in the graph.

Judge each listed candidate directly against the new trait.

Direction rule:
- The candidate node id is source_id, the new_trait id is target_id.

Relation rules:
- state ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT
- memory ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT
- SHIFT_TO is NOT allowed in this call. Cross-type pairs (state↔trait, memory↔trait) cannot be temporal replacements.

The listed candidates are surfaced by global similarity mining over unconnected state↔trait and memory↔trait pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each candidate's content directly.

SUPPORT calibration:
- Topical similarity alone is NOT SUPPORT.
- For (state, new_trait): use SUPPORT only when the state describes a current/recent condition that would be expected GIVEN the new trait, or that the trait would naturally generalize from. Sharing a domain or keyword is not enough.
- For (memory, new_trait): use SUPPORT only when the memory captures concrete user behavior that the trait would predict; the memory must add evidential force beyond mere topic overlap.
- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.

CONTRADICT calibration:
- Use CONTRADICT only when the candidate clearly conflicts with the new trait at the persona level.
- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the trait.
- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.

Output requirements:
Output exactly {NUM_CANDIDATES} judgments — one for each listed candidate, in the listed order.
Do not skip or duplicate candidates.

[New Trait]
{new_trait_block}

[Additional Candidate States]
{candidate_states_block}

[Additional Candidate Memories]
{candidate_memories_block}

Final instruction:
Output exactly {NUM_CANDIDATES} judgments — one for each listed candidate, in the listed order.
Use the integer indices shown in brackets above for source_id and target_id.
Return strict JSON only.
```

`{NUM_CANDIDATES}` is `|candidate_states| + |candidate_memories|`.
All blocks use integer indices from a single combined `id_map` (trait first,
then candidate states, then candidate memories). Blocks use the §3.2 listing format.

Candidate counts are bounded by `TRAIT_EXTRA_REL_TOPK_STATE` (= 7) and
`TRAIT_EXTRA_REL_TOPK_MEMORY` (= 2) (post-union caps; see
`gmem5_storage_extraction.md` §9.3).

`expected_judgment_count = post-cap candidate count`. Judgment-retry policy
applies (§1.10).

### Output schema

`JUDGMENTS_SCHEMA` with reduced relation enum, integer IDs (see §1.9).

---

## 11. ⑤c Additional State-State Relation Extraction

> Judge global top-k unconnected state-state candidate pairs accumulated in
> a reservoir between ④ runs.
>
> Source: `GraphUpdater._build_5c_call` ([updater.py:1076](../updater.py#L1076)).

### System (`SYS_STATE_STATE_5C`)

```text
You are an evidence classification assistant.
Classify direct relationships for candidate state-state pairs.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships for candidate state-state pairs.
Each pair is currently unconnected in the graph.

The listed pairs are surfaced by global similarity mining over unconnected state-state pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each pair's content directly.

The pair is shown in chronological order: older_state first, newer_state second.

Direction rule:
- For SUPPORT, CONTRADICT, IRRELEVANT: source_id = older_state id, target_id = newer_state id.
- For SHIFT_TO: source_id MUST be the older_state id, target_id MUST be the newer_state id.

Relation rules:
- SUPPORT: the two states are consistent or mutually reinforcing.
- CONTRADICT: the two states are in tension but can coexist as evidence.
- SHIFT_TO: the older state has changed into the newer state and is no longer currently valid.
- IRRELEVANT: the two states are unrelated, OR are merely topically similar without evidential force in either direction.

SUPPORT calibration:
- Topical similarity alone is NOT SUPPORT.
- Temporal proximity alone is NOT SUPPORT.
- Use SUPPORT only when one state provides independent evidential force for the other (e.g., one state would be expected GIVEN the other, or both reinforce the same persona signal from independent angles).
- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.

CONTRADICT calibration:
- Use CONTRADICT only when the two states are in clear tension at the persona level.
- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the state.
- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.

SHIFT_TO caution:
- Be conservative with SHIFT_TO.
- A short-term or narrow preference does not replace a broader condition, value, role, lifestyle constraint, or safety condition unless one state explicitly invalidates the other.
- Do not use SHIFT_TO when the newer state only adds detail without invalidating the older state.
- Do not use SHIFT_TO for a topic change without explicit replacement.
- When uncertain between SHIFT_TO and CONTRADICT, choose CONTRADICT.

Output requirements:
Output exactly {NUM_PAIRS} judgments — one for each listed pair, in the listed order.
Do not skip or duplicate pairs.

[Candidate State-State Pairs]
{state_state_pair_block}

Final instruction:
Output exactly {NUM_PAIRS} judgments — one for each listed pair, in the listed order.
Use the integer indices shown in brackets above for source_id and target_id.
Return strict JSON only.
```

`{NUM_PAIRS}` is the count of candidate pairs after the stale-pair re-check.
`{state_state_pair_block}` follows the §3.3 pair format with integer indices
into the call's `id_map` (unique state IDs in order of first appearance across all pairs).
Scope is exposed per §3.2; `current_decision_impact` is not.

`expected_judgment_count = |{state_state_pair_block}|` (after stale-pair filtering).
`STATE_STATE_EXTRA_REL_TOPK` reservoir cap = 5. Judgment-retry policy applies (§1.10).

### Output schema (`JUDGMENTS_SCHEMA`, full enum, integer IDs — see §1.9).

---

## 12. ⑤d Additional State-Memory Relation Extraction

> Judge global top-k unconnected state-memory candidate pairs.
>
> Source: `GraphUpdater._build_5d_call` ([updater.py:1134](../updater.py#L1134)).

### System (`SYS_STATE_MEMORY_5D`)

```text
You are an evidence classification assistant.
Classify direct relationships for candidate state-memory pairs.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_reduced}
```

### User

```text
You will judge direct evidence relationships for candidate state-memory pairs.
Each pair is currently unconnected in the graph.

The listed pairs are surfaced by global similarity mining over unconnected state-memory pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each pair's content directly.

Direction rule:
- This call uses keyed identifiers `state_id` and `memory_id` (not `source_id`/`target_id`). Output one judgment per listed pair, naming each pair by its state and memory ids.

Relation rules:
- SUPPORT: the memory provides concrete episodic evidence that independently grounds or confirms the state.
- CONTRADICT: the memory provides concrete episodic evidence that conflicts with the state.
- IRRELEVANT: merely topically related without evidential force, or unrelated.

SUPPORT calibration:
- Topical similarity alone is NOT SUPPORT.
- Same-conversation co-extraction or temporal adjacency alone is NOT SUPPORT.
- Use SUPPORT only when the memory provides concrete episodic evidence that independently grounds or confirms the state. The memory must add an episodic fact that would still ground the state if read in isolation.
- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.

CONTRADICT calibration:
- Use CONTRADICT only when the memory contains an episodic fact that directly invalidates or conflicts with the state's claim.
- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the state.
- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.

Relation context interpretation:
Each pair carries a "Relation context" line indicating whether the state and memory come from the same conversation and how far apart they are.
- "same conversation" pairs deserve extra scrutiny: same-conversation co-extraction alone is NOT SUPPORT. Look for an episodic fact in the memory that would still ground the state outside that conversation.
- "different conversations" pairs come from temporally distinct episodes; SUPPORT here is most justified when the memory's events directly evidence the state.

Output requirements:
Output exactly {NUM_PAIRS} judgments — one for each listed pair, in the listed order.
Do not skip or duplicate pairs.

[Candidate State-Memory Pairs]
{state_memory_pair_block}

Final instruction:
Output exactly {NUM_PAIRS} judgments — one for each listed pair, in the listed order.
Use the integer indices shown in brackets above for state_id and memory_id.
Return strict JSON only.
```

`{NUM_PAIRS}` is the count of candidate pairs after the stale-pair re-check.
`{state_memory_pair_block}` follows the §3.3 pair format with separate integer
index spaces: `state_id` indexes into the call's `id_map` (state IDs in order
of first appearance), `memory_id` indexes into `id_map_b` (memory IDs in order
of first appearance). No `(scope)` annotation.

`expected_judgment_count = |{state_memory_pair_block}|` (after stale-pair filtering).
`STATE_MEMORY_EXTRA_REL_TOPK` reservoir cap = 2. Judgment-retry policy applies (§1.10).

### Output schema (`STATE_MEMORY_JUDGMENTS_SCHEMA`)

Note the **renamed keys** vs. the standard `JUDGMENTS_SCHEMA`: this call uses
`state_id` / `memory_id` instead of `source_id` / `target_id`, both as integers:

```json
{
  "judgments": [
    {
      "state_id": 0,
      "memory_id": 1,
      "relation": "SUPPORT|CONTRADICT|IRRELEVANT"
    }
  ]
}
```

---

## 13. ⑥ QA Answering (Multichoice)

> The PersonaMem benchmark uses multiple-choice questions with four options (a/b/c/d).
> The same retrieval serialization (§3.4) is used for the QA prompt.
>
> Source: `GraphGenerator.build_qa_prompt` / `answer_qa`
> ([generator.py](../generator.py)).

### 13.1 Multichoice QA

#### System (`SYS_QA_MULTICHOICE`)

```text
You are a helpful assistant answering a multiple-choice question about a user based on their memory.
The information below describes what is known about the user from past conversations.
{node_type_desc}

[Traits] are currently reliable; [Challenged Traits] may be outdated, replaced, or contradicted. For a challenged trait, prefer any "shifted to" entry and weigh listed conflicting evidence before using the trait.

Current Constraints are high-impact user states that should be honored unless the question explicitly overrides them.

Task:
- Use the user information when it is relevant to the question.
- Choose the single best option (a, b, c, or d) that fits this specific user given the retrieved memory.
- If the retrieved memory is insufficient or irrelevant, choose the option most consistent with the question on its own.

Output requirements:
Respond in JSON with two fields, in this order:
- "reasoning": one short sentence (≤ 1 sentence) stating the key factor from the retrieved memory (or its absence) that drove the choice. Write reasoning BEFORE answer.
- "answer": exactly one of "a", "b", "c", or "d".
```

`{node_type_desc}` (`_NODE_TYPE_DESC`) is the verbose extraction-oriented
definition imported from `generator.py` and reused across all extraction system
prompts in `updater.py`.

#### User (`QA_PROMPT_MULTICHOICE`)

```text
[Retrieved Memory]
{retrieved_memory}

[Question]
{question}

[Options]
{options_text}

Choose the single best answer (a, b, c, or d) based only on the information in memory.
Output JSON with "reasoning" first (≤ 1 sentence), then "answer".
```

#### Output schema (`QA_SCHEMA_MULTICHOICE`)

```json
{
  "reasoning": "...",
  "answer": "a"
}
```

`reasoning` is listed first in `properties` and `required` to elicit
chain-of-thought before the final answer. The post-processor reads only the
`answer` field (validated against `enum: ["a", "b", "c", "d"]`); anything
outside that set is mapped to `"unknown"`.

### 13.2 Common behavior

- If retrieval produces an empty string, the formatter substitutes the literal
  `No memory available.`.
- `answer_qa` retries up to `_RETRY_MAX = 3` times. On a
  `decoder prompt (length …) … maximum model length` error the
  `retrieved_memory` string is halved and the prompt is rebuilt. On any other
  error it gives up and returns `"unknown"`.
- `JSON_RETRY` (default 3) wraps the underlying generation call for parse
  failures.

---

## 14. Output Schemas Summary

Per-judgment objects for relation-extraction calls use integer `source_id`/`target_id`
(or `state_id`/`memory_id` for ⑤d). No `reasoning` or `evidence_quote` fields.

| Call                                              | Top-level keys                       | `relation` enum | `expected_judgment_count` formula |
|---------------------------------------------------|--------------------------------------|-----------------|-----------------------------------|
| ② State extraction (per-turn, ≤ 1 state)          | `states`                             | (n/a)           | `0` (extraction-only; no judgments) |
| ②b New-state relations                            | `judgments`                          | full            | `C(|new_states|, 2) + |new_states| × |prev_state_ids|` |
| ③ Memory extraction (extraction-only)             | `memory`                             | (n/a)           | `0` (extraction-only; no judgments) |
| ③b New-memory relations                           | `judgments`                          | reduced         | `(1 if previous_memory else 0) + |chunk_state_ids|` |
| ④ Trait extraction                                | `traits`                             | (n/a)           | (n/a — no judgments)              |
| ⑤a Local trait evidence judgment                  | `judgments`                          | full            | `|local_state_pool| + |local_memory_pool| + |existing_traits|` |
| ⑤b Trait-centered additional relation extraction  | `judgments`                          | reduced         | post-cap candidate count (cap=7 states, 2 memories) |
| ⑤c Additional state-state relation extraction     | `judgments`                          | full            | pairs after stale-pair re-check (reservoir cap=5) |
| ⑤d Additional state-memory relation extraction    | `judgments` (state_id / memory_id)   | reduced         | pairs after stale-pair re-check (reservoir cap=2) |
| ⑥ QA multichoice                                  | `reasoning`, `answer` (`a`/`b`/`c`/`d`) | (n/a)        | (n/a)                             |

`full` enum: `SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT`.
`reduced` enum: `SUPPORT | CONTRADICT | IRRELEVANT`.

### Retry stack
On JSON parse failure each call is re-issued with a strict repair instruction,
preserving the original semantic intent, up to `JSON_RETRY` times.

On empty `judgments` when `expected_judgment_count > 0`, the call is re-issued
up to `JUDGMENT_RETRY` times. On each retry attempt (attempt 2 onward) the hint
`"Previous attempt returned empty judgments; you MUST output exactly N judgments."`
is appended to the user prompt. Each judgment-retry attempt internally allows up
to `JSON_RETRY` JSON-parse retries. Final fallback after exhaustion: every
expected pair is treated as `IRRELEVANT` (no edges); the call completes
without hard failure.

### Canonical storage note
The prompt outputs need not encode storage direction directly. Canonical
storage direction is handled after parsing by the updater/store layer
(see `_apply_*_result` in [updater.py](../updater.py)).

### Logging
When `ENABLE_LLM_CALL_LOGGING=True`, every call's `system_prompt`,
`user_prompt`, and parsed `output` (with `_usage` stripped) is appended as
JSONL to `<prompt_log_dir>/<call_dir>/calls.jsonl`. Logging is further gated
by `LLM_CALL_LOG_FIRST_N_SESSIONS`: only sessions whose `session_id` is
strictly less than this value have their per-call logs persisted. The
`call_dir` values are listed at the top of this document; in particular this
variant adds two extra directories that the PersonaMem variant does not
have: `call_2b_state_new_rel/` and `call_3b_memory_new_rel/`.
