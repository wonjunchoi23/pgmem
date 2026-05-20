# GraphMem — Prompt Templates (ImplexConv no-response variant)

> **Scope**: Centralizes every LLM prompt emitted by the ImplexConv
> (no-response) implementation in `gmem6/`. Mirrors the current state of
> [generator.py](../generator.py), [updater.py](../updater.py), and
> [retriever.py](../retriever.py).
>
> Calls covered:
> - ② State extraction (per-turn; extraction only, no judgments) → `call_2_state`
> - ②b New-state relations (new↔new + new↔prev judgments)        → `call_2b_state_new_rel`
> - ③ Episode extraction (extraction only, no judgments)         → `call_3_episode`
> - ③b New-episode relations (new_episode↔chunk_states + new_episode↔prev_episode) → `call_3b_episode_new_rel`
> - ④ Trait extraction                                            → `call_4_trait`
> - ⑤a Local trait evidence judgment                              → `call_5a_trait_evidence`
> - ⑤b Trait-centered extra relations                             → `call_5b_trait_extra_rel`
> - ⑤c Extra state-state relations                                → `call_5c_state_state_rel`
> - ⑤d Extra state-episode relations                              → `call_5d_state_episode_rel`
> - ⑥ QA answering (opposed / supportive)                         → `call_6_qa`
>
> ⚠ **Variant notes**:
> - There is **no ① Response prompt** call.
> - States and Traits **do not carry `stability` metadata** in this variant
>   (only `scope` and, for states, `recall_priority`).
> - State extraction is **per-turn** and emits at most one state per call
>   (`STATE_EXTRACTION_H = 1`, `STATE_MAX_COUNT = 1`). ② is extraction-only —
>   it does not emit a `judgments` array. All relation judgments involving
>   newly extracted states (both new↔new and new↔previous) are produced by
>   `call_2b_state_new_rel`.
> - Episode uses the same extraction-only / relations split. ③ is
>   extraction-only and emits no `judgments`. ③b
>   (`call_3b_episode_new_rel`) judges every (new_episode, chunk_state_i)
>   pair AND the (new_episode, previous_episode) pair, in that order.
> - Retrieval has no `Tentative Traits` section — only `Traits` (stable)
>   and `Challenged Traits`.
> - User prompts do **not** include a `[Recent Domain Labels]` block.
> - QA has **two subsets**: `opposed` (free-text, ≤ 100 words) and
>   `supportive` (yes/no enum). Both repeat the question above and below
>   the retrieved episode and ask the model to incorporate any relevant
>   user circumstance — including those off the question's topic.
> - Every relation-extraction call's per-judgment output schema is minimal:
>   integer `source_id`/`target_id` (or `state_id`/`episode_id` in ⑤d) plus
>   `relation`. No `reasoning` or `evidence_quote` fields are produced
>   (see §1.9).
> - Every node-listing format drops `(scope, recall_priority)` —
>   nodes still carry these fields, but they are not surfaced to the LLM
>   (see §3.2 / §3.3).
> - The system-internal term "chunk" is replaced with natural-language
>   wording in user-facing strings (see §2.1, §6, §8, §11).
>
> Retrieval logic, storage schema, and execution flow are defined in the
> companion gmem6 design documents (`gmem6_retrieval.md`,
> `gmem6_storage_extraction.md`, `gmem6_bigflow.md`,
> `gmem6_implementation.md`, `gmem6_config.md`).

---

## Table of Contents

1. [Global Prompt Conventions](#1-global-prompt-conventions) — incl. §1.9 per-judgment auditability schema, §1.10 empty-judgment retry
2. [Reusable Prompt Components](#2-reusable-prompt-components) — incl. §2.3 keyword/domain_label disjointness block
3. [Formatting Conventions](#3-formatting-conventions)
4. [② State Extraction (per-turn, ≤ 1 state)](#4-②-state-extraction-per-turn--1-state)
5. [②b New-State Relations](#5-②b-new-state-relations)
6. [③ Episode Extraction + Episode↔State Evidence](#6-③-episode-extraction--episodestate-evidence)
7. [③b Episode ↔ Previous-Episode Relation](#7-③b-episode--previous-episode-relation)
8. [④ Trait Extraction](#8-④-trait-extraction)
9. [⑤a Local Trait Evidence Judgment](#9-⑤a-local-trait-evidence-judgment)
10. [⑤b Trait-Centered Additional Relation Extraction](#10-⑤b-trait-centered-additional-relation-extraction)
11. [⑤c Additional State-State Relation Extraction](#11-⑤c-additional-state-state-relation-extraction)
12. [⑤d Additional State-Episode Relation Extraction](#12-⑤d-additional-state-episode-relation-extraction)
13. [⑥ QA Answering (Opposed / Supportive)](#13-⑥-qa-answering-opposed--supportive)
14. [Output Schemas Summary](#14-output-schemas-summary)

---

## 1. Global Prompt Conventions

### 1.1 JSON-only rule
All ②–⑥ calls return **strict JSON only** (enforced via `guided_json` schemas).
No prose, no markdown fences, no explanations outside JSON. The QA `opposed`
subset still wraps its free-text answer inside the JSON `answer` field.

### 1.2 Evidence judgment discipline
When the model is uncertain it should prefer:
- `IRRELEVANT` over speculative linkage,
- `CONTRADICT` only when there is real tension,
- `SHIFT_TO` only for genuine temporal replacement within same-type pairs.

### 1.3 Node-writing discipline
- `State.content` — single concise sentence beginning with `The user`.
- `Episode.content` — 1–2 sentences beginning with `The user`.
- `Trait.content` — 2–3 complete sentences beginning with `The user`.

### 1.4 Label discipline
- `keywords` — up to `MAX_KEYWORDS` specific one-word terms (concrete entities, actions, constraints, or important terminology).
- `domain_label` — `MIN_DOMAIN_LABELS` to `MAX_DOMAIN_LABELS` broader topical categories at a higher abstraction level than `keywords`. Multiple labels in the same domain are encouraged (different abstraction levels / alternative names).
- Both fields: each item must be **one continuous word** (no whitespace). Compounds like `machinelearning` are allowed; items containing whitespace are dropped at storage time.
- A string may appear in only one of the two fields (apply-side `deduplicate_labels` enforces this; keywords win on overlap).
- The user prompt always inlines `MAX_KEYWORDS`, `MIN_DOMAIN_LABELS`, `MAX_DOMAIN_LABELS` from config.
- There is **no** `[Recent Domain Labels]` reuse hint block in this variant.

### 1.5 Relation enum and `SHIFT_TO` rule
Every relation-extraction call uses the same unified four-relation enum
`SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT` regardless of pair family.

`SHIFT_TO` is the natural form only for same-type temporal transitions
(`old_state → new_state`, `old_trait → new_trait`) and only when:
- the old node is no longer currently valid,
- the new node is its temporal replacement,
- the pair is not merely coexisting tension.

For cross-type pairs (state↔trait, episode↔trait, state↔episode, episode↔episode)
the apply layer stores `SHIFT_TO` verbatim if emitted; per-call user prompts
carry calibration text discouraging its use for those families.

### 1.7 Placeholder IDs
Extraction calls emit nodes with placeholder IDs that the updater rewrites to
real graph IDs after parsing:

| Call | Placeholder pattern |
|------|---------------------|
| ② state extract   | `new_0`, `new_1`, … (≤ `STATE_MAX_COUNT`) |
| ③ episode extract | `new_episode` (single)                     |
| ④ trait extract   | `new_trait` (0 or 1)                       |

The follow-up calls ②b and ③b operate on already-stored nodes, but the
LLM-facing listed blocks show **zero-based integer indices**. The model outputs
those integer indices in `source_id`/`target_id`, and the apply layer maps them
back to real graph IDs through the call-local `id_map`.

### 1.8 Metadata fields actually emitted

Distinct from the PersonaMem variant, the implex variant **omits `stability`**
from both states and traits:

| Node  | Fields emitted in this variant                   |
|-------|--------------------------------------------------|
| State   | `scope`, `recall_priority` (no stability)      |
| Trait   | `scope` (no stability)                         |
| Episode | `scope`                                        |

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

(In ⑤d the keys are `state_id`/`episode_id` instead of `source_id`/`target_id`,
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
- For episode ↔ state judgments:
    source_id = episode id, target_id = state id.
- For state ↔ trait judgments:
    source_id = state id, target_id = trait id.
- For episode ↔ trait judgments:
    source_id = episode id, target_id = trait id.
- For new_episode ↔ previous_episode judgments:
    source_id = new_episode id, target_id = previous_episode id.
- For same-type SHIFT_TO judgments (state↔state, trait↔trait):
    source_id = older node id, target_id = newer node id.

```

⑤d uses keyed identifiers `state_id` and `episode_id` (not source_id /
target_id) with separate integer index spaces. Storage is canonical `e → s`.

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

- **sequential** (`updater.py:execute_call`) — used for one-off testing and the
  PersonaMem-style synchronous loop;
- **batched** (`run_experiment.py:_drain_pending_calls`) — used by the
  production runner. The batched path was previously bypassing
  `JUDGMENT_RETRY` entirely; it now mirrors the sequential policy:
  empty-judgment jobs are re-batched with the hint, retry token usage is
  accumulated on top of the original, and the same `JUDGMENT_RETRY` cap
  applies.

After exhaustion, the **batched** path stores the call's `expected_pairs` as
**`IRRELEVANT` edges** in the graph (via `apply_irrelevant_fallback`,
canonical direction; missing nodes and pairs already directly connected are
skipped). This is the change from the earlier "no edges created" fallback:
storing IRRELEVANT edges means that subsequent ⑤b/⑤c/⑤d candidate selection
(gated by `has_direct_edge`) skips these pairs instead of re-judging them.
The call is considered **complete**, not failed.

The **sequential** path (`GraphMemModule.process_turn`) performs the
empty-judgment retry through `execute_call`, but its loop only chains
`execute_pending_call` → `apply_pending_call` and does **not** invoke
`apply_irrelevant_fallback` automatically. Expected pairs from an exhausted
call therefore remain unwritten on that path. `GraphMemModule` still exposes
`apply_irrelevant_fallback(call)` for callers that want the same behavior.

`IRRELEVANT` is a valid judgment, not an empty case — a fully `IRRELEVANT`
array is **not** retried. See `gmem6_implementation.md` §5 (Judgment-retry policy).

`JSON_RETRY` and `JUDGMENT_RETRY` stack: each judgment-retry attempt
internally allows up to `JSON_RETRY` JSON-parse retries.

② is exempt: it is extraction-only and does not produce `judgments`. All new-state
relation judgments are handled by ②b.

---

## 2. Reusable Prompt Components

### 2.1 Node-type description (`_NODE_TYPE_DESC`)

Defined once in `updater.py` and imported by `generator.py`. Shared verbatim
across every ②–⑥ system prompt — both extraction/relation calls and QA calls
read the same node-type definitions.

```text
Node types:
  State: A user condition true at the time it is expressed but subject to change — a current stance, ongoing goal, active constraint, present situation, or preference. Time-bounded and context-sensitive.
  Trait: A generalized user characteristic that tends to persist across situations and time — recurring disposition, stable preference, value, or habitual tendency. Cross-situational and relatively context-independent, not tied to one episode or temporary situation.
  Episode: A summary of what happened in a past conversation — concrete events, topics, and actions at a particular time. Not a generalized persona attribute.
```

The system-internal term "chunk" is mostly hidden from LLM-facing prompts
(rendered as "conversation" / "recent conversation" / "recent conversations"),
with one deliberate exception: ③b uses the `[Chunk States — listed in
extraction order]` block header and refers to "chunk state" pairs in the
instructions (see §7). The word is retained there because ③b judges
`new_episode ↔ chunk_state_i` for each state extracted from the same chunk,
and ordering by extraction position inside the chunk is part of the
deterministic pair layout. Spec documents may continue to use "chunk"
internally throughout.

### 2.2 Evidence relation description (`_EVID_DESC_FULL`)

Used by every relation-extraction call: ②b, ③b, ⑤a, ⑤b, ⑤c, ⑤d. There is no
longer a "reduced" variant — all calls share the same four-relation
description and the same `SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT` enum.

```text
Evidence relationships:
  IRRELEVANT: The pair is unrelated, or only weakly associated. Use this when the two pieces of information do not help interpret, update, support, or challenge each other.
  SHIFT_TO: A same-type temporal transition where the older information has changed into the newer information and is no longer currently valid. The earlier node is the source (old); the later node is the target (new).
  CONTRADICT: The two pieces of information are in clear tension or conflict, but not necessarily a temporal replacement. Both may still be meaningful evidence.
  SUPPORT: The two pieces are consistent or mutually reinforcing. Shared topic with mutual consistency is enough — SUPPORT does not require strong independent evidential force.
```

The system prompt carries **only the four label definitions**. All
calibration rules (e.g. "topical similarity alone is not SUPPORT", uncertainty
priors, SHIFT_TO cautions) are placed in the **user prompt** of each call so
that the rules can be tailored to that call's pair family.

`SHIFT_TO` from cross-type pairs (state↔trait, episode↔trait, state↔episode) is
stored verbatim if the LLM emits it; there is no apply-side filter that drops
cross-type `SHIFT_TO`. The relation enum is fully unified.

### 2.3 keywords vs domain_label discipline (`_LABEL_DISCIPLINE_BLOCK`)

Used in every node-emitting prompt: ② state extraction, ③ episode extraction,
④ trait extraction. Inserted near the per-node metadata description.

```text
keywords:
  Specific one-word terms from the source text or close variants.
  Use concrete entities, actions, constraints, or important terminology.
  Avoid broad topical categories, speaker names, and time references.

domain_label:
  Broader topical categories that could group this node with other nodes sharing the same subject area.
  Use a higher level of abstraction than keywords.
  Multiple labels in the same domain are encouraged when they capture different abstraction levels or alternative names.

Constraints (both fields):
  - Each item is one continuous word (no whitespace). Compounds like "machinelearning" are fine; items with whitespace are dropped at storage time.
  - A string may appear in only one of the two fields.
```

Apply-side enforcement:
- `deduplicate_labels` strips any `domain_label` token that case-insensitively matches a `keyword` (keywords win on overlap).
- `_validate_keywords` and `_validate_domain_labels` silently drop any item containing whitespace, so multi-word phrases never reach the graph. See `gmem6_storage_extraction.md` §4.6.

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
[{idx}] [{elapsed}]: {content}
```

`{idx}` is a **zero-based integer index** assigned at build time for each
relation-extraction call. Each call builds its own local `id_map: List[str]`
(index → real graph node ID); the LLM outputs integer `source_id`/`target_id`
values referencing this list. The apply layer maps indices back to real graph IDs.
The integer index space is local to each call and not shared across calls.

Empty lists are rendered as `(none)` (consistent with §3.1).

Node-listing prompts hide `(scope, recall_priority)` in **all**
relation-extraction calls. The fields are still produced and stored on the
nodes; they are simply not surfaced to the LLM. This applies uniformly to
②b, ③, ④, ⑤a, ⑤b, ⑤c, ⑤d.

### 3.3 Pair-list formatting for ⑤c and ⑤d

#### ⑤c state ↔ state

Pairs are rendered in **chronological order** (older first, newer second) so the
LLM has the SHIFT_TO direction encoded in the layout. `scope` is not exposed
(uniform with §3.2).

```text
Pair {pair_idx}
- older_state: [{old_node_idx}] [{old_elapsed}]: {old_content}
- newer_state: [{new_node_idx}] [{new_elapsed}]: {new_content}
```

`{old_node_idx}` and `{new_node_idx}` are integer indices into the call's `id_map`.

Append to the ⑤c user prompt:

```text
The pair is shown in chronological order: older_state first, newer_state second.
Be conservative with SHIFT_TO. Do not assume that a new topical preference replaces a broader lifestyle constraint, value, or long-running condition. SHIFT_TO is only for true content-level replacement of the same kind of attribute.
```

#### ⑤d state ↔ episode

To reduce same-conversation bias, append a "Relation context" hint with the
conversation gap and whether the two come from the same conversation:

```text
Pair {pair_idx}
- state:   [{s_idx}] [{elapsed_s}]: {content_s}
- episode: [{e_idx}] [{elapsed_e}]: {content_e}
  Relation context: {relation_context_str}
```

Where `{s_idx}` indexes into the call's state `id_map` and `{e_idx}` indexes
into the call's episode `id_map_b` (separate zero-based index spaces).

`{relation_context_str}` is:
- `"same conversation."` — when `s.conv_id == e.conv_id` (no "apart" suffix);
- `"different conversations, {gap} apart."` — otherwise, where `{gap}` is the
  compact-elapsed gap (§3.1).

Schema and storage are unaffected. Only the prompt-side rendering is changed.

### 3.4 Retrieval serialization for ⑥ QA

The retriever produces a single string with the following fixed section order
(see `GraphRetriever._serialize`). Each section header is rendered as
`[Title]` on its own line, followed directly by the body lines.

```text
[Current Constraints]
[{elapsed}] {state.content}
...

[Traits]
[{elapsed}] {trait.content}
...

[Challenged Traits]
[{elapsed}] {trait.content}
  ↳ shifted to: [{elapsed}] {newer_trait.content}
  ↳ conflicting evidence: [{elapsed}] {state_or_episode_or_trait.content}
...

[Relevant States]
[{elapsed}] {state.content}
...

[Relevant Episodes]
[{elapsed}] {episode.content}
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
- `Challenged Traits` body distinguishes two sub-bullets:
  - `↳ shifted to:` — newer trait that the challenged trait was replaced by
    (sourced from `SHIFT_TO`-out edges into the pool);
  - `↳ conflicting evidence:` — state, episode, or trait nodes contributing
    `CONTRADICT` signal toward the challenged trait.
- Section headers carry **no inline parenthetical descriptions**; only the
  `[Title]` line and body lines are emitted.

#### Final serialization example

```text
[Current Constraints]
[2d ago] The user recently injured their leg and should avoid strenuous physical activity.

[Traits]
[6mo ago] The user generally enjoys playing soccer and outdoor sports.

[Challenged Traits]
[1y ago] The user usually eats meat-heavy meals.
  ↳ shifted to: [2mo ago] The user recently became vegetarian.
  ↳ conflicting evidence: [1mo ago] The user avoided meat when choosing restaurants.

[Relevant States]
[2mo ago] The user recently became vegetarian.

[Relevant Episodes]
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
Respond in strict JSON.

{node_type_desc}
```

② is extraction-only and does **not** consume `_EVID_DESC_FULL`. All new-state
relation judgments live in ②b.

### User

```text
Task: Extract up to {STATE_MAX_COUNT} persona State(s) from the CURRENT USER UTTERANCE only. If none, return an empty states list. Extract useful persona-relevant signals as States when they are not stable enough to be Traits. Do not invent or speculate.

Use prior context and the assistant response only to resolve references. Do not extract information stated only outside the current user utterance.

Do not extract:
  - greetings, thanks, or conversational behavior,
  - descriptions of the current query rather than the user,
  - assistant-side information,
  - transient emotions with no impact on a task, decision, constraint, or safety issue,
  - stable Traits (cross-situational disposition handled by a later stage),
  - past episodes without a currently valid implication.

Each State must begin with "The user", be one concise sentence, and express a currently valid condition, constraint, goal, stance, or preference. Rewrite episodic descriptions as implied current States, not raw narration. Use placeholder ids new_0, new_1, ... in extraction order.

Metadata:
scope:
  BROAD: likely to affect decisions across multiple future tasks or topics, even when the future query does not explicitly mention this state.
  NARROW: mainly affects the current task, topic, or short-term situation.
If uncertain, choose NARROW.

recall_priority:
  HIGH: ignoring this state would make the response clearly wrong, unsafe, inconsistent with an explicit constraint, or noticeably frustrating, and the user would reasonably expect it to be remembered.
  LOW: useful context, but not necessary for an appropriate response.
If uncertain, choose LOW.

Follow the label rules:
{label_discipline_block}    # see §2.3

[PRIOR CONTEXT — disambiguation only; do not extract from here]
{prior_context_block}

[ASSISTANT RESPONSE — read-only context; do not extract from here]
{assistant_response_block}

[CURRENT USER UTTERANCE — extract from here only]
{current_user_utterance}
```

Notes:
- `{current_user_utterance}` is `current_turn.user_utterance` (raw string, no elapsed prefix).
- `{assistant_response_block}` is `current_turn.gt_response` (raw string). Used only to resolve references in the user utterance; do not extract from it.
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
      "recall_priority": "HIGH|LOW"
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
> using integer indices backed by the call-local `id_map`. ② is extraction-only
> — all new-state relation judgments live here.
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
Classify direct relationships between listed State nodes.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships involving newly extracted states.

Each node is identified by an integer index shown in brackets: [index].
{index_descriptor}    # see "Index descriptor wording" below

Judge every listed previous↔new pair. If more than one new state is listed, also judge each unordered new↔new pair exactly once. Do not output both (A,B) and (B,A).

Direction rule:
- For previous↔new pairs: source_id = previous state index, target_id = new state index.
- For previous↔new SHIFT_TO, this means older previous state → newer replacement state.
- For previous↔new SUPPORT, CONTRADICT, and IRRELEVANT, the relation is semantically symmetric, but still output source_id = previous state index and target_id = new state index for consistency.
- For new↔new SHIFT_TO: source_id = older state index as described in the content, target_id = newer replacement state index.
- For new↔new SUPPORT, CONTRADICT, and IRRELEVANT: source_id = earlier-listed new state index, target_id = later-listed new state index.

Decision priority:
- First choose IRRELEVANT if the pair has no clear evidential force.
- If related, choose SHIFT_TO only for a clear old→new replacement of the same underlying state.
- If not replacement but clearly in tension, choose CONTRADICT.
- If the two states reinforce the same persona signal, choose SUPPORT.
- When uncertain, choose IRRELEVANT.

For new↔new pairs, co-extracted states usually coexist; use SHIFT_TO only when the current utterance explicitly states temporal replacement.

Output exactly {NUM_REQUIRED_PAIRS} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs.

[Previous States — extracted before the current turn]
{previous_states_block}

[New States — extracted from the current turn]
{new_states_block}
```

**Index descriptor wording** (`{index_descriptor}`) — picks readable language
for each list depending on count:

- 0 items: omit the descriptor for that list entirely.
- 1 item: `Index {i} is a previous state.` (or `... a new state.`).
- ≥2 items: `Indices {start}..{end} are previous states.` (or `... new states.`).

The two clauses are concatenated with a single space. Example for `n_prev=2`
and `n_new=1`: `Indices 0..1 are previous states. Index 2 is a new state.`

`{previous_states_block}` and `{new_states_block}` use the §3.2 default
(no-scope) listing format. Both blocks use integer indices from a single
combined `id_map` (prev states first, new states after). If
`{previous_states_block}` is empty (session start) it renders as `(none)` and
only new↔new pairs are judged.

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
- `IRRELEVANT` judgments are stored as `IRRELEVANT` evidence edges in the canonical direction (same as `SUPPORT` / `CONTRADICT`). The IRRELEVANT edge serves two purposes: it lets the retrieval-time `signed_cache` distinguish "explicitly judged irrelevant" from "no evidence yet", and it suppresses the same pair from being re-judged in later ⑤b/⑤c/⑤d candidate selection via the `EXTRA_REL_ONLY_IF_UNCONNECTED` filter.

---

## 6. ③ Episode Extraction (extraction-only)

> Summarize the chunk into one episode. ③ does **not** classify
> any relations; all new-episode relation judgments live in ③b.
>
> Source: `GraphUpdater._build_episode_call` ([updater.py](../updater.py)).

### System (`SYS_EPISODE_EXTRACT`)

```text
You are an episode extraction assistant.
Summarize the recent conversation into one episode.
Respond in strict JSON.

{node_type_desc}
```

③ is extraction-only and does **not** consume `_EVID_DESC_FULL`. All
new-episode relation judgments (episode↔chunk_states + episode↔previous_episode)
live in ③b.

### User

```text
Task: Create exactly one Episode from the RECENT CONVERSATION. Summarize what happened in the conversation. Do not invent or speculate.

Do not extract generalized persona traits or separate state nodes. Include user traits, preferences, or conditions only when they are part of the concrete episode being summarized.

Each episode must:
- begin with "The user",
- be 1–2 sentences,
- summarize concrete events, discussed topics, actions, and developments,
- describe the episode itself, not generalized persona traits.

Metadata:
scope:
  BROAD: the episode reveals or confirms information likely to affect decisions across multiple future tasks or topics.
  NARROW: the episode is mainly tied to the current task, topic, or short-term conversation thread.
If uncertain, choose NARROW.

Follow the label rules:
{label_discipline_block}    # see §2.3

Also provide keywords and domain_label according to the label rules.

Assign the episode the placeholder id: new_episode.

[RECENT CONVERSATION]
{chunk_conversation_block}
```

The user prompt contains **only** the conversation block — ③ is purely an
episode-summary task. There is no `[Previous Episode]` block, no
`[Chunk States]` block, no `[Recent Domain Labels]` block.

### Output schema (`EPISODE_EXTRACT_SCHEMA`)

```json
{
  "episode": {
    "id": "new_episode",
    "content": "The user ...",
    "keywords": ["..."],
    "domain_label": ["..."],
    "scope": "BROAD|NARROW"
  }
}
```

The schema has **no `judgments` field** — ③ is extraction-only and the
judgment-retry wrapper skips it (`expected_judgment_count = 0`). All
new-episode relation judgments are produced by ③b (§7).

---

## 7. ③b New-Episode Relations

> Run after ③ whenever there is at least one judgeable pair involving the
> newly extracted episode. Judges, in order:
>
>   - one (new_episode, previous_episode) pair, if a previous episode exists; and
>   - one (new_episode, chunk_state_i) pair for each chunk state, in the listed order.
>
> ③ is extraction-only — all new-episode relation judgments live here.
>
> Trigger condition (apply side):
>
>   `|chunk_state_ids| ≥ 1`  OR  previous_episode exists
>
> In practice ③b fires on virtually every chunk boundary because chunks
> almost always contain ≥ 1 state.
>
> Source: `GraphUpdater._build_episode_new_rel_call` ([updater.py](../updater.py)).

### System (`SYS_EPISODE_NEW_REL`)

```text
You are an evidence classification assistant.
Classify direct relationships involving a newly extracted Episode and listed Episode/State nodes.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships involving the newly extracted episode.

Each node is identified by an integer index shown in brackets: [index].
Index 0 is the new episode. Index {prev_idx} is the previous episode. Indices {state_start}..{state_start + n_states - 1} are chunk states.

Judge:
- the (new_episode, previous_episode) pair, when a previous episode exists; and
- one (new_episode, chunk_state_i) pair for each listed chunk state, in the listed order.

Direction rule:
- For new_episode↔previous_episode pairs with SUPPORT, CONTRADICT, or IRRELEVANT: source_id = 0, target_id = previous episode index.
- For new_episode↔previous_episode pairs with SHIFT_TO: source_id = previous episode index, target_id = 0.
- For new_episode↔chunk_state pairs: source_id = 0, target_id = chunk state index for every relation.
- For cross-type SHIFT_TO in a new_episode↔chunk_state pair, this means the new episode provides evidence that updates, replaces, or invalidates the listed state.

Decision priority:
- First choose IRRELEVANT if the pair has no clear evidential force.
- If related, choose SHIFT_TO when the source information clearly updates, replaces, or shifts the target information according to the direction rule above.
- If not replacement but clearly in tension, choose CONTRADICT.
- If the new episode concretely continues, confirms, grounds, or reinforces the target, choose SUPPORT.
- When uncertain, choose IRRELEVANT.

For new_episode↔chunk_state pairs, same-conversation co-extraction alone is not SUPPORT. The episode must contain concrete evidence that grounds or confirms the state.

[New Episode]
{new_episode_block}

[Previous Episode]
{previous_episode_block_or_none}

[Chunk States — listed in extraction order]
{chunk_states_block_or_none}

Output exactly {N} judgments in this order:
  1) the (new_episode, previous_episode) judgment, if a previous episode exists;
  2) one (new_episode, chunk_state_i) judgment for each listed chunk state, in the listed order.
Do not invent states, episodes, or judgments.
```

`{new_episode_block}`, `{previous_episode_block_or_none}`, and `{chunk_states_block_or_none}`
use the §3.2 default listing format with integer indices from a single combined
`id_map` (new_episode first, then prev_episode if exists, then chunk states in order).
Either of the latter two blocks may render as `(none)`.

`{N}` is `(1 if previous_episode exists else 0) + |chunk_state_ids|`.

`expected_judgment_count = (1 if previous_episode exists else 0) + |chunk_state_ids|`.
Judgment-retry policy applies (§1.10).

### Output schema

`JUDGMENTS_SCHEMA` with the unified full relation enum, integer IDs:

```json
{
  "judgments": [
    {
      "source_id": 0,
      "target_id": 1,
      "relation": "SUPPORT|CONTRADICT|SHIFT_TO|IRRELEVANT"
    }
  ]
}
```

### Apply-side notes

- The apply layer extracts the "other" endpoint from each judgment (whichever of `source_id` / `target_id` is not `new_episode`) and stores the edge as `new_episode → other`, regardless of the LLM output direction. The direction rule above therefore affects only LLM-side reasoning, not graph storage.
- `IRRELEVANT` judgments are stored as `IRRELEVANT` evidence edges (same canonical direction as `SUPPORT` / `CONTRADICT`). See §5 for the rationale (signed_cache distinguishability + unconnected-filter suppression in ⑤b/⑤c/⑤d).

---

## 8. ④ Trait Extraction

> Extract at most one new trait from the recent two-chunk window (the
> LLM-facing header is `[Recent conversations]`).
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
Task: Infer at most {TRAIT_MAX_COUNT} persona Trait(s) from the recent conversations below. If the recent conversations do not reveal a clear new persistent pattern, return an empty traits list. Do not invent or speculate.

Use the "in general" test: a trait should still be true if you asked the user about themselves "in general" with no specific time, place, or context attached.

A trait should:
- begin with "The user",
- be 2–3 complete sentences,
- be inferable from the recent conversations as a likely persistent pattern, not just a one-time event or momentary feeling.

Trait metadata:
scope:
  BROAD  : the trait applies across all domains of the user's life — it would shape the user's approach regardless of the subject being discussed
  NARROW : a domain-specific tendency or preference mainly expressed within a particular area (e.g., a consistent style in one hobby or activity)

If uncertain between BROAD and NARROW, choose NARROW.

{label_discipline_block}    # see §2.3

Also provide:
- keywords: up to {MAX_KEYWORDS} specific single-token nouns (no whitespace, no multi-word phrases)
- domain_label: {MIN_DOMAIN_LABELS} to {MAX_DOMAIN_LABELS} single-token topical labels (no whitespace, no multi-word phrases)

Assign the trait the placeholder id: new_trait.

[Recent conversations]
{two_chunk_conversation_block}
```

The user prompt contains **only** the recent-conversations block.
Chunk states, chunk memories, and the previously extracted trait are **not**
exposed to ④ — the trait is inferred directly from the raw conversation as
a likely persistent pattern. The system-internal term "chunk" is not exposed —
the LLM-facing header is `[Recent conversations]` regardless of how many
chunks are concatenated.

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
Classify direct relationships involving a newly extracted Trait and listed State, Episode, and Trait nodes.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships involving the newly extracted trait.

Each node is identified by an integer index shown in brackets: [index].
Index 0 is the new trait. Indices {state_start}..{state_end} are recent states. Indices {episode_start}..{episode_end} are recent episodes. Index {prev_trait_idx} is the previous trait.

Judge each listed state↔new_trait pair, each listed episode↔new_trait pair, and the previous_trait↔new_trait pair when a previous trait exists.

Direction rule:
- source_id = listed state, episode, or previous trait index; target_id = 0.
- SUPPORT, CONTRADICT, and IRRELEVANT are semantically symmetric, but keep this source→target direction for consistency.
- For previous_trait↔new_trait SHIFT_TO, source_id = previous trait index and target_id = 0, following older information → newer replacement.
- For cross-type SHIFT_TO, the source state or episode provides evidence that updates, replaces, or invalidates the new trait.

Decision priority:
- First choose IRRELEVANT if the pair has no clear evidential force.
- If related, choose SHIFT_TO only when the source clearly updates, replaces, or invalidates the target.
- If not replacement but clearly in tension, choose CONTRADICT.
- If the source grounds, confirms, generalizes into, or reinforces the new trait, choose SUPPORT.
- When uncertain, choose IRRELEVANT.

Topical overlap alone is not SUPPORT. For states and episodes, the source must provide concrete evidence for the new trait. For previous_trait↔new_trait, use SHIFT_TO only when the new trait replaces the previous trait on the same underlying dimension.

[New Trait]
{new_trait_block}

[Recent States]
{recent_states_block}

[Recent Episodes]
{recent_episodes_block}

[Previous Trait]
{previous_trait_block_or_none}

Output exactly {NUM_REQUIRED_PAIRS} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs. Use the integer indices shown above for source_id and target_id.
```

`{NUM_REQUIRED_PAIRS}` is `|recent_states| + |recent_episodes| + (1 if previous_trait exists else 0)`.

All blocks use integer indices from a single combined `id_map` (new_trait first
at index 0, then recent states, then recent episodes, then previous_trait if
it exists). Blocks use the §3.2 default listing format (no scope/recall_priority). The
`previous_trait` block may render as `(none)`.

`expected_judgment_count = |recent_states_pool| + |recent_episodes_pool| + (1 if previous_trait_exists else 0)`
where `recent_states_pool` and `recent_episodes_pool` are the full sets of states / episodes from the most recent two chunks (no unconnected-only filter is applied at ⑤a — the call lists every recent-2-chunk state / episode against the new trait). The previous-trait pair is included only when a previous trait exists. Judgment-retry policy applies (§1.10).

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

> Judge extra unconnected state/trait and episode/trait candidates retrieved
> from outside the local two-chunk window.
>
> Source: `GraphUpdater._build_5b_call` ([updater.py:970](../updater.py#L970)).
> Skipped (returns `None`) when both candidate lists are empty.

### System (`SYS_TRAIT_EXTRA_REL_5B`)

```text
You are an evidence classification assistant.
Classify direct relationships involving a newly extracted Trait and additional listed State/Episode nodes.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships involving the newly extracted trait and additional candidate nodes.

Each node is identified by an integer index shown in brackets: [index].
Index 0 is the new trait. Indices {state_start}..{state_end} are candidate states. Indices {episode_start}..{episode_end} are candidate episodes.

Judge each listed candidate directly against the new trait.

Direction rule:
- source_id = candidate state or episode index, target_id = 0.
- SUPPORT, CONTRADICT, and IRRELEVANT are semantically symmetric, but keep this candidate→trait direction for consistency.
- For cross-type SHIFT_TO, the source state or episode provides evidence that updates, replaces, or invalidates the new trait.

Decision priority:
- First choose IRRELEVANT if the pair has no clear evidential force.
- If related, choose SHIFT_TO only when the source clearly updates, replaces, or invalidates the target.
- If not replacement but clearly in tension, choose CONTRADICT.
- If the source grounds, confirms, generalizes into, or reinforces the new trait, choose SUPPORT.
- When uncertain, choose IRRELEVANT.

The listed candidates were retrieved by semantic or lexical similarity, but retrieval similarity is not evidence. Topical overlap alone is not SUPPORT; the source must provide concrete evidence for the new trait.

[New Trait]
{new_trait_block}

[Additional Candidate States]
{candidate_states_block}

[Additional Candidate Episodes]
{candidate_episodes_block}

Output exactly {NUM_CANDIDATES} judgments — one for each listed candidate, in the listed order. Do not skip or duplicate candidates. Use the integer indices shown above for source_id and target_id.
```

`{NUM_CANDIDATES}` is `|candidate_states| + |candidate_episodes|`.
All blocks use integer indices from a single combined `id_map` (trait first,
then candidate states, then candidate episodes). Blocks use the §3.2 listing format.

Candidate selection is `sem_topk ∪ lex_topk → dedup`, with no rerank and no
post-union cap. `TRAIT_EXTRA_REL_TOPK_STATE` (= 7) and
`TRAIT_EXTRA_REL_TOPK_EPISODE` (= 3) are the per-list `k` (semantic and
lexical each fetched at top-`k`); the union has at most `2k` items per list
but the exact size is non-deterministic in the overlap (see
`gmem6_storage_extraction.md` §9.3).

`expected_judgment_count = |candidate_states| + |candidate_episodes|`
(post-union dedup count). Judgment-retry policy applies (§1.10).

### Output schema

`JUDGMENTS_SCHEMA` with the unified full relation enum, integer IDs (see §1.9).

---

## 11. ⑤c Additional State-State Relation Extraction

> Judge a top-k selection of unconnected state-state candidate pairs at the
> next ⑤c trigger. Between triggers, only the IDs of newly added states are
> tracked; at flush time the candidate pair pool (pending new state × all
> states, unconnected only) is reduced by a pair-level `sem_topK ∪ lex_topK`.
>
> Source: `GraphUpdater._build_5c_call` ([updater.py](../updater.py)).

### System (`SYS_STATE_STATE_5C`)

```text
You are an evidence classification assistant.
Classify direct relationships between listed State-State pairs.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships for candidate State-State pairs.
Each pair is currently unconnected in the graph.

Each node is identified by an integer index shown in brackets: [index].
The listed pairs were surfaced by semantic and lexical candidate mining; similarity alone does not imply any relation.

Each pair is shown in chronological order: older_state first, newer_state second.

Direction rule:
- For every relation, source_id = older_state index and target_id = newer_state index.
- For SHIFT_TO, this means older state → newer replacement state.
- For SUPPORT, CONTRADICT, and IRRELEVANT, the relation is semantically symmetric, but keep the older→newer output direction for consistency.

Decision priority:
- First choose IRRELEVANT if the pair has no clear evidential force.
- If related, choose SHIFT_TO only when the newer state clearly replaces the older state on the same underlying condition, constraint, stance, goal, or preference.
- If not replacement but clearly in tension, choose CONTRADICT.
- If the two states reinforce the same persona signal, choose SUPPORT.
- When uncertain, choose IRRELEVANT.

Calibration note: Topical similarity, retrieval similarity, or temporal proximity alone is not SUPPORT. A newer state that only adds detail, changes topic, or expresses a short-lived preference does not replace an older broader condition unless it explicitly invalidates it.

[Candidate State-State Pairs]
{state_state_pair_block}

Output exactly {NUM_PAIRS} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs. Use the integer indices shown above for source_id and target_id.
```

`{NUM_PAIRS}` is the count of pairs selected by the pair-level top-k.
`{state_state_pair_block}` follows the §3.3 pair format with integer indices
into the call's `id_map` (unique state IDs in order of first appearance across all pairs).
No `(scope)` or `recall_priority` annotation (uniform with §3.2).

`expected_judgment_count = |{state_state_pair_block}|`.
`STATE_STATE_EXTRA_REL_TOPK` (= 5) is the `k` used by both the semantic and the
lexical top-k over candidate `(state, state)` pairs — at most `2k = 10` pairs
per ⑤c call after sem/lex union and dedup (§9.4 of `gmem6_storage_extraction.md`).
Judgment-retry policy applies (§1.10).

### Output schema (`JUDGMENTS_SCHEMA`, full enum, integer IDs — see §1.9).

---

## 12. ⑤d Additional State-Episode Relation Extraction

> Judge a top-k selection of unconnected state-episode candidate pairs at the
> next ⑤d trigger. Between triggers, only the IDs of newly added states and
> episodes are tracked; at flush time the candidate pair pool (pending new
> state × all episodes ∪ all states × pending new episode, unconnected only) is
> reduced by a pair-level `sem_topK ∪ lex_topK`.
>
> Source: `GraphUpdater._build_5d_call` ([updater.py](../updater.py)).

### System (`SYS_STATE_EPISODE_5D`)

```text
You are an evidence classification assistant.
Classify direct relationships between listed State-Episode pairs.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
You will judge direct evidence relationships for candidate State-Episode pairs.
Each pair is currently unconnected in the graph.

State nodes use one integer index space (state_id); episode nodes use a separate integer index space (episode_id). The listed pairs were surfaced by semantic and lexical candidate mining; similarity alone does not imply any relation.

Direction rule:
- Output one judgment per listed pair using its state_id and episode_id.
- Interpret the relation as an episode→state evidence relation: the episode is the evidence source, and the state is the persona-state target.
- For cross-type SHIFT_TO, this means the episode provides evidence that updates, replaces, or invalidates the listed state.

Decision priority:
- First choose IRRELEVANT if the pair has no clear evidential force.
- If related, choose SHIFT_TO only when the episode clearly updates, replaces, or invalidates the state.
- If not replacement but the episode clearly conflicts with the state, choose CONTRADICT.
- If the episode concretely grounds, confirms, or reinforces the state, choose SUPPORT.
- When uncertain, choose IRRELEVANT.

Calibration note: Topical similarity, retrieval similarity, same-conversation co-extraction, or temporal adjacency alone is not SUPPORT. The episode must contain concrete evidence that would still ground or challenge the state if read in isolation.

Each pair includes a "Relation context" line. Use it only as a caution signal: same-conversation pairs need extra scrutiny, while different-conversation pairs may provide stronger independent evidence.

[Candidate State-Episode Pairs]
{state_episode_pair_block}

Output exactly {NUM_PAIRS} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs. Use the integer indices shown above for state_id and episode_id.
```

`{NUM_PAIRS}` is the count of pairs selected by the pair-level top-k.
`{state_episode_pair_block}` follows the §3.3 pair format with separate integer
index spaces: `state_id` indexes into the call's `id_map` (state IDs in order
of first appearance), `episode_id` indexes into `id_map_b` (episode IDs in order
of first appearance). No `(scope)` annotation.

`expected_judgment_count = |{state_episode_pair_block}|`.
`STATE_EPISODE_EXTRA_REL_TOPK` (= 3) is the `k` used by both the semantic and
the lexical top-k over candidate `(state, episode)` pairs — at most `2k = 6`
pairs per ⑤d call after sem/lex union and dedup (§9.5 of
`gmem6_storage_extraction.md`). Judgment-retry policy applies (§1.10).

### Output schema (`STATE_EPISODE_JUDGMENTS_SCHEMA`)

Note the **renamed keys** vs. the standard `JUDGMENTS_SCHEMA`: this call uses
`state_id` / `episode_id` instead of `source_id` / `target_id`, both as integers:

```json
{
  "judgments": [
    {
      "state_id": 0,
      "episode_id": 1,
      "relation": "SUPPORT|CONTRADICT|SHIFT_TO|IRRELEVANT"
    }
  ]
}
```

---

## 13. ⑥ QA Answering (Opposed / Supportive)

> The ImplexConv-no-response benchmark splits questions into two subsets.
> The same retrieval serialization (§3.4) is used for both. The choice of
> system prompt and schema is keyed by the dataset's `subset` field.
>
> Source: `GraphGenerator.build_qa_prompt` / `answer_qa`
> ([generator.py](../generator.py)).

### 13.1 `subset == "opposed"` — free-text answer

#### System (`SYS_QA_OPPOSED`)

```text
You are an assistant providing personalized help based on prior conversations with this user.

{node_type_desc}

Treat retrieved episode as candidate evidence for personalization, not as something to force into every answer. Use a retrieved episode item when it directly answers the question or materially changes the user's ability, safety, cost, time, access, motivation, or appropriateness for the requested task. When such a factor applies, reflect it concretely: acknowledge the user's goal, name the relevant factor, and adjust the recommendation accordingly. Otherwise, answer the question normally without forcing personalization.

If the current user message or recent conversation clearly conflicts with, updates, or overrides a retrieved episode item, the current message takes precedence.

Answer naturally and personalize only when the retrieved episode meaningfully supports it. Do not say "based on what I remember" or similar.
```

`{node_type_desc}` is the shared `_NODE_TYPE_DESC` defined in `updater.py` and
imported by `generator.py` (see §2.1). Both `SYS_QA_OPPOSED` and
`SYS_QA_SUPPORTIVE` inline it verbatim.

#### User (`QA_PROMPT_OPPOSED`)

```text
[Question]
{question}

[Retrieved Episode]
{retrieved_episode}

[Question]
{question}

Answer in 100 words or less. Make the answer appropriately personalized using the retrieved episode. Output JSON: {"answer": "..."}
```

The question is repeated above and below the retrieved episode so the model
keeps the question in attention while reading a long retrieval block.

#### Output schema (`QA_SCHEMA_OPPOSED`)

```json
{
  "answer": "..."
}
```

`answer` is a free-form string. The `reasoning` field that previous gmem6
revisions emitted has been removed.

### 13.2 `subset == "supportive"` — yes/no answer

#### System (`SYS_QA_SUPPORTIVE`)

```text
You are an assistant answering a yes/no question about a user based on past conversations.
{node_type_desc}

Answer based only on the retrieved user information. Answer "yes" only if the retrieved episode clearly supports yes. If the retrieved episode is insufficient, irrelevant, or ambiguous, answer "no".

Output JSON: {"answer": "yes"} or {"answer": "no"}.
```

`{node_type_desc}` is the same shared block defined in §13.1.

#### User (`QA_PROMPT_SUPPORTIVE`)

```text
[Question]
{question}

[Retrieved Episode]
{retrieved_episode}

[Question]
{question}

Answer with exactly one of: yes or no. Output JSON: {"answer": "yes"} or {"answer": "no"}.
```

#### Output schema (`QA_SCHEMA_SUPPORTIVE`)

```json
{
  "answer": "yes"
}
```

`answer` is `enum: ["yes", "no"]`. The post-processor lowercases and strips
the answer; anything outside the two-letter set is mapped to `"no"`
(default-safe fallback).

### 13.3 Common behavior

- If retrieval produces an empty string, the formatter substitutes the literal
  `No episode available.`.
- `answer_qa` retries up to `_RETRY_MAX = 3` times. On a
  `decoder prompt (length …) … maximum model length` error the
  `retrieved_episode` string is halved and the prompt is rebuilt. On any other
  error it gives up and returns `""` for both subsets.
- `JSON_RETRY` (default 3) wraps the underlying generation call for parse
  failures.

---

## 14. Output Schemas Summary

Per-judgment objects for relation-extraction calls use integer `source_id`/`target_id`
(or `state_id`/`episode_id` for ⑤d). No `reasoning` or `evidence_quote` fields.

| Call                                              | Top-level keys                       | `relation` enum | `expected_judgment_count` formula |
|---------------------------------------------------|--------------------------------------|-----------------|-----------------------------------|
| ② State extraction (per-turn, ≤ 1 state)          | `states`                             | (n/a)           | `0` (extraction-only; no judgments) |
| ②b New-state relations                            | `judgments`                          | full            | `C(|new_states|, 2) + |new_states| × |prev_state_ids|` |
| ③ Episode extraction (extraction-only)            | `episode`                            | (n/a)           | `0` (extraction-only; no judgments) |
| ③b New-episode relations                          | `judgments`                          | full            | `(1 if previous_episode else 0) + |chunk_state_ids|` |
| ④ Trait extraction                                | `traits`                             | (n/a)           | (n/a — no judgments)              |
| ⑤a Local trait evidence judgment                  | `judgments`                          | full            | `|recent_2_chunk_states| + |recent_2_chunk_episodes| + (1 if previous_trait else 0)` (no unconnected filter at ⑤a) |
| ⑤b Trait-centered additional relation extraction  | `judgments`                          | full            | post-dedup candidate count (sem-topk ∪ lex-topk; k=7 states, k=3 episodes) |
| ⑤c Additional state-state relation extraction     | `judgments`                          | full            | pair-level sem_topK ∪ lex_topK over pending unconnected state-state pool, k=5 (≤ 2k = 10) |
| ⑤d Additional state-episode relation extraction   | `judgments` (state_id / episode_id)  | full            | pair-level sem_topK ∪ lex_topK over pending unconnected state-episode pool, k=3 (≤ 2k = 6) |
| ⑥ QA opposed                                      | `answer` (free string)               | (n/a)           | (n/a)                             |
| ⑥ QA supportive                                   | `answer` (`yes` / `no`)              | (n/a)           | (n/a)                             |

`full` enum: `SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT`. There is no
longer a reduced enum — every relation-extraction call shares the same four-relation
space.

### Retry stack
On JSON parse failure each call is re-issued with a strict repair instruction,
preserving the original semantic intent, up to `JSON_RETRY` times.

On empty `judgments` when `expected_judgment_count > 0`, the call is re-issued
up to `JUDGMENT_RETRY` times. On each retry attempt (attempt 2 onward) the hint
`"Previous attempt returned empty judgments; you MUST output exactly N judgments."`
is appended to the user prompt. Each judgment-retry attempt internally allows up
to `JSON_RETRY` JSON-parse retries.

Final fallback after exhaustion (batched runner only): every expected pair in
`call.expected_pairs` is stored as an `IRRELEVANT` edge in the canonical
direction (skipping missing nodes and pairs that already have a direct edge),
and the call completes without hard failure. The sequential
`GraphMemModule.process_turn` path performs the empty-judgment retry but does
**not** invoke `apply_irrelevant_fallback`; expected pairs from an exhausted
call remain unwritten on that path (the helper is still exposed for callers
that want to invoke it manually).

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
have: `call_2b_state_new_rel/` and `call_3b_episode_new_rel/`.
