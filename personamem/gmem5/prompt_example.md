# GraphMem v5 — Concrete Prompt Examples (running state)

> **Scope**: Worked examples of every LLM call gmem5 emits at runtime, with
> realistic memory-serialized blocks. Templates and schemas live in
> [`description/gmem5_prompt.md`](description/gmem5_prompt.md); this file
> shows what the model **actually sees** during a session.
>
> Conventions:
> - Time format `[Xh Ym ago]` is computed from `created_at` via
>   `TIME_PER_CONV_ID_HOURS = 12`, `TIME_PER_TURN_MINUTES = 10`.
> - Node ids: `s_<n>`, `m_<n>`, `t_<n>`, `c_<n>`. Placeholder ids during
>   extraction: `new_0`, `new_memory`, `new_trait`.
> - Output JSON examples are illustrative; whitespace is for readability.
> - `node_type_desc`, `evidence_relation_desc_full`, `evidence_relation_desc_reduced`,
>   and `label_discipline_block` are reused everywhere — defined once in §0
>   and referenced by name afterwards.
>
> Scenario used throughout:
> - User is planning a 2-week trip to Kyoto, has a peanut allergy, recently
>   switched from running to swimming due to a knee injury, and works as a
>   freelance illustrator.
> - Current QA subset: `opposed`.

---

## §0. Reused Blocks

### `node_type_desc`

```text
Node types:
  State:  A user-specific condition that is currently or recently valid and may change over time. It captures the user's present stance, feeling, goal, constraint, situation, or preference shift. States are time-bounded and context-sensitive, and they may affect upcoming decisions or responses.
  Trait:  A generalized user characteristic that persists across situations and time. It represents recurring dispositions, stable preferences, values, or habitual tendencies. Traits are cross-situational, relatively context-independent, and supported by repeated evidence across states and memories.
  Memory: An episodic summary of what happened during a recent conversation. It captures concrete events, topics, and actions at a particular time, not generalized persona attributes.
```

### `evidence_relation_desc_full`

```text
Evidence relationships:
  SUPPORT:    The two pieces of information are consistent or mutually reinforcing.
  CONTRADICT: The two pieces of information are in tension, but they do not necessarily form a temporal replacement. Both may still be meaningful evidence.
  SHIFT_TO:   A same-type temporal transition where the older information has changed into newer information and is no longer currently valid. Use only for true old → new replacement in state-state or trait-trait pairs.
  IRRELEVANT: The pair was judged and found unrelated.
```

### `evidence_relation_desc_reduced`

```text
Evidence relationships:
  SUPPORT:    The two pieces of information are consistent or mutually reinforcing.
  CONTRADICT: The two pieces of information are in tension or conflict.
  IRRELEVANT: The pair was judged and found unrelated.
```

### `label_discipline_block`

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

---

## call_2_state — ② State Extraction (per-turn, ≤ 1 state)

**When**: Every user turn. `STATE_EXTRACTION_H = 1`, `STATE_MAX_COUNT = 1`.
**Inputs sourced from**: current `(user_utterance, gt_response)` pair plus
`STATE_REF_CONTEXT_TURNS = 3` previous `(user, assistant)` pairs from the
context cache (ignores `conv_id` boundary).

### System

```text
You are a persona state extraction assistant.
Extract user states from the current user turn and classify relationships between states.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

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

[CURRENT TURN — extract a state from this turn only]
[just now] User: My knee has been bothering me again. I've actually swapped my morning runs for swimming for now.
[just now] Agent: That sounds like a smart adjustment — swimming should be easier on the joint while you recover.

[PRIOR CONTEXT — for disambiguation only.
 DO NOT extract states from these turns. They have already been processed.]
[10m ago] User: I'm trying to lock in flights to Kyoto for the second half of next month.
[10m ago] Agent: Cherry-blossom timing might already be tight — want me to check fare windows?
[20m ago] User: Quick reminder before we look at restaurants — I'm allergic to peanuts, so anything with peanut sauce is out.
[20m ago] Agent: Understood. I'll filter peanut-containing dishes out of any suggestions.
[30m ago] User: I've been freelancing as an illustrator for a couple of years now. Project flow is steady.
[30m ago] Agent: Got it — I'll keep that in mind when we talk about scheduling or invoicing tools.

Extract user states
Extract up to 1 persona state(s) expressed by the user in the CURRENT TURN.
If the current turn contains no new persona-relevant condition, output 0 states.
Each state must:
- be a single concise sentence,
- begin with "The user",
- describe a currently or recently valid user-specific condition,
- avoid raw episodic narration unless it directly functions as a current condition.
Do NOT extract:
- pure one-time events with no current bearing,
- generic biography unless it clearly functions as a current state,
- assistant-side information.
State metadata:
scope:
  BROAD  : a persistent personal attribute, value, health condition, or lifestyle constraint that applies regardless of the current topic — it would still be relevant if the conversation shifted to a completely different subject
  NARROW : a preference or condition tied to the current task or topic — it would not transfer meaningfully to an unrelated conversation
current_decision_impact:
  HIGH : the assistant must actively remember this right now. Use HIGH only when BOTH are true: (1) the user would reasonably expect this to be remembered without re-stating it, (2) ignoring it would cause a response that is clearly wrong, unsafe, or would noticeably frustrate the user. Typical HIGH: a hard constraint just stated (allergy, refusal, deadline), a safety-relevant condition, an explicit expectation in the current exchange.
  LOW  : useful persona context, but the response would still be appropriate and acceptable without it.
Expect at most 1 HIGH per extraction call. If uncertain, always choose LOW.

{label_discipline_block}

Also provide for each state:
- keywords: up to 5 specific nouns or noun phrases
- domain_label: 3 to 5 short topical labels
Assign the state the placeholder id: new_0
```

### Expected Output

```json
{
  "states": [
    {
      "id": "new_0",
      "content": "The user has temporarily replaced morning running with swimming due to a recurring knee issue.",
      "keywords": ["knee", "swimming", "morning runs"],
      "domain_label": ["fitness", "injury_recovery", "health"],
      "scope": "BROAD",
      "current_decision_impact": "HIGH"
    }
  ]
}
```

② is extraction-only — its schema has **no `judgments` field**. All
new-state relation judgments live in ②b. `expected_judgment_count = 0` →
judgment-retry wrapper skips this call. After parsing, `deduplicate_labels`
runs (here keywords and domain_label are already disjoint, so no-op).

---

## call_2b_state_new_rel — ②b New-State Relation Judgments

**When**: Chained after ② when there is at least one judgeable pair
involving the new state(s):

```
trigger ⇔ |new_state_ids| ≥ 2  ∨  (|new_state_ids| ≥ 1 ∧ |previous_state_ids| ≥ 1)
```

In this scenario `STATE_MAX_COUNT = 1` and the previous-state set holds the
state from the immediately previous ② call (`s_14`, the Kyoto trip), so
the trigger reduces to "new ≥ 1 ∧ prev ≥ 1" and the call judges only the
single (new, previous) pair.

### System

```text
You are an evidence classification assistant.
Classify direct relationships involving newly extracted states: pairs among the new states themselves, and pairs between each new state and each previously extracted state.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
[New States]
State:  [new_0] [just now]: The user has temporarily replaced morning running with swimming due to a recurring knee issue.

[Previous States]
State:  [s_14] [10m ago]: The user is planning a 2-week trip to Kyoto in the second half of next month.

Judge:
- every pair of new states (each unordered pair, exactly once); and
- every (new state, previous state) pair.
Use SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT.
Use SHIFT_TO only for a true old_state → new_state transition.
Use the existing node ids shown in the blocks for source_id and target_id.
For every judgment, also produce:
- reasoning: 1-2 sentences explaining the call.
- evidence_quote.a / evidence_quote.b: short decisive snippets (≤ ~15 words each)
  from the source and target nodes respectively. Produce them even for IRRELEVANT.
```

### Expected Output

```json
{
  "judgments": [
    {
      "source_id": "new_0",
      "target_id": "s_14",
      "relation": "IRRELEVANT",
      "reasoning": "The knee-driven running-to-swimming switch is unrelated to the Kyoto trip planning; different life domains.",
      "evidence_quote": {
        "a": "replaced morning running with swimming due to a recurring knee issue",
        "b": "planning a 2-week trip to Kyoto in the second half of next month"
      }
    }
  ]
}
```

`expected_judgment_count = C(|new_state_ids|, 2) + |new_state_ids| × |previous_state_ids| = 0 + 1×1 = 1`.
The single `IRRELEVANT` judgment is a valid (non-empty) array → no retry.
`IRRELEVANT` produces no edge; `reasoning` and `evidence_quote` are
discarded after parsing.

### Variant 1: SHIFT_TO across ② calls

If the prior ② had instead produced `s_15: The user runs every morning to
stay in shape`, the same call would yield:

```json
{
  "judgments": [
    {
      "source_id": "s_15",
      "target_id": "new_0",
      "relation": "SHIFT_TO",
      "reasoning": "The user explicitly states the morning running has been replaced; this is a temporal transition, not just tension.",
      "evidence_quote": {
        "a": "runs every morning to stay in shape",
        "b": "swapped my morning runs for swimming for now"
      }
    }
  ]
}
```

### Variant 2: new ↔ new judgment (when `STATE_MAX_COUNT ≥ 2`)

Under a future config with `STATE_MAX_COUNT = 2`, suppose ② emitted both
`new_0` (the swimming-switch state above) and `new_1: The user can no longer
do their usual morning runs because of the knee.` Plus a previous state
`s_15: The user runs every morning to stay in shape.` Then ②b produces
`C(2,2) + 2×1 = 1 + 2 = 3` judgments — one new↔new and two new↔prev:

```json
{
  "judgments": [
    {
      "source_id": "new_0",
      "target_id": "new_1",
      "relation": "SUPPORT",
      "reasoning": "Both states describe the same knee-driven shift away from running; mutually reinforcing.",
      "evidence_quote": {
        "a": "replaced morning running with swimming due to a recurring knee issue",
        "b": "can no longer do their usual morning runs because of the knee"
      }
    },
    {
      "source_id": "s_15",
      "target_id": "new_0",
      "relation": "SHIFT_TO",
      "reasoning": "Morning running has been replaced by swimming.",
      "evidence_quote": {
        "a": "runs every morning to stay in shape",
        "b": "replaced morning running with swimming"
      }
    },
    {
      "source_id": "s_15",
      "target_id": "new_1",
      "relation": "CONTRADICT",
      "reasoning": "States the user runs every morning vs. can no longer do morning runs; coexisting tension at the moment of recording, but the SHIFT_TO above is what carries the temporal transition.",
      "evidence_quote": {
        "a": "runs every morning to stay in shape",
        "b": "can no longer do their usual morning runs"
      }
    }
  ]
}
```

The apply layer normalizes SHIFT_TO direction by `new_state_ids` membership;
when both endpoints are new, it falls back to `created_at`.

The apply side normalizes SHIFT_TO direction using `new_state_ids`
membership: source is the previous (old) `s_15`, target is the new
`new_0` → `s_15 →SHIFT_TO→ new_0` edge stored, with
`rationale="The user explicitly states the morning running has been replaced; this is a temporal transition, not just tension."`,
`evidence_quote_a="runs every morning to stay in shape"`,
`evidence_quote_b="swapped my morning runs for swimming for now"`.

---

## call_3_memory — ③ Memory Extraction (extraction-only)

**When**: At a chunk boundary (`conv_id` change). Summarizes the previous
chunk into a single episodic memory. ③ does **not** classify any
relationships; relation judgments live in ③b.

In this example the previous chunk had 4 user turns; ② produced 3 states
during it (one turn produced no state).

### System

```text
You are an episodic memory extraction assistant.
Summarize the recent conversation into one episodic memory.
Do not classify relationships in this call.
Respond in strict JSON.

{node_type_desc}
```

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
  BROAD  : the episode reveals or confirms a cross-topic user characteristic
           (e.g., a health event, a major life decision, a value-revealing
           exchange, or a standing constraint the user reaffirmed).
  NARROW : the episode is self-contained within the current topic or task —
           its implications do not extend beyond the current conversation thread.

{label_discipline_block}

Also provide:
- keywords: up to 5 specific nouns or noun phrases
- domain_label: 3 to 5 short topical labels

Assign the memory the placeholder id: new_memory.

[Recent Conversation]
[1h ago] User: I've been freelancing as an illustrator for a couple of years now. Project flow is steady.
[1h ago] Assistant: Got it — I'll keep that in mind when we talk about scheduling or invoicing tools.
[50min ago] User: Quick reminder before we look at restaurants — I'm allergic to peanuts, so anything with peanut sauce is out.
[50min ago] Assistant: Understood. I'll filter peanut-containing dishes out of any suggestions.
[40min ago] User: I'm trying to lock in flights to Kyoto for the second half of next month.
[40min ago] Assistant: Cherry-blossom timing might already be tight — want me to check fare windows?
[30min ago] User: My knee has been bothering me again. I've actually swapped my morning runs for swimming for now.
[30min ago] Assistant: That sounds like a smart adjustment — swimming should be easier on the joint while you recover.

Final instruction:
Output exactly one memory object describing the recent conversation as an episode.
Do not include relation judgments — those are produced by a follow-up call.
```

### Expected Output

```json
{
  "memory": {
    "id": "new_memory",
    "content": "The user briefly anchored their working context (freelance illustrator) and then opened planning topics: a Kyoto trip and dietary/exercise constraints, including a peanut allergy and a knee-driven switch from running to swimming.",
    "keywords": ["Kyoto", "peanut allergy", "knee", "swimming"],
    "domain_label": ["travel_planning", "health", "lifestyle"],
    "scope": "BROAD"
  }
}
```

③ is extraction-only — its schema has **no `judgments` field**. All
new-memory relation judgments live in ③b. `expected_judgment_count = 0` →
the judgment-retry wrapper skips this call.

---

## call_3b_memory_new_rel — ③b New-Memory Relations

**When**: Chained after ③ when there is at least one judgeable pair
involving the new memory:

```
trigger ⇔ |chunk_state_ids| ≥ 1  ∨  previous_memory exists
```

In practice ③b fires on virtually every chunk boundary because chunks
almost always contain ≥ 1 state. In this scenario the chunk produced
3 states (`s_13`, `s_14`, `s_16`) and there is a previous memory `m_7`,
so ③b judges `(1) + 3 = 4` pairs.

### System

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

Judge:
- the (new_memory, previous_memory) pair, when a previous memory exists; and
- one (new_memory, chunk_state_i) pair for each listed chunk state, in the listed order.

Direction rules (FIXED):

For (new_memory, previous_memory):
  - source_id MUST be the new_memory id.
  - target_id MUST be the previous_memory id.
  - evidence_quote.a is the new-memory snippet.
  - evidence_quote.b is the previous-memory snippet.

For (new_memory, chunk_state_i):
  - source_id MUST be the new_memory id.
  - target_id MUST be the chunk_state_i id.
  - evidence_quote.a is the memory-side snippet.
  - evidence_quote.b is the state-side snippet.

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

Output requirements:
Output exactly 4 judgments in this order:
  1) the (new_memory, previous_memory) judgment, if a previous memory exists;
  2) one (new_memory, chunk_state_i) judgment for each listed chunk state, in the listed order.
Use each listed chunk_state id exactly once as a target_id.
If no chunk states are listed AND no previous memory exists, return "judgments": [].
Do not invent states, memories, or judgments.

For every judgment, also produce:
- reasoning: 1-2 sentences explaining the call.
- evidence_quote.a / evidence_quote.b: short decisive snippets (≤ ~15 words each).
  Produce them even for IRRELEVANT.

[New Memory]
[m_8] [now]: The user briefly anchored their working context (freelance illustrator) and then opened planning topics: a Kyoto trip and dietary/exercise constraints, including a peanut allergy and a knee-driven switch from running to swimming.

[Previous Memory]
[m_7] [12h ago]: The user discussed budgeting tools for irregular freelance income and walked through a recurring monthly transfer setup.

[Chunk States — listed in extraction order]
State:  [s_13] [1h ago]: The user works as a freelance illustrator with steady project flow.
State:  [s_14] [40min ago]: The user is planning a 2-week trip to Kyoto in the second half of next month.
State:  [s_16] [30min ago]: The user has temporarily replaced morning running with swimming due to a recurring knee issue.

Final instruction:
Output exactly 4 judgments in the order specified above (previous_memory first, then chunk states in listed order).
Use the existing node ids shown above for source_id and target_id.
Return strict JSON only.
```

### Expected Output

```json
{
  "judgments": [
    {
      "source_id": "m_8",
      "target_id": "m_7",
      "relation": "IRRELEVANT",
      "reasoning": "Both episodes mention freelance context but describe different concrete activities (travel/health planning vs. budgeting tooling); the new memory does not continue, confirm, or reinforce the previous one.",
      "evidence_quote": {
        "a": "anchored their working context (freelance illustrator)",
        "b": "budgeting tools for irregular freelance income"
      }
    },
    {
      "source_id": "m_8",
      "target_id": "s_13",
      "relation": "IRRELEVANT",
      "reasoning": "The memory only references freelance work as background framing; no episodic event would independently ground the freelance-illustrator state if read in isolation.",
      "evidence_quote": {
        "a": "briefly anchored their working context (freelance illustrator)",
        "b": "works as a freelance illustrator with steady project flow"
      }
    },
    {
      "source_id": "m_8",
      "target_id": "s_14",
      "relation": "SUPPORT",
      "reasoning": "The memory captures the user actively planning Kyoto flights, providing episodic evidence for the trip-planning state.",
      "evidence_quote": {
        "a": "opened planning topics: a Kyoto trip",
        "b": "planning a 2-week trip to Kyoto"
      }
    },
    {
      "source_id": "m_8",
      "target_id": "s_16",
      "relation": "SUPPORT",
      "reasoning": "The memory records the explicit running-to-swimming switch motivated by the knee, directly grounding the state.",
      "evidence_quote": {
        "a": "knee-driven switch from running to swimming",
        "b": "temporarily replaced morning running with swimming"
      }
    }
  ]
}
```

`expected_judgment_count = (1 if previous_memory else 0) + |chunk_states| = 1 + 3 = 4`.
Four judgments returned → no retry. After apply: one `m → s` edge for `s_14`,
one for `s_16`, both with `rationale` and `evidence_quote_{a,b}` attributes;
the `IRRELEVANT` judgments for `m_7` and `s_13` produce no edges.

`expected_judgment_count = 1`. One judgment, non-empty array → no retry.
No edge stored (IRRELEVANT).

---

## call_4_trait — ④ Trait Extraction

**When**: Every 2-chunk boundary (`TRAIT_EXTRACTION_CHUNKS = 2`). Runs
after ③.

### System

```text
You are a persona trait extraction assistant.
Infer at most one new long-term trait from the accumulated recent evidence.
Respond in strict JSON.

{node_type_desc}
```

### User

```text
A trait is a generalized user characteristic that persists across
situations and time. Use the "in general" test: a trait should still
be true if you asked the user about themselves "in general" with no
specific time, place, or context attached.

Extract 0 or 1 trait from the recent two conversations below.
A trait should:
- begin with "The user",
- be 2–3 complete sentences,
- describe a generalized characteristic that persists across situations,
- capture a stable preference, value, disposition, or habitual tendency,
- be inferable from the recent conversations as a likely persistent pattern,
  not just a one-time event or momentary feeling.

If the recent conversations do not reveal a clear new persistent pattern, output 0 traits.

Trait metadata:
scope:
  BROAD  : the trait applies across all domains of the user's life — it would shape the user's approach regardless of the subject being discussed
  NARROW : a domain-specific tendency or preference mainly expressed within a particular area (e.g., a consistent style in one hobby or activity)

{label_discipline_block}

Also provide:
- keywords: up to 5 specific nouns or noun phrases
- domain_label: 3 to 5 short topical labels

Assign the trait the placeholder id: new_trait.

[Recent Two Conversations]
[12h ago] User: I need a budgeting setup that handles uneven freelance pay.
[12h ago] Assistant: Sure — let's start by listing fixed monthly outflows.
[11h ago] User: I'd rather automate as much as possible. Manual tracking falls apart for me.
[11h ago] Assistant: Then a recurring-transfer rule plus an automatic categorizer should help.
[11h ago] User: Yes, exactly. I want to set it once and forget it.
[11h ago] Assistant: Got it. Here is a minimal rule list you can paste into your bank app.
[11h ago] User: Perfect. That's the kind of low-maintenance setup I want.
[11h ago] Assistant: Glad it works for you. We can revisit thresholds in a few months.
[1h ago] User: I've been freelancing as an illustrator for a couple of years now. Project flow is steady.
[1h ago] Assistant: Got it — I'll keep that in mind when we talk about scheduling or invoicing tools.
[50min ago] User: Quick reminder before we look at restaurants — I'm allergic to peanuts, so anything with peanut sauce is out.
[50min ago] Assistant: Understood. I'll filter peanut-containing dishes out of any suggestions.
[40min ago] User: I'm trying to lock in flights to Kyoto for the second half of next month.
[40min ago] Assistant: Cherry-blossom timing might already be tight — want me to check fare windows?
[30min ago] User: My knee has been bothering me again. I've actually swapped my morning runs for swimming for now.
[30min ago] Assistant: That sounds like a smart adjustment — swimming should be easier on the joint while you recover.

Final instruction:
Output 0 or 1 trait inferred from the recent two conversations.
If no clear persistent pattern is revealed, return an empty traits list.
Return strict JSON only.
```

The user prompt contains **only** `[Recent Two Conversations]`. Chunk
states, chunk memories, and the most-recent-existing-trait block are not
exposed to ④ — the trait is inferred directly from raw conversation as a
likely persistent pattern. Concrete trait examples and state-vs-trait
comparison pairs are also no longer included, to avoid canonical-bias
mimicry.

### Expected Output

```json
{
  "traits": [
    {
      "id": "new_trait",
      "content": "The user gravitates toward low-maintenance, automated systems across both personal-finance and lifestyle decisions. They prefer to define rules once and let the system run, rather than handle ongoing manual upkeep.",
      "keywords": ["automation", "set-and-forget", "low-maintenance"],
      "domain_label": ["lifestyle", "decision_style", "tooling"],
      "scope": "BROAD"
    }
  ]
}
```

No judgments emitted by ④ itself. The follow-up calls ⑤a (and possibly
⑤b) judge this new trait against the surrounding evidence.

---

## call_5a_trait_evidence — ⑤a Local Trait Evidence Judgment

**When**: After ④ when a new trait was created.

### System

```text
You are an evidence classification assistant.
Classify the direct relationship between a newly extracted trait and nearby existing nodes.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
[New Trait]
Trait:  [new_trait] [just now]: The user gravitates toward low-maintenance, automated systems across both personal-finance and lifestyle decisions. They prefer to define rules once and let the system run, rather than handle ongoing manual upkeep.

[Recent States]
State:  [s_11] [12h 0m ago]: The user prefers fully automated, set-and-forget personal-finance setups.
State:  [s_13] [1h 0m ago]: The user works as a freelance illustrator with steady project flow.
State:  [s_14] [40m ago]: The user is planning a 2-week trip to Kyoto in the second half of next month.
State:  [s_16] [30m ago]: The user has temporarily replaced morning running with swimming due to a recurring knee issue.

[Recent Memories]
Memory: [m_7] [12h 0m ago]: The user discussed budgeting tools for irregular freelance income and walked through a recurring monthly transfer setup.
Memory: [m_8] [just now]: The user briefly anchored their working context (freelance illustrator) and then opened planning topics: a Kyoto trip and dietary/exercise constraints, including a peanut allergy and a knee-driven switch from running to swimming.

[Previous Trait]
Trait:  [t_3] [3d 0h ago]: The user prefers concrete, structured plans over open-ended discussion. They tend to push toward concrete deliverables quickly when given a choice.

Judge all required pairs:
- each recent state ↔ new trait
- each recent memory ↔ new trait
- previous trait ↔ new trait
Rules:
- For state/trait and memory/trait pairs, use only SUPPORT, CONTRADICT, or IRRELEVANT.
- For previous_trait ↔ new_trait, SHIFT_TO is allowed only for true old_trait → new_trait replacement.
- Prefer IRRELEVANT when no real semantic relation is present.
For every judgment, also produce:
- reasoning: 1-2 sentences explaining the call.
- evidence_quote.a (source side, ≤ ~15 words) and evidence_quote.b (target side, ≤ ~15 words).
  Produce them even for IRRELEVANT.
```

### Expected Output

```json
{
  "judgments": [
    {
      "source_id": "s_11",
      "target_id": "new_trait",
      "relation": "SUPPORT",
      "reasoning": "The set-and-forget finance preference is exactly the automation/low-maintenance pattern the trait generalizes.",
      "evidence_quote": {
        "a": "fully automated, set-and-forget personal-finance setups",
        "b": "gravitates toward low-maintenance, automated systems"
      }
    },
    {
      "source_id": "s_13",
      "target_id": "new_trait",
      "relation": "IRRELEVANT",
      "reasoning": "Freelance-illustrator employment is a domain fact that does not speak to automation preference.",
      "evidence_quote": {
        "a": "works as a freelance illustrator",
        "b": "low-maintenance, automated systems"
      }
    },
    {
      "source_id": "s_14",
      "target_id": "new_trait",
      "relation": "IRRELEVANT",
      "reasoning": "Trip-planning intent does not indicate automation style either way.",
      "evidence_quote": {
        "a": "planning a 2-week trip to Kyoto",
        "b": "low-maintenance, automated systems"
      }
    },
    {
      "source_id": "s_16",
      "target_id": "new_trait",
      "relation": "IRRELEVANT",
      "reasoning": "Health-driven exercise switch is unrelated to automation/low-maintenance disposition.",
      "evidence_quote": {
        "a": "replaced morning running with swimming",
        "b": "low-maintenance, automated systems"
      }
    },
    {
      "source_id": "m_7",
      "target_id": "new_trait",
      "relation": "SUPPORT",
      "reasoning": "The episode shows the user choosing recurring rules over manual tracking — concrete instance of the trait.",
      "evidence_quote": {
        "a": "recurring monthly transfer setup",
        "b": "define rules once and let the system run"
      }
    },
    {
      "source_id": "m_8",
      "target_id": "new_trait",
      "relation": "IRRELEVANT",
      "reasoning": "This memory captures travel and health topics, with no evidence about automation style.",
      "evidence_quote": {
        "a": "Kyoto trip and dietary/exercise constraints",
        "b": "low-maintenance, automated systems"
      }
    },
    {
      "source_id": "t_3",
      "target_id": "new_trait",
      "relation": "SUPPORT",
      "reasoning": "Both traits describe a structured, plan-first disposition; the new trait extends it specifically toward automation. Not a temporal replacement.",
      "evidence_quote": {
        "a": "prefers concrete, structured plans",
        "b": "define rules once and let the system run"
      }
    }
  ]
}
```

`expected_judgment_count = |recent_states| + |recent_memories| + |existing_traits| = 4 + 2 + 1 = 7`.
Seven judgments returned → no retry. Edges stored: `s_11 →SUPPORT→ new_trait`,
`m_7 →SUPPORT→ new_trait`, `t_3 →SUPPORT→ new_trait`. The four `IRRELEVANT`
judgments produce no edges.

---

## call_5b_trait_extra_rel — ⑤b Trait-Centered Additional Relation Extraction

**When**: After ⑤a, when `ENABLE_EXTRA_RELATION_EXTRACTION = True` and at
least one extra-candidate is produced. Candidates are unconnected
state/memory nodes outside the local 2-chunk window, retrieved via
semantic+lexical top-k union, deduplicated, reranked by `pair_score`, then
capped at `TRAIT_EXTRA_REL_TOPK_STATE = 10` and
`TRAIT_EXTRA_REL_TOPK_MEMORY = 2`.

### System

```text
You are an evidence classification assistant.
Classify direct relationships between a new trait and additional candidate nodes retrieved from the graph.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_reduced}
```

### User

```text
[New Trait]
Trait:  [new_trait] [just now]: The user gravitates toward low-maintenance, automated systems across both personal-finance and lifestyle decisions. They prefer to define rules once and let the system run, rather than handle ongoing manual upkeep.

[Additional Candidate States]
State:  [s_4] [5d 0h ago]: The user uses a single email rule to triage incoming work requests by client priority.
State:  [s_7] [4d 6h ago]: The user resists adopting new productivity tools that require ongoing configuration.
State:  [s_9] [3d 12h ago]: The user paid extra for a smart-home hub specifically to consolidate device control.

[Additional Candidate Memories]
Memory: [m_4] [6d 0h ago]: The user spent an afternoon migrating from a manual time-tracking spreadsheet to an automatic time-tracker app.

The listed candidates are currently unconnected to the new trait.
Judge each candidate directly against the new trait.
Rules:
- state ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT
- memory ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT
- Prefer IRRELEVANT when the candidate is topically similar but not genuinely evidential.
For every judgment, also produce:
- reasoning: 1-2 sentences explaining the call.
- evidence_quote.a (candidate side, ≤ ~15 words) and evidence_quote.b (new trait side, ≤ ~15 words).
  Produce them even for IRRELEVANT.
```

### Expected Output

```json
{
  "judgments": [
    {
      "source_id": "s_4",
      "target_id": "new_trait",
      "relation": "SUPPORT",
      "reasoning": "Single-rule email triage is a concrete instance of defining a rule once and letting the system run.",
      "evidence_quote": {
        "a": "single email rule to triage incoming work requests",
        "b": "define rules once and let the system run"
      }
    },
    {
      "source_id": "s_7",
      "target_id": "new_trait",
      "relation": "SUPPORT",
      "reasoning": "Resisting tools that require ongoing configuration is the negative-framed version of the same low-maintenance disposition.",
      "evidence_quote": {
        "a": "resists adopting new productivity tools that require ongoing configuration",
        "b": "low-maintenance, automated systems"
      }
    },
    {
      "source_id": "s_9",
      "target_id": "new_trait",
      "relation": "SUPPORT",
      "reasoning": "Paying to consolidate device control reflects the same preference for unified, low-effort setups.",
      "evidence_quote": {
        "a": "smart-home hub specifically to consolidate device control",
        "b": "low-maintenance, automated systems"
      }
    },
    {
      "source_id": "m_4",
      "target_id": "new_trait",
      "relation": "SUPPORT",
      "reasoning": "An explicit migration from manual to automatic time-tracking is a clear episodic instance of the trait.",
      "evidence_quote": {
        "a": "migrating from a manual time-tracking spreadsheet to an automatic time-tracker app",
        "b": "define rules once and let the system run"
      }
    }
  ]
}
```

`expected_judgment_count = post-cap candidate count = 3 + 1 = 4`.
Four edges stored.

---

## call_5c_state_state_rel — ⑤c Additional State-State Relation Extraction

**When**: Every 2-chunk boundary regardless of new trait, when
`ENABLE_EXTRA_RELATION_EXTRACTION = True`. Consumes the global unconnected
state-state pair reservoir's current top-k.

### System

```text
You are an evidence classification assistant.
Classify direct relationships for candidate state-state pairs.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_full}
```

### User

```text
[Candidate State-State Pairs]
Pair 0
- source: [s_11] [12h 0m ago]: The user prefers fully automated, set-and-forget personal-finance setups.
- target: [s_4]  [5d 0h ago]: The user uses a single email rule to triage incoming work requests by client priority.
Pair 1
- source: [s_16] [30m ago]: The user has temporarily replaced morning running with swimming due to a recurring knee issue.
- target: [s_2]  [10d 0h ago]: The user runs three times a week as part of their regular routine.
Pair 2
- source: [s_14] [40m ago]: The user is planning a 2-week trip to Kyoto in the second half of next month.
- target: [s_5]  [20d 0h ago]: The user is saving toward a flexible 6-month sabbatical.

Each pair is currently unconnected in the graph.
Judge each pair directly.
Use:
- SUPPORT when the two states are consistent or mutually reinforcing,
- CONTRADICT when they are in tension but can coexist as evidence,
- SHIFT_TO only when the older state has changed into the newer state,
- IRRELEVANT when they are unrelated.
Use SHIFT_TO conservatively. It must be a real temporal transition of the form old_state → new_state.
Prefer IRRELEVANT when two states are topically similar but not genuinely evidential.
For every judgment, also produce:
- reasoning: 1-2 sentences explaining the call.
- evidence_quote.a (source side, ≤ ~15 words) and evidence_quote.b (target side, ≤ ~15 words).
  Produce them even for IRRELEVANT.
```

### Expected Output

```json
{
  "judgments": [
    {
      "source_id": "s_11",
      "target_id": "s_4",
      "relation": "SUPPORT",
      "reasoning": "Both states describe one-rule, hands-off setups in different domains; mutually reinforcing evidence of the same preference.",
      "evidence_quote": {
        "a": "fully automated, set-and-forget personal-finance setups",
        "b": "single email rule to triage incoming work requests"
      }
    },
    {
      "source_id": "s_2",
      "target_id": "s_16",
      "relation": "SHIFT_TO",
      "reasoning": "The older running routine has been explicitly replaced by swimming due to the knee; this is a true temporal replacement.",
      "evidence_quote": {
        "a": "runs three times a week as part of their regular routine",
        "b": "replaced morning running with swimming due to a recurring knee issue"
      }
    },
    {
      "source_id": "s_14",
      "target_id": "s_5",
      "relation": "IRRELEVANT",
      "reasoning": "A 2-week trip and a 6-month sabbatical plan are different scope and time horizon; no direct evidential link.",
      "evidence_quote": {
        "a": "planning a 2-week trip to Kyoto",
        "b": "saving toward a flexible 6-month sabbatical"
      }
    }
  ]
}
```

Note that the LLM swapped the order in pair 1 to express the correct
SHIFT_TO direction (`s_2 → s_16`, old → new). The apply side accepts any
ordering for `SHIFT_TO` and re-normalizes to old→new using `created_at`.

After consumption, all 3 pairs are removed from the reservoir regardless
of result; the reservoir rebuilds incrementally before the next ⑤c trigger.

---

## call_5d_state_memory_rel — ⑤d Additional State-Memory Relation Extraction

**When**: Every 2-chunk boundary regardless of new trait, when
`ENABLE_EXTRA_RELATION_EXTRACTION = True`. Consumes the global unconnected
state-memory pair reservoir's current top-k.

### System

```text
You are an evidence classification assistant.
Classify direct relationships for candidate state-memory pairs.
Respond in strict JSON.

{node_type_desc}

{evidence_relation_desc_reduced}
```

### User

```text
[Candidate State-Memory Pairs]
Pair 0
- state:  [s_16] [30m ago]: The user has temporarily replaced morning running with swimming due to a recurring knee issue.
- memory: [m_3] [8d 0h ago]: The user complained about a sore knee after a half-marathon and asked about recovery routines.
Pair 1
- state:  [s_14] [40m ago]: The user is planning a 2-week trip to Kyoto in the second half of next month.
- memory: [m_5] [4d 0h ago]: The user browsed cherry-blossom viewing locations and saved a shortlist of three city options.

Each pair is currently unconnected in the graph.
Judge each pair directly.
Rules:
- Use only SUPPORT, CONTRADICT, or IRRELEVANT.
- If the memory provides concrete episodic evidence that supports the state, use SUPPORT.
- If the memory provides concrete episodic evidence against the state, use CONTRADICT.
- If they are merely topically related without evidential force, use IRRELEVANT.
For every judgment, also produce:
- reasoning: 1-2 sentences explaining the call.
- evidence_quote.a (state side, ≤ ~15 words) and evidence_quote.b (memory side, ≤ ~15 words).
  Produce them even for IRRELEVANT.
```

### Expected Output

```json
{
  "judgments": [
    {
      "state_id": "s_16",
      "memory_id": "m_3",
      "relation": "SUPPORT",
      "reasoning": "The earlier knee complaint episode supplies concrete history that grounds the recurring-knee-issue framing in the current state.",
      "evidence_quote": {
        "a": "recurring knee issue",
        "b": "complained about a sore knee after a half-marathon"
      }
    },
    {
      "state_id": "s_14",
      "memory_id": "m_5",
      "relation": "SUPPORT",
      "reasoning": "Browsing and shortlisting cherry-blossom viewing locations is direct episodic evidence of trip planning.",
      "evidence_quote": {
        "a": "planning a 2-week trip to Kyoto",
        "b": "browsed cherry-blossom viewing locations"
      }
    }
  ]
}
```

Stored canonically as `m_3 → s_16` and `m_5 → s_14` (both `SUPPORT`).
Reservoir entries removed after consumption.

---

## Retrieval Serialization (input to ⑥ QA)

For the running scenario, suppose during QA:

- `query = "I'm at the airport heading to Kyoto. Any quick lunch ideas before boarding?"`
- After APS construction (top-`k_aps = 3` HIGH-impact non-SHIFT_TO-source
  states ranked by `seed_score`), `s_aps` is:
  - `s_15` (peanut allergy, HIGH)
  - `s_14` (Kyoto trip, HIGH)
  - `s_16` (knee/swimming switch, HIGH) — note this is HIGH because it
    was just stated in the recent exchange.
- After `s_seed` retrieval (excluding APS members, top-`k_s = 7`):
  the freelance state `s_13` enters; older non-APS states like `s_4`,
  `s_5` enter via expansion only.
- After expansion + final-set assembly:
  - `t_final = [t_5]` (the new automation trait — stable, no contradictions)
  - `s_final = [s_13, s_4, s_11, s_9, s_7]` (top-`k_sf = 5`)
  - `m_final = [m_8, m_5, m_7, m_3, m_4]` (top-`k_m_final = 5`)

The retriever produces this exact string for `{retrieved_memory}`:

```text
[Current Constraints]
(These are high-impact user states that should be reflected in the response.)
[20m ago] The user is allergic to peanuts.
[40m ago] The user is planning a 2-week trip to Kyoto in the second half of next month.
[30m ago] The user has temporarily replaced morning running with swimming due to a recurring knee issue.

[Traits]
[just now] The user gravitates toward low-maintenance, automated systems across both personal-finance and lifestyle decisions. They prefer to define rules once and let the system run, rather than handle ongoing manual upkeep.

[Challenged Traits]

[Relevant States]
[1h 0m ago] The user works as a freelance illustrator with steady project flow.
[5d 0h ago] The user uses a single email rule to triage incoming work requests by client priority.
[12h 0m ago] The user prefers fully automated, set-and-forget personal-finance setups.
[3d 12h ago] The user paid extra for a smart-home hub specifically to consolidate device control.
[4d 6h ago] The user resists adopting new productivity tools that require ongoing configuration.

[Relevant Memories]
[just now] The user briefly anchored their working context (freelance illustrator) and then opened planning topics: a Kyoto trip and dietary/exercise constraints, including a peanut allergy and a knee-driven switch from running to swimming.
[4d 0h ago] The user browsed cherry-blossom viewing locations and saved a shortlist of three city options.
[12h 0m ago] The user discussed budgeting tools for irregular freelance income and walked through a recurring monthly transfer setup.
[8d 0h ago] The user complained about a sore knee after a half-marathon and asked about recovery routines.
[6d 0h ago] The user spent an afternoon migrating from a manual time-tracking spreadsheet to an automatic time-tracker app.
```

For QA, `[Recent Conversation]` is omitted (`INCLUDE_RECENT_CONVERSATION_FOR_QA = False`).

### Variant: with a Challenged Trait

If `t_3` had a CONTRADICT edge from a recent state (e.g., a state showing
the user actually preferring open-ended brainstorming), the section would
render as:

```text
[Challenged Traits]
[3d 0h ago] The user prefers concrete, structured plans over open-ended discussion. They tend to push toward concrete deliverables quickly when given a choice.
  [contradicts] [2h 0m ago] The user just spent 30 minutes brainstorming open-ended ideas for the Kyoto trip with no concrete deliverable.
```

Note that the `rationale` and `evidence_quote_{a,b}` attached to the
CONTRADICT edge are **not** rendered in this serialization — they are edge
metadata only.

---

## call_6_qa — ⑥ QA Answering (`subset == "opposed"`)

**When**: Once per QA question. Memory is not mutated.

### System (`SYS_QA_OPPOSED`)

```text
You are a helpful assistant who has been talking with this user across multiple sessions.
The information below describes what is known about the user from past conversations.

{node_type_desc}

[Traits] contains currently reliable traits.
[Challenged Traits] contains traits that may be outdated, replaced, or contradicted.
If a challenged trait has a "shifted to" entry, prefer the newer shifted-to information over the old trait.
If a challenged trait has conflicting evidence, weigh the conflicting evidence before using the trait.

Current Constraints are high-impact user states that should be honored in the response unless the current question explicitly overrides them.

Task:
- Use the user information when it is relevant to the question.
- Give an answer that fits this specific user when relevant; otherwise answer normally.
- Answer naturally. Do not explicitly reference the memory (e.g., do not say "based on what I remember" or "according to your past conversations").

Output requirements:
Respond in JSON with two fields, in this order:
- "reasoning": one short sentence (≤ 1 sentence) stating the key factor from the retrieved memory that drove your answer. Write reasoning BEFORE answer.
- "answer": the final reply to the user.
```

### User (`QA_PROMPT_OPPOSED`)

```text
[Retrieved Memory]
[Current Constraints]
(These are high-impact user states the assistant should honor in the response, unless the user explicitly overrides them in the current message.)
[20min ago] The user is allergic to peanuts.
[40min ago] The user is planning a 2-week trip to Kyoto in the second half of next month.
[30min ago] The user has temporarily replaced morning running with swimming due to a recurring knee issue.

[Traits]
[now] The user gravitates toward low-maintenance, automated systems across both personal-finance and lifestyle decisions. They prefer to define rules once and let the system run, rather than handle ongoing manual upkeep.

[Challenged Traits]
(none)

[Relevant States]
[1h ago] The user works as a freelance illustrator with steady project flow.
[5d ago] The user uses a single email rule to triage incoming work requests by client priority.
[12h ago] The user prefers fully automated, set-and-forget personal-finance setups.
[3d ago] The user paid extra for a smart-home hub specifically to consolidate device control.
[4d ago] The user resists adopting new productivity tools that require ongoing configuration.

[Relevant Memories]
[now] The user briefly anchored their working context (freelance illustrator) and then opened planning topics: a Kyoto trip and dietary/exercise constraints, including a peanut allergy and a knee-driven switch from running to swimming.
[4d ago] The user browsed cherry-blossom viewing locations and saved a shortlist of three city options.
[12h ago] The user discussed budgeting tools for irregular freelance income and walked through a recurring monthly transfer setup.
[8d ago] The user complained about a sore knee after a half-marathon and asked about recovery routines.
[6d ago] The user spent an afternoon migrating from a manual time-tracking spreadsheet to an automatic time-tracker app.

[Question]
I'm at the airport heading to Kyoto. Any quick lunch ideas before boarding?

Answer in 1-2 sentences, personalized to this user when the retrieved memory is relevant.
Output JSON with "reasoning" first (≤ 1 sentence), then "answer".
```

### Expected Output

```json
{
  "reasoning": "Peanut allergy is a HIGH-impact current constraint, so the answer must be peanut-safe and convenience-store-fast for the airport context.",
  "answer": "Grab an onigiri and a clear miso soup combo from a convenience store — fast, filling, and easy to scan ingredient labels for peanuts. If you want something warmer, a plain udon set is usually safe and quick."
}
```

The answer naturally honors both the peanut allergy (Current Constraint)
and the destination context (Kyoto), without saying "based on what I
remember" or referencing memory explicitly — per the negative instruction
in the system prompt. `reasoning` is recorded for offline analysis; the
post-processor reads only `answer`.

---

## call_6_qa — ⑥ QA Answering (`subset == "supportive"`, variant)

For the same retrieval block, suppose the question is yes/no:

### System (`SYS_QA_SUPPORTIVE`)

```text
You are a helpful assistant answering a yes/no question about a user based on past conversations with them.
The information below describes what is known about the user.

{node_type_desc}

[Traits] contains currently reliable traits.
[Challenged Traits] contains traits that may be outdated, replaced, or contradicted.
If a challenged trait has a "shifted to" entry, prefer the newer shifted-to information over the old trait.
If a challenged trait has conflicting evidence, weigh the conflicting evidence before using the trait.

Current Constraints are high-impact user states that should be honored in the response unless the current question explicitly overrides them.

Task:
- Answer based only on the retrieved user information.
- Answer "yes" only if the retrieved memory clearly supports yes.
- If the retrieved memory is insufficient, irrelevant, or ambiguous, answer "no".

Output requirements:
Respond in JSON with two fields, in this order:
- "reasoning": one short sentence (≤ 1 sentence) stating which retrieved evidence (or its absence) drove the yes/no decision. Write reasoning BEFORE answer.
- "answer": exactly one of "yes" or "no".
```

### User (`QA_PROMPT_SUPPORTIVE`)

```text
[Retrieved Memory]
... (same retrieval block as the opposed example) ...

[Question]
Would the user typically choose a meal-kit subscription that requires weekly menu selections?

Answer the yes/no question based only on the information in memory.
You MUST answer with exactly one of: yes or no.
Output JSON with "reasoning" first (≤ 1 sentence), then "answer".
```

### Expected Output

```json
{
  "reasoning": "The trait and multiple states show the user avoids ongoing manual upkeep; weekly menu selection is the recurring configuration the user resists.",
  "answer": "no"
}
```

---

## Notes on what is NOT shown to the LLM

The following exist in storage but never appear in any LLM-facing prompt:

- Edge attributes `rationale`, `evidence_quote_a`, `evidence_quote_b` —
  used only in `save_snapshot()` and offline analysis.
- Node fields `scope` and `current_decision_impact` are stored on every
  state, but **not** rendered in any node-listing block (Change 10). They
  influence retrieval-time scoring (seed weights, APS membership) only.
- The system-internal term "chunk" — replaced with "recent conversation"
  / "recent two conversations" everywhere it would face the LLM.
- `retrieval_count`, `created_at`, raw `conv_id`/`turn_id`, `session_id` —
  internal book-keeping only. Time is rendered as the human-readable
  `[Xh Ym ago]` form derived from `created_at`.
