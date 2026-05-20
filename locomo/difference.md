# ImplexConv to LoComo Migration Notes

This document summarizes the **global changes required to port an existing
memory-augmented QA-only experiment implementation from ImplexConv to LoComo**.

The goal is not to describe one specific memory module. Instead, this is a
migration guide for anyone who already has an ImplexConv-style implementation
and wants to adapt it to the LoComo setting while preserving the same overall
experiment contract.

In short:

- many **memory-internal ideas can stay**
- the **dataset interface, replay loop, time model, QA task, and metadata
  schema must change**

---

## 1. High-Level Shift

The ImplexConv no-response experiment and the LoComo experiment are both
QA-only, but they differ in what a single example looks like and therefore in
how memory must be constructed.

### ImplexConv no-response

- unit of processing: **session**
- dialogue shape: one flattened sequence of alternating `user` / `assistant`
  turns
- dataset split: `opposed` vs `supportive`
- time model: **virtual**
- QA task: subset-dependent

### LoComo

- unit of processing: **sample**
- dialogue shape: one **multi-session** conversation between two named speakers
- dataset split: **single dataset file**, no opposed/supportive subset routing
- time model: **real session date/time strings**
- QA task: **5-category QA**

This means a straight file-for-file rename is not enough. The migration is
mostly about changing the experiment's **data contract**.

---

## 2. What Can Stay the Same

The following parts usually do **not** need conceptual redesign:

- the idea of replaying dialogue into a memory module first, then running QA
- token accounting by call type
- prompt logging and retrieval logging
- atomic checkpoint/result writes
- batched generation infrastructure
- shared embedding model reuse
- memory snapshots
- module-internal retrieval, summarization, graph evolution, persona tracking,
  or compression logic

What changes is **how the module is driven by the dataset and experiment loop**.

---

## 3. Dataset-Level Changes

### 3.1 Processing unit changes

The first required change is:

- **session-based processing** becomes **sample-based processing**

That affects:

- CLI argument names
- checkpoint keys
- output directory names
- result schema field names
- logging metadata

### 3.2 Dataset file routing changes

ImplexConv no-response typically routes by subset:

```text
dataset/implexconv/ImplexConv_opposed_processed.json
dataset/implexconv/ImplexConv_supportive_processed.json
```

LoComo uses a single canonical file:

```text
dataset/locomo10.json
```

Required migration:

- remove subset-based dataset selection logic
- remove `--subset` from the main run path
- replace subset-specific output naming with sample-range naming

### 3.3 Structural schema changes

ImplexConv gives one flattened conversation list with turn metadata like:

- `conv_id`
- `turn_id`
- `global_turn_id`
- `speaker` in `{user, assistant}`
- `utterance`

LoComo instead gives:

- `sample_id`
- `speaker_a`, `speaker_b`
- multiple `session_N` lists
- per-session `session_N_date_time`
- per-turn `speaker`, `dia_id`, `text`

So the loader must change from:

```text
Session -> flat turn list
```

to:

```text
Sample -> ordered sessions -> ordered turns
```

### 3.4 QA schema changes

ImplexConv QA is subset-oriented:

- `answer`
- `retrieved_conv_ids`
- optional `opposed_implicit_reasoning`

LoComo QA is category-oriented:

- `question`
- `category`
- `answer` for categories 1-4
- `adversarial_answer` for category 5
- `evidence` as a list of `dia_id`s

Recommended loader abstraction:

```python
final_answer = adversarial_answer if category == 5 else answer
```

### 3.5 Image-turn normalization

LoComo may contain `img_url` and `blip_caption` on turns. These need to be
collapsed into plain text during loading:

```python
caption = f"[Image: {turn['blip_caption']}]"
text = f"{caption} {text}" if text else caption
```

This is a new normalization step that does not exist in the ImplexConv path.

---

## 4. Replay Loop Changes

### 4.1 Stop assuming user/assistant turn pairs

In ImplexConv no-response, a common loop shape is:

```text
for each (user_turn, assistant_turn) pair:
    process user turn
    process GT assistant turn
```

That assumption must be removed.

LoComo replay should instead follow:

```text
for each sample:
    for each session in chronological order:
        for each turn in original order:
            store/process the observed turn
```

### 4.2 Remove the GT assistant-response reuse assumption

ImplexConv no-response relies on a **GT agent response rule** because the setup
is built around a conversational agent with explicit assistant turns that would
otherwise need to be generated.

LoComo is different:

- both speakers are just observed participants in a conversation
- there is no special “generated assistant response” role to preserve
- every turn should be treated as observed dialogue input

Required migration:

- remove any logic that distinguishes “user input” from “GT agent response for
  storage”
- remove any code path that re-inserts assistant responses as a special case
- remove any prompt framing that assumes one side is the agent being simulated

### 4.3 Session boundaries now matter

ImplexConv uses one flat session object with internal conversation IDs.
LoComo has explicit session boundaries plus per-session timestamps.

Required migration:

- replay must preserve session order
- modules that maintain context should handle session transitions explicitly
- any summary/finalization logic that previously triggered at virtual day or
  conversation boundaries may need to trigger at session boundaries instead

### 4.4 Phase-1 retrieval is no longer part of the global contract

In ImplexConv no-response, Phase 1 retrieval often exists because the module is
still structured around turn-level conversational context building.

In LoComo, the global contract is simpler:

- store and update memory while replaying the observed conversation
- finalize memory
- answer QA afterward

If a module still wants internal retrieval during Phase 1, that is allowed. But
it is no longer something the experiment should assume is required.

---

## 5. Time Model Changes

This is one of the biggest conceptual changes.

### 5.1 Remove the virtual time model

ImplexConv no-response uses a virtual scheme such as:

- `CONV_IDS_PER_DAY`
- `MINUTES_PER_TURN`

LoComo should instead use the real session-level date/time strings supplied by
the dataset.

Required migration:

- remove the virtual day/time abstraction from the experiment contract
- stop basing boundaries on `conv_id`
- stop deriving elapsed time from `turn_id` alone

### 5.2 Use session timestamps as the temporal anchor

For LoComo, `session_N_date_time` is the authoritative time signal.

If a module needs only coarse time:

- use the session timestamp directly for all turns in that session

If a module needs per-turn ordering within a session:

- derive per-turn timestamps deterministically from
  `session_N_date_time + turn_index`

The exact derivation can vary by module, but it should be:

- deterministic
- documented
- consistent across runs

### 5.3 Boundary logic must be rethought

Any old logic tied to:

- virtual days
- conversation ID rollover
- “every N conversations”

should be replaced by logic tied to one of:

- explicit session boundaries
- actual elapsed time between session timestamps
- module-specific deterministic per-turn offsets within a session

---

## 6. QA Task Changes

### 6.1 Remove subset branching

ImplexConv QA behavior depends on `opposed` vs `supportive`:

- free-form answer generation for one subset
- `yes` / `no` style answering for the other

LoComo does not follow that split.

Required migration:

- remove subset-specific QA schemas
- remove yes/no normalization logic that only exists for supportive QA
- remove subset-specific prompt templates

### 6.2 Add category-aware QA handling

LoComo QA must branch by category:

- category 1: multi-hop
- category 2: temporal
- category 3: open-domain
- category 4: single-hop
- category 5: adversarial

At minimum, migration requires:

- using `category` in QA prompt construction
- adding special handling for temporal questions
- adding special handling for adversarial questions

### 6.3 Add category-5 deterministic choice ordering

For category 5, the model must choose between:

- `adversarial_answer`
- `"Not mentioned in the conversation"`

To keep outputs reproducible in batch mode, this order should be deterministic.

Recommended seed:

```text
f"{sample_id}::{qa_idx}::{question}"
```

### 6.4 Normalize ground truth at loader or runner level

Any old logic that assumes a single `qa.answer` field should be replaced with a
normalized rule:

```text
ground_truth_answer = qa.final_answer
```

where category 5 maps to `adversarial_answer`.

---

## 7. Speaker and Memory Formatting Changes

### 7.1 Replace fixed role assumptions

ImplexConv is built around stable roles:

- `user`
- `assistant`

LoComo uses two named speakers per sample:

- `speaker_a`
- `speaker_b`

Required migration:

- do not hardcode `User` / `Assistant` semantics into the experiment contract
- allow turn formatting to use speaker names from the sample
- treat both speakers as first-class conversation participants

### 7.2 Speaker prefix is now a module decision

Existing ImplexConv code may already store turns with prefixes like:

- `User: ...`
- `Assistant: ...`
- `Speaker user says: ...`

When migrating to LoComo, the experiment contract should not force one format.
The only requirement is that the representation preserve speaker identity well
enough for downstream QA.

---

## 8. CLI, Naming, and Path Changes

### 8.1 CLI argument renaming

Recommended global renames:

| ImplexConv no-response | LoComo |
|---|---|
| `--start-session` | `--start-sample` |
| `--end-session` | `--end-sample` |
| `--subset` | removed |

### 8.2 Output directory naming

ImplexConv output naming is usually subset/session oriented:

```text
{config}_outputs_{model}_{subset}/session_{start}_{end}/
```

LoComo should become sample oriented:

```text
{config}_outputs_{model}/sample_{start}_{end}/
```

### 8.3 File naming

Recommended replacements:

| ImplexConv no-response | LoComo |
|---|---|
| `results_{model}_{subset}_session_{start}_{end}.json` | `results_{model}_sample_{start}_{end}.json` |
| `checkpoint_{model}_{subset}_session_{start}_{end}.json` | `checkpoint_{model}_sample_{start}_{end}.json` |

### 8.4 Config metadata

Remove subset/session-specific metadata and replace it with sample-oriented
metadata.

Recommended examples:

- remove:
  - `subset`
  - `start_session`
  - `end_session`
- add or rename:
  - `sample_range`
  - `temperature_c5`
  - any LoComo-specific retrieval or time settings

---

## 9. Checkpoint and Result Schema Changes

### 9.1 Prefer sample-ID checkpoints

ImplexConv checkpoints often use:

```json
{"completed_session_ids": [0, 1, 2]}
```

For LoComo, prefer:

```json
{"completed_sample_ids": ["conv-1", "conv-2"]}
```

Why this is better:

- sample IDs are the natural dataset key
- it works well with batched execution
- it is safer than index-only tracking if batching changes

### 9.2 Rename result-level identity fields

Recommended replacements:

| ImplexConv no-response | LoComo |
|---|---|
| `session_id` | `sample_id` |
| `session_range` | `sample_range` |

### 9.3 Update QA metadata fields

ImplexConv QA result metadata often traces back to:

- `session_id`
- `conv_id`
- `turn_id`

LoComo should instead trace back to:

- `sample_id`
- `session_id`
- `dia_id`

### 9.4 Replace dataset evidence metadata

ImplexConv QA often exposes:

- `retrieved_conv_ids`

LoComo ground-truth support uses:

- `evidence` as a list of `dia_id`s

Migration requirement:

- copy `qa.evidence` into result records
- do not try to reinterpret old `retrieved_conv_ids` logic as if it were
  equivalent

---

## 10. Retrieval Logging Changes

The logging contract should become sample-aware and LoComo-aware.

### Old-style identifiers

- `session_id`
- `conv_id`
- `turn_id`

### New-style identifiers

- `sample_id`
- `session_id` when relevant
- `dia_id` when relevant

Recommended QA retrieval log shape:

```json
{
  "phase": "qa",
  "sample_id": "conv-26",
  "session_id": null,
  "dia_id": null,
  "query": "...",
  "num_retrieved": 3,
  "retrieved_items": [
    {
      "dia_id": "D1:3",
      "content_preview": "...",
      "score": 0.91
    }
  ],
  "module_specific": {}
}
```

Notes:

- `conv_id` and `turn_id` should disappear from the global logging contract
- modules may still expose richer internal metadata, but LoComo-facing logs
  should be expressed in sample/session/`dia_id` terms

---

## 11. Prompting Changes

### 11.1 Stop framing QA as subset-dependent label prediction

Any prompt logic specific to:

- “opposed” free-form persona reasoning
- “supportive” yes/no classification

should be removed.

### 11.2 Use LoComo category-aware QA prompts

At the global level, prompt behavior should support:

- short phrase answers for categories 1, 3, 4
- approximate date reasoning for category 2
- binary answer choice for category 5

### 11.3 Keep QA separate from dialogue continuation

This remains the same idea as in ImplexConv no-response:

- QA is not a conversational reply turn
- QA prompt framing should make it clear that the model is answering a question
  about the conversation from memory

---

## 12. Batch-Execution Changes

Batching can stay, but the batching axis often needs adjustment.

### ImplexConv no-response tendency

Batching is often built around:

- aligned turn pairs
- user turn stage
- assistant turn stage

### LoComo requirement

Batching must work over:

- samples with multiple sessions
- arbitrary numbers of turns per session
- no special user/assistant reply-generation split

This usually means refactoring batching from:

```text
paired-turn execution
```

to something like:

```text
ordered-turn replay or ordered-session replay
```

The exact batching strategy is free, but it must preserve:

- sample isolation
- per-sample chronological order
- deterministic category-5 choice ordering

---

## 13. Migration Checklist by Subsystem

### Loader

- Replace subset file routing with one LoComo dataset path.
- Replace flat session schema with sample/session/turn schema.
- Add image-caption normalization.
- Add `category`, `evidence`, and `adversarial_answer` handling.
- Add normalized `final_answer`.

### Runner

- Rename session-based concepts to sample-based concepts.
- Replace paired user/assistant replay with full ordered turn replay.
- Preserve session boundaries explicitly.
- Remove subset argument and subset branching.
- Update output path naming and result file naming.

### Time handling

- Remove virtual-time constants from the global contract.
- Use session date strings as the temporal anchor.
- Add deterministic within-session timestamp derivation if needed.

### QA layer

- Remove supportive/opposed prompt branching.
- Add category-aware prompt branching.
- Add category-5 deterministic choice ordering.
- Use normalized ground-truth answer logic.

### Logging and outputs

- Replace `session_id` identity with `sample_id` where appropriate.
- Replace `conv_id` / `turn_id` retrieval metadata with `session_id` / `dia_id`.
- Replace `retrieved_conv_ids` references with LoComo `evidence`.
- Prefer `completed_sample_ids` checkpoints.

---

## 14. Common Migration Pitfalls

1. Keeping the old paired-turn loop and forcing LoComo into a fake
   user/assistant structure.
2. Accidentally preserving subset-specific QA prompts.
3. Forgetting to remove the GT assistant-response rule.
4. Reusing the old virtual time model instead of LoComo session times.
5. Continuing to log `conv_id` / `turn_id` as if LoComo had the same semantics.
6. Treating category 5 as if it used the normal `answer` field.
7. Forgetting to normalize image turns into plain text.
8. Reusing old output/checkpoint naming so mixed datasets become hard to
   distinguish.

---

## 15. Practical Summary

To migrate an existing ImplexConv no-response implementation to LoComo, the
main changes are:

1. replace the dataset loader and identity schema
2. replace the paired session-turn replay loop with multi-session sample replay
3. replace the virtual time model with session-based real timestamps
4. replace subset-specific QA logic with category-specific QA logic
5. replace session/conv/turn metadata with sample/session/`dia_id` metadata
6. keep the memory module internals, token tracking, logging, and batching
   ideas where they still fit

If a memory module is already cleanly separated from the experiment runner, the
conversion is usually a **protocol migration**, not a full rewrite.
