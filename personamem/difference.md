# Difference: ImplexConv → PersonaMem

This document records every concrete change when adapting memory modules
from the ImplexConv experiment protocol to PersonaMem.

- §1–9: A-MEM module (amem)
- §10:   MemoryBank module (memorybank)
- §11:   Theanine module (theanine)
- §12:   LD-Agent module (ldagent)

---

## 1. Dataset & Unit Mapping

| Concept | ImplexConv | PersonaMem |
|---|---|---|
| Experiment unit | Session (`session_id`, int) | Shared Context (`context_index`, int) |
| Conversation group | Conversation (`conv_id`) | Block (delimited by `system` messages) |
| Single utterance | Turn (`turn_id`, local per conv) | Message (local index per block) |
| Evaluation | QA pair (per session) | QA pair (per shared context) |
| Subsets | `opposed` / `supportive` | `32k` / `128k` / `1M` (benchmark size) |

**PersonaMem terminology in brief:**

- **Shared context** — one full conversation history for a persona. The unit of experiment. One persona can have 1–2 shared contexts (topological ordering variants).
- **Block** — a contiguous group of messages sharing the same topic and time period, bounded by a `system` message at the start.
- **Message** — a single `user` or `assistant` utterance within a block. `system` messages are used only as block boundary markers and are never stored in memory.

---

## 2. Data Loading

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Source files | Single JSON per subset | `questions_[SIZE].csv` + `shared_contexts_[SIZE].jsonl` |
| Join key | n/a (flat list) | `shared_context_id` |
| Session count (32k) | varies | 37 |
| QA per session | varies | avg 15.9 (min 5, max 28) |
| Messages per session | varies | avg 169 (user+assistant only) |
| Load sort order | dataset order | QA count descending |

---

## 3. Phase 1 — Memory Construction

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Turn iteration | `get_turn_pairs()` → (user, assistant) pairs | Sequential message list (user/assistant, no strict pairing) |
| Memory content format | `"Speaker {role} says: {utterance}"` | Content as-is (`"User: ..."` / `"Assistant: ..."` already embedded) |
| System messages | n/a | Skipped; used only as block boundary markers |
| Timestamp format | `"{session_id:04d}_{conv_id:04d}_{turn_id:04d}"` | `"{context_index:04d}_{block_idx:04d}_{local_msg_idx:04d}"` |
| Virtual time granularity | `CONV_IDS_PER_DAY=2` | `CONV_IDS_PER_DAY=1` (1 block = 1 virtual day) |
| Batching unit across sessions | turn-index-aligned across sessions | message-index-aligned across sessions |

**Block index** (`block_idx`): increments each time a `system` message is encountered within the shared context.  
**Local message index** (`local_msg_idx`): 0-based index of the message within its block, counting only `user`/`assistant` messages.

---

## 4. Phase 2 — QA Answering

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Answer format | Free-form text (opposed) / `yes` or `no` (supportive) | Multiple-choice letter: `a`, `b`, `c`, or `d` |
| QA prompt | retrieved memory + question | retrieved memory + question + all 4 options |
| Guided JSON schema | `{"answer": string}` or `{"answer": enum[yes,no]}` | `{"answer": enum[a,b,c,d]}` |
| Evaluation metric | Accuracy + F1 (opposed); accuracy (supportive) | Accuracy (multiple-choice) |
| `end_index_in_shared_context` | n/a | Recorded in results as metadata; not used to limit memory during Phase 1 |

---

## 5. CLI Arguments

| Argument | ImplexConv | PersonaMem |
|---|---|---|
| `--subset` | `opposed` or `supportive` | **removed** |
| `--benchmark-size` | n/a | **new**: `32k`, `128k`, or `1M` |
| `--start-session` | session index | context index (into QA-count-sorted list) |
| `--end-session` | session index | context index |
| All other arguments | unchanged | unchanged |

---

## 6. Config Parameters

| Parameter | ImplexConv | PersonaMem |
|---|---|---|
| `DATASET_OPPOSED` / `DATASET_SUPPORTIVE` | ImplexConv JSON paths | **removed** |
| `DATASET_QUESTIONS_*` / `DATASET_CONTEXTS_*` | n/a | **new**: CSV and JSONL paths per size |
| `CONV_IDS_PER_DAY` | `2` | `1` |
| `MINUTES_PER_TURN` | `10` | `10` (unchanged) |

---

## 7. Result Schema

Changes from ImplexConv to PersonaMem (see `personamem_experiment.md` for full schema):

| Field | ImplexConv | PersonaMem |
|---|---|---|
| Top-level key | `session_id` (int) | `context_index` (int) |
| `persona_id` | absent | **added** (int) |
| `shared_context_id` | absent | **added** (str) |
| `config_metadata.subset` | `"opposed"` / `"supportive"` | **renamed** to `benchmark_size`: `"32k"` / `"128k"` / `"1M"` |
| `qa_results[i].generated_answer` | free text or `yes`/`no` | `a`, `b`, `c`, or `d` |
| `qa_results[i].ground_truth_answer` | free text or `yes`/`no` | `a`, `b`, `c`, or `d` |
| `qa_results[i].question_type` | absent | **added** |
| `qa_results[i].topic` | absent | **added** |
| `qa_results[i].all_options` | absent | **added** (list of 4 strings) |
| `qa_results[i].end_index_in_shared_context` | absent | **added** (int, metadata) |
| `retrieved_memories[j].session_id` | int | **renamed** to `context_index` |
| `retrieved_memories[j].conv_id` | int | **renamed** to `block_idx` |
| `retrieved_memories[j].turn_id` | int | **renamed** to `local_msg_idx` |

---

## 8. Output Directory Naming

```
# ImplexConv
{config}_outputs_{model}_{subset}/          e.g. config_0_outputs_Qwen3-1.7B_opposed/

# PersonaMem
{config}_outputs_{model}_{benchmark_size}/  e.g. config_0_outputs_Qwen3-1.7B_32k/
```

File names inside follow the same pattern with `{benchmark_size}` replacing `{subset}`.

---

## 9. What Is Unchanged

- `memory_layer.py` — no changes required
- `agent.py` — QA prompts updated; memory/retrieval interface unchanged
- Checkpoint format (`completed_session_ids` set-based)
- Token tracking structure (`call_2_note_construction`, `call_3_evolution`, `call_4_qa`)
- Evolution statistics
- Memory snapshot format
- Retrieval log format
- Prompt log format
- LLM client infrastructure (`llm_module/`)
- `MINUTES_PER_TURN`, `RETRIEVE_K`, `EVOLUTION_THRESHOLD`, `BATCH_SIZE`, `QA_BATCH_SIZE`, `MAX_TOKENS`, `JSON_RETRY`

---

## 10. MemoryBank Module

This section records the changes when adapting the `memorybank/` module from
the ImplexConv experiment protocol to PersonaMem. Mapping convention follows
amem: internal MemoryBank field names (`session_id`, `conv_id`, `turn_id`,
`CONVS_PER_DAY`) are kept; the runner injects PersonaMem identifiers
(`context_index`, `block_idx`, `local_msg_idx`) into those slots and remaps
result metadata at the boundary.

### 10.1 Dataset & Unit Mapping

Same as §1. The `block` boundary (PersonaMem `system` message) replaces the
ImplexConv `conv_id` boundary as the trigger for forgetting curve and
hierarchical summarization.

### 10.2 Phase 1 — Memory Construction

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Iteration | `(user_turn, assistant_turn)` pairs from `get_turn_pairs()` | Sequential `user`/`assistant` messages (no pairing) |
| Memory entry granularity | One snippet per `(user, assistant)` pair, formatted as `[\|User\|]: ... [\|AI\|]: ...` | One snippet per individual message (option (A)); content stored as-is (already includes `User: ` / `Assistant: ` prefix) |
| `system` messages | n/a | Skipped; used only as block boundary markers |
| Boundary detection | `prev_conv_id != current_conv_id` | `prev_block_idx != current_block_idx` (read from `msg.block_idx`) |
| Forgetting trigger | At each conv_id boundary | At each block boundary |
| Daily-summary trigger | `current_conv_id % CONVS_PER_DAY == 0` | `current_block_idx % CONVS_PER_DAY == 0` (with `CONVS_PER_DAY=1` → every block boundary) |
| Global-summary trigger | `current_conv_id % GLOBAL_SUMMARY_INTERVAL == 0` | `current_block_idx % GLOBAL_SUMMARY_INTERVAL == 0` (still 10; rarely fires before Phase 1 end given ~5 blocks/context) |
| Phase-1-end global synthesis | Always run | Always run (unchanged) |
| Timestamp format | `"{session_id:04d}_{conv_id:04d}_{turn_id:04d}"` | `"{context_index:04d}_{block_idx:04d}_{local_msg_idx:04d}"` |
| Phase 1 retrieve-on-store | Not performed (store only) | Not performed (unchanged) |

### 10.3 Phase 2 — QA Answering

QA prompt structure is rebuilt for PersonaMem multiple-choice (option (C) of
the migration spec): the SiliconFriend persona prompt and the original
`[|User|]/[|AI|]:`-only conversation framing are removed.

```
[User Portrait]      ← global user portrait (if non-empty)
[Memory]             ← retrieved snippets + daily event summaries
[Memory Dates]       ← e.g. "Day 1, Day 3"
[Recent History]     ← last HISTORY_CONV_WINDOW blocks of user/assistant messages,
                       relabelled as [|User|]: / [|AI|]: (option (B))
Question: ...
Options:
(a) ...
(b) ...
(c) ...
(d) ...
```

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Answer format | Free-form (opposed) / `yes` or `no` (supportive) | Multiple-choice letter `a`/`b`/`c`/`d` |
| Guided JSON schema | `{"answer": string}` or `{"answer": enum[yes,no]}` | `{"answer": enum[a,b,c,d]}` |
| Subset-specific suffix | `OPPOSED` / `SUPPORTIVE` suffix templates | Removed; one unified template |
| `subset` argument | Required | Removed |
| Fallback on parse failure | `""` (opposed) / `"unknown"` (supportive) | `"unknown"` |

### 10.4 CLI Arguments

| Argument | ImplexConv | PersonaMem |
|---|---|---|
| `--subset` | `opposed` / `supportive` | **removed** |
| `--benchmark-size` | n/a | **new**: `32k` / `128k` / `1M` |
| `--start-session` / `--end-session` | session indices | context indices (into QA-count-sorted list) |
| All other arguments | unchanged | unchanged |

### 10.5 Config Parameters

| Parameter | ImplexConv | PersonaMem |
|---|---|---|
| `DATASET_OPPOSED` / `DATASET_SUPPORTIVE` | ImplexConv JSON paths | **removed** |
| `DATASET_QUESTIONS_*` / `DATASET_CONTEXTS_*` | n/a | **new**: per benchmark size |
| `BENCHMARK_SIZES` | n/a | **new**: `["32k", "128k", "1M"]` |
| `CONVS_PER_DAY` | `2` | `1` (1 block = 1 virtual day) |
| `HISTORY_CONV_WINDOW` | `2` (conv_ids) | `1` (blocks) |
| `GLOBAL_SUMMARY_INTERVAL` | `10` | `10` (kept; Phase-1-end synthesis covers practical use) |
| `RETRIEVE_K`, `FORGETTING_DIVISOR`, `MINUTES_PER_TURN`, `SUMMARIZE_TEMPERATURE`, `SUMMARIZE_MAX_TOKENS`, `BATCH_SIZE`, `QA_BATCH_SIZE`, `MAX_TOKENS`, `TEMPERATURE`, `JSON_RETRY` | unchanged | unchanged |

### 10.6 Result Schema

| Field | ImplexConv | PersonaMem |
|---|---|---|
| Top-level key | `session_id` (int) | `context_index` (int) |
| `persona_id` | absent | **added** (int) |
| `shared_context_id` | absent | **added** (str) |
| `config_metadata.subset` | `opposed` / `supportive` | **renamed** to `benchmark_size` |
| `qa_results[i].generated_answer` | free text or `yes`/`no` | `a` / `b` / `c` / `d` / `unknown` |
| `qa_results[i].ground_truth_answer` | free text or `yes`/`no` | `a` / `b` / `c` / `d` |
| `qa_results[i].question_type` | absent | **added** |
| `qa_results[i].topic` | absent | **added** |
| `qa_results[i].all_options` | absent | **added** (4 strings) |
| `qa_results[i].end_index_in_shared_context` | absent | **added** (int, metadata) |
| `retrieved_memories[j].session_id` | int | **renamed** to `context_index` |
| `retrieved_memories[j].conv_id` | int | **renamed** to `block_idx` |
| `retrieved_memories[j].turn_id` | int | **renamed** to `local_msg_idx` |
| `retrieved_memories[j].source_summary_conv_id` | int (when `memory_subtype="daily_summary"`) | **renamed** to `source_summary_block_idx` |
| `retrieved_memories[j].memory_subtype` / `source_label` / `score` / `strength` | unchanged | unchanged |
| `phase1_statistics` | `{forgetting_events_count, memories_forgotten}` | **unchanged** |
| `token_statistics` 5 call types | `call_2_daily_event`, `call_3_daily_personality`, `call_4_global_event`, `call_5_global_personality`, `call_6_qa` | **unchanged** |

### 10.7 Retrieval Log

Top-level keys renamed:

| Field | ImplexConv | PersonaMem |
|---|---|---|
| `session_id` / `conv_id` / `turn_id` | int | renamed to `context_index` / `block_idx` / `local_msg_idx` |
| `module_specific.current_conv_id` | int | renamed to `current_block_idx` |
| `module_specific.module` | `"memorybank"` | unchanged |
| Other `module_specific.*` fields | unchanged | unchanged |

### 10.8 Output Directory Naming

```
# ImplexConv
{config}_outputs_{model}_{subset}/          e.g. config_0_outputs_Qwen3-1.7B_opposed/

# PersonaMem
{config}_outputs_{model}_{benchmark_size}/  e.g. config_0_outputs_Qwen3-1.7B_32k/
```

### 10.9 What Is Unchanged

- `retriever.py` — byte-identical
- `memory_bank.py` — only the module docstring; all logic, variable names, and field keys preserved
- Forgetting curve formula, summarization prompts, hierarchical synthesis logic
- Memory snapshot file layout (`entries.json`, `embeddings.npy`, `corpus.json`, `summaries.json`, `metadata.json`)
- LLM call logger directory layout (5 call-type subfolders)
- Checkpoint format (`completed_session_ids` set-based; backward-compatible)
- LLM client infrastructure (`llm_module/`)

### 10.10 Memory Content Wrapping

`MemoryBankSystem.add_memory` retains its internal wrapping
`"Conversation content on Day {N}: {content}"`. With PersonaMem messages stored
as-is (option (A)), each entry becomes e.g.
`"Conversation content on Day 1: User: Hi there! ..."` — applied uniformly to
both user and assistant messages.

---

## 11. Theanine Module

This section records the changes when adapting the `theanine/` module from the
ImplexConv experiment protocol to PersonaMem. Mapping convention follows amem
and memorybank: internal Theanine field names (`session_id`, `conv_id`,
`turn_id`, `c{conv_id}-m{idx}` node keys, `CONV_IDS_PER_DAY`,
`FINALIZE_EVERY_N_CONVS`) are kept; the runner injects PersonaMem identifiers
(`context_index`, `block_idx`, `local_msg_idx`) into those slots and remaps
result metadata at the boundary.

### 11.1 Dataset & Unit Mapping

Same as §1. The `block` boundary (PersonaMem `system` message) replaces the
ImplexConv `conv_id` boundary as the trigger for Theanine `finalize_conv`
(summarization + relation extraction).

### 11.2 Phase 1 — Memory Construction

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Iteration | `(user_turn, assistant_turn)` pairs from `get_turn_pairs()` | Sequential `user`/`assistant` messages (no pairing) |
| `system` messages | n/a | Skipped at load time; never enter memory or dialogue |
| Boundary detection | `user_turn.conv_id != prev_conv_id` + `pending_count >= FINALIZE_EVERY_N_CONVS` | `msg.block_idx != prev_block_idx`; finalize is fired immediately because `FINALIZE_EVERY_N_CONVS=1` |
| `pending_count` counter | Used | Removed (no longer needed; structural variables retained in config for symmetry) |
| `current_dialogue` reset | Each virtual day boundary | Each block boundary (Q1=A, Q16); Q24=A: if the final block has zero `user`/`assistant` messages, fall back to the previous block's dialogue |
| `finalize_messages` content | GT user + assistant turns within a conv | All `user`/`assistant` messages within a block, content stored as-is (no `[|User|]/[|AI|]` rewriting; Q7=A) |
| `source_conv_dialogue` (per node) | Full conv dialogue | Full block dialogue (joined message contents) |
| `global_msg_idx` | n/a | Computed in the runner (Q23=C): each user/assistant message is assigned a 0-based index in iteration order, no change to `load_dataset.py` |
| Node key format | `c{conv_id}-m{idx}` | Unchanged; runner passes `block_idx` into the `conv_id` slot, so node keys become e.g. `c0-m1` for the first block's first fact |

### 11.3 Phase 2 — QA Answering

QA prompt structure follows option (A): Theanine's original skeleton with a
multiple-choice suffix added.

```
Memory:
1: <refined timeline 1>
2: <refined timeline 2>
...

Current conversation:
<final block dialogue, content as-is>

Question:
<...>

Options:
(a) ...
(b) ...
(c) ...
(d) ...

Answer with a/b/c/d (JSON: {"answer": "..."}).
```

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Answer format | Free-form (opposed) / `yes`/`no` (supportive) | Multiple-choice letter `a`/`b`/`c`/`d` |
| Guided JSON schema | `{"answer": string}` or `{"answer": enum[yes,no]}` | `{"answer": enum[a,b,c,d]}` |
| Subset-specific QA prompts | `QA_PROMPT_OPPOSED` / `QA_PROMPT_SUPPORTIVE` | `QA_PROMPT_MULTICHOICE` (single template) |
| `subset` parameter | Required by Generator | Removed |
| `options` parameter | n/a | Required by Generator |
| Refinement step | Kept (Q20) | Kept; refined timeline texts are numbered `1: ... 2: ...` and inserted into the Memory section |
| `current_dialogue` injection in refinement and QA | Kept | Kept (Q1=A) |
| Parse-failure fallback | `""` (opposed) / `"unknown"` (supportive) | `"unknown"` (Q21=A+C — batch failure → sequential retry → `"unknown"`) |

### 11.4 CLI Arguments

| Argument | ImplexConv | PersonaMem |
|---|---|---|
| `--subset` | `opposed` / `supportive` | **removed** |
| `--benchmark-size` | n/a | **new**: `32k` / `128k` / `1M` |
| `--start-session` / `--end-session` | session indices | context indices (into QA-count-sorted list) |
| All other arguments | unchanged | unchanged |

### 11.5 Config Parameters

| Parameter | ImplexConv | PersonaMem |
|---|---|---|
| `DATASET_OPPOSED` / `DATASET_SUPPORTIVE` | ImplexConv JSON paths | **removed** |
| `DATASET_QUESTIONS_*` / `DATASET_CONTEXTS_*`, `BENCHMARK_SIZES` | n/a | **new** |
| `CONV_IDS_PER_DAY` | `2` | `1` (1 block = 1 virtual day) |
| `FINALIZE_EVERY_N_CONVS` | `= CONV_IDS_PER_DAY = 2` | `= CONV_IDS_PER_DAY = 1` (every block boundary triggers finalize) |
| `LINKING_TOP_J`, `RETRIEVE_TOP_K`, `TIMELINE_SAMPLE_N` | `3, 5, 1` | unchanged |
| `MAX_TOKENS`, `SUMMARIZE_MAX_TOKENS`, `JSON_RETRY` | `2048, 1500, 3` | unchanged (Q19) |
| `BATCH_SIZE`, `SUMMARIZE_BATCH_SIZE`, `RELATION_BATCH_SIZE`, `REFINE_BATCH_SIZE`, `QA_BATCH_SIZE`, `EMBEDDING_MODEL`, `MINUTES_PER_TURN`, `TEMPERATURE` | unchanged | unchanged |

### 11.6 Result Schema

| Field | ImplexConv | PersonaMem |
|---|---|---|
| Top-level key | `session_id` (int) | `context_index` (int) |
| `persona_id` | absent | **added** (int) |
| `shared_context_id` | absent | **added** (str) |
| `config_metadata.subset` | `opposed` / `supportive` | **renamed** to `benchmark_size`; also adds `timeline_sample_n` |
| `qa_results[i].generated_answer` | free text or `yes`/`no` | `a` / `b` / `c` / `d` / `unknown` |
| `qa_results[i].ground_truth_answer` | free text or `yes`/`no` | `a` / `b` / `c` / `d` |
| `qa_results[i].question_type` | absent | **added** |
| `qa_results[i].topic` | absent | **added** |
| `qa_results[i].all_options` | absent | **added** (4 strings) |
| `qa_results[i].end_index_in_shared_context` | absent | **added** (int, metadata) |
| `qa_results[i].retrieved_memories[j].session_id` | int | **renamed** to `context_index` |
| `qa_results[i].retrieved_memories[j].conv_id` | int | **renamed** to `block_idx` |
| `qa_results[i].retrieved_memories[j].turn_id` (= `turn_id_start`) | int | **renamed** to `local_msg_idx` |
| `token_statistics` (5 buckets + `qa_input` + totals) | unchanged | **unchanged** (Q22) |

### 11.7 Retrieval Log

Top-level keys renamed:

| Field | ImplexConv | PersonaMem |
|---|---|---|
| `session_id` / `conv_id` / `turn_id` | int | renamed to `context_index` / `block_idx` / `local_msg_idx` |
| `module_specific.module` | `"theanine"` | unchanged |
| `module_specific.use_timelines` / `timeline_info` / `num_paths_used` | unchanged | unchanged |
| Memory-graph internal node IDs (`c{block_idx}-m{idx}`) inside paths | unchanged | unchanged (block_idx flows into the `conv_id` slot) |

### 11.8 Output Directory Naming

```
# ImplexConv
{config}_outputs_{model}_{subset}/          e.g. config_0_outputs_Qwen3-1.7B_opposed/

# PersonaMem
{config}_outputs_{model}_{benchmark_size}/  e.g. config_0_outputs_Qwen3-1.7B_32k/
```

### 11.9 What Is Unchanged

- `memory_graph.py` — only the module docstring; all logic, variable names, dataclass field names, and node-key format preserved
- `timeline.py` — byte-identical
- `theanine_module.py` — byte-identical
- Summarization / relation extraction / refinement prompt templates and JSON schemas
- Memory snapshot file layout (`nodes.json`, `embeddings.npy`, `embedding_ids.json`)
- LLM call logger directory layout (`call_2_refinement`, `call_3_summarization`, `call_4_relation`, `call_5_qa`)
- Checkpoint format (`completed_session_ids` set-based; backward-compatible)
- LLM client infrastructure (`llm_module/`)

---

## 12. LD-Agent Module

This section records the changes when adapting the `ldagent/` module from the
ImplexConv experiment protocol to PersonaMem. Mapping convention follows amem
/ memorybank / theanine: internal LD-Agent field names (`session_id`,
`conv_id`, `turn_id`, `CONV_IDS_PER_DAY`, `FINALIZE_EVERY_N_CONVS`) are kept;
the runner injects PersonaMem identifiers (`context_index`, `block_idx`,
`local_msg_idx`) into those slots and remaps result metadata at the boundary.

### 12.1 Dataset & Unit Mapping

Same as §1. PersonaMem `block_idx` is fed into the `conv_id` slot of LD-Agent
(used both by the forgetting curve / virtual-time decay and by the STM
boundary trigger).

### 12.2 Phase 1 — Memory Construction

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Iteration unit | `(user_turn, assistant_turn)` pairs from `Session.get_turn_pairs()` | `(user_msg, assistant_msg)` pairs from `get_message_pairs()` (Q1=A) |
| Pairing scope | within a session | **crosses block boundaries** (Q8=A): `user → assistant` forms a pair regardless of their `block_idx`; an unpaired user becomes `(user, None)`; lone assistant is dropped |
| Sentence input to persona prompt | raw `Turn.utterance` | `strip_speaker_prefix(msg.content)` — `"User: "` / `"Assistant: "` prefix removed (Q2) |
| STM stored format | `f"{usr_name}: {utterance}"` (LD-Agent re-wraps internally) | unchanged: LD-Agent re-wraps the prefix-stripped utterance, so STM ends up as `"User: ..."` / `"Assistant: ..."` matching the PersonaMem format |
| `usr_name` / `agent_name` | `"User"` / `"Agent"` | `"User"` / `"Assistant"` (Q3=A) — matches PersonaMem prefixes |
| Unpaired user (Q5=A) | n/a | user persona update + STM user append; agent persona / STM agent append are **skipped** |
| `compute_virtual_seconds` arguments (Q9=A) | `(user_turn.conv_id, user_turn.turn_id)` | `(user_msg.block_idx, user_msg.local_msg_idx)` — both members of a pair share the same virtual_seconds, mirroring the ImplexConv per-pair time stamping |
| `current_conv_id` passed to memory_bank | `user_turn.conv_id` | `user_msg.block_idx` |
| `current_session_id` passed to memory_bank | `session.session_id` | `context.context_index` |

### 12.3 Phase 2 — QA Answering

QA prompt uses the LD-Agent 4-section structure with a multiple-choice suffix
(Q4=A):

```
[SYSTEM]
You are a helpful assistant answering a multiple-choice question about a user
based on recorded conversation memories.
Your personal traits as the assistant: {agent_traits}.

[USER]
<CONTEXT>
Recent conversation turns:
{stm_context}

<MEMORIES>
The following are conversation memories about the user:
{memories}

<USER_TRAITS>
User characteristics:
{user_traits}

<QUESTION>
Answer the following question based on the context, memories, and traits above.
Question: ...

Options:
(a) ...
(b) ...
(c) ...
(d) ...

Choose the single best answer (a, b, c, or d) and respond in JSON format with
key "answer" containing only the letter.
```

| Item | ImplexConv | PersonaMem |
|---|---|---|
| Answer format | Free-form (opposed) / `yes`/`no` (supportive) | Multiple-choice letter `a`/`b`/`c`/`d` |
| Guided JSON schema | `{"answer": string}` or `{"answer": enum[yes,no]}` | `{"answer": enum[a,b,c,d]}` |
| `subset` parameter | Required by Generator | Removed |
| `options` parameter | n/a | Required by Generator |
| `flush_to_ltm` between Phase 1 and Phase 2 (Q7=B) | Called: STM → final LTM entry, STM retained | **NOT called.** STM is retained as the QA `<CONTEXT>` exactly as left by Phase 1 (= the last block's messages, since `FINALIZE_EVERY_N_CONVS=1` clears STM at every block boundary). Avoids the empty-STM-after-flush concern requested by user |
| Parse-failure fallback | `""` (opposed) / `"unknown"` (supportive) | `"unknown"` for any non-`a/b/c/d` letter; sequential JSON-retry then unknown (matches memorybank / theanine pattern) |

### 12.4 CLI Arguments

| Argument | ImplexConv | PersonaMem |
|---|---|---|
| `--subset` | `opposed` / `supportive` | **removed** |
| `--benchmark-size` | n/a | **new**: `32k` / `128k` / `1M` |
| `--start-session` / `--end-session` | session indices | context indices (into QA-count-sorted list) |
| All other arguments | unchanged | unchanged |

### 12.5 Config Parameters

| Parameter | ImplexConv | PersonaMem |
|---|---|---|
| `DATASET_OPPOSED` / `DATASET_SUPPORTIVE` | ImplexConv JSON paths | **removed** |
| `DATASET_QUESTIONS_*` / `DATASET_CONTEXTS_*`, `BENCHMARK_SIZES` | n/a | **new** |
| `CONV_IDS_PER_DAY` | `2` | `1` (1 block = 1 virtual day) |
| `FINALIZE_EVERY_N_CONVS` | `= CONV_IDS_PER_DAY = 2` | `= CONV_IDS_PER_DAY = 1` (STM cleared at every block boundary) |
| `USR_NAME` / `AGENT_NAME` | `"User"` / `"Agent"` | `"User"` / `"Assistant"` (Q3=A) |
| `RELEVANCE_MEMORY_NUMBER`, `RETRIEVE_K`, `DIST_THRESHOLD`, `ORI_MEM_QUERY`, `DECAY_TEMP`, `MAX_USER_PERSONAS`, `MAX_AGENT_PERSONAS`, `MAX_TOKENS`, `TEMPERATURE`, `JSON_RETRY`, `BATCH_SIZE`, `QA_BATCH_SIZE`, `MINUTES_PER_TURN` | unchanged | unchanged |

### 12.6 Result Schema

| Field | ImplexConv | PersonaMem |
|---|---|---|
| Top-level key | `session_id` (int) | `context_index` (int) |
| `persona_id` | absent | **added** (int) |
| `shared_context_id` | absent | **added** (str) |
| `config_metadata.subset` | `opposed` / `supportive` | **renamed** to `benchmark_size`; also adds `usr_name`, `agent_name` |
| `qa_results[i].generated_answer` | free text or `yes`/`no` | `a` / `b` / `c` / `d` / `unknown` |
| `qa_results[i].ground_truth_answer` | free text or `yes`/`no` | `a` / `b` / `c` / `d` |
| `qa_results[i].question_type` | absent | **added** |
| `qa_results[i].topic` | absent | **added** |
| `qa_results[i].all_options` | absent | **added** (4 strings) |
| `qa_results[i].end_index_in_shared_context` | absent | **added** (int, metadata) |
| `qa_results[i].retrieved_memories[j].session_id` | int | **renamed** to `context_index` |
| `qa_results[i].retrieved_memories[j].conv_id` | int | **renamed** to `block_idx` |
| `qa_results[i].retrieved_memories[j].virtual_seconds` / `score` | unchanged | unchanged |
| `token_statistics` (4 buckets `call_2_user_persona` / `call_3_agent_persona` / `call_4_summarization` / `call_5_qa` + totals) | unchanged | **unchanged**; `call_4_summarization` now records boundary calls only (flush removed, Q7=B) |

### 12.7 Retrieval Log

| Field | ImplexConv | PersonaMem |
|---|---|---|
| `session_id` / `conv_id` / `turn_id` | int | renamed to `context_index` / `block_idx` / `local_msg_idx` |
| `retrieved_items[j].source_turn.session_id` / `conv_id` | int | renamed to `context_index` / `block_idx` |
| `retrieved_items[j].source_turn.virtual_seconds` | unchanged | unchanged |

### 12.8 Output Directory Naming

```
# ImplexConv
{config}_outputs_{model}_{subset}/          e.g. config_0_outputs_Qwen3-1.7B_opposed/

# PersonaMem
{config}_outputs_{model}_{benchmark_size}/  e.g. config_0_outputs_Qwen3-1.7B_32k/
```

### 12.9 What Is Unchanged

- `event_memory.py` — only the module docstring; all logic, dataclass fields, retrieval / forgetting-curve scoring, and snapshot file layout preserved
- `personas.py` — byte-identical
- `ldagent_module.py` — byte-identical (its `process_turn` is unused by the batched runner but kept for completeness)
- LTM noun-overlap × time-decay scoring formula
- Persona extraction prompts and JSON schemas
- STM finalization policy (LD-Agent boundary clear is unchanged; only the post-Phase-1 `flush_to_ltm` is removed at the runner level, see Q7=B)
- Memory snapshot file layout (`short_term_memory.json`, `long_term_memory.json`, `ltm_embeddings.npy`, `memory_state.json`, `personas.json`)
- LLM call logger directory layout (`call_2_user_persona`, `call_3_agent_persona`, `call_4_summarization`, `call_5_qa`)
- Checkpoint format (`completed_session_ids` set-based; backward-compatible)
- LLM client infrastructure (`llm_module/`)

### 12.10 Persona Call Frequency Note

Per the original LD-Agent design, persona extraction calls fire on **every
pair**: 1 user-persona + 1 agent-persona LLM call per pair. With PersonaMem's
~85 pairs/context (32k variant), this yields ~170 persona calls/context and
~6,290 calls across the 32k benchmark (37 contexts). The 128k / 1M benchmarks
scale up proportionally. This frequency is kept unchanged from the original
LD-Agent design (Q6=A) for cross-module comparability.
