# Memory Corruption Experiment

## 1. Overview

This experiment loads `memory_snapshot` files produced by Phase 1, randomly deletes a specified fraction (`corrupt_rate`) of memory items, then runs Phase 2 QA on the degraded memory. The goal is to quantify how memory loss affects QA accuracy.

- Corruption = uniform random deletion of memory items, `seed=42` fixed
- Embedding vectors and metadata files are preserved; only JSON-based memory items are deleted
- `run_experiment.py` in each module is not modified; a standalone script `run_corruption.py` is written instead

---

## 2. Experiment Flow

```
[Existing experiment output]
  {module}/config_N_outputs_{model}_{subset}/
    session_{range}/memory_snapshots/session_{id}/
                                                    ↓
                              [run_corruption.py]
                              1. load snapshot files
                              2. corrupt: randomly delete X% of memory items (seed=42)
                              3. rebuild in-memory state from corrupted data
                              4. run Phase 2 QA (reuse existing module Phase 2 logic)
                              5. write results
                                                    ↓
[Output]
  {module}/corrupt{rate_pct}_config_N_outputs_{model}_{subset}/
    session_{range}/
      results_session_{id}.json
      retrieval_logs/
      (corrupted_snapshots/)   ← optional
```

Phase 1 is not re-run. Snapshots from completed experiments are used directly.

---

## 3. Directory Structure

### Input (existing snapshots)
```
{module}/config_N_outputs_{model}_{subset}/
  session_{start}_{end}/
    memory_snapshots/
      session_0/
      session_1/
      ...
```

### Output
```
{module}/corrupt{rate_pct}_config_N_outputs_{model}_{subset}/
  session_{start}_{end}/
    results_session_0.json
    results_session_1.json
    ...
    retrieval_logs/
      session_0_retrieval_log.jsonl
      ...
```

`rate_pct` is an integer. Example: `--corrupt-rate 0.3` → directory prefix `corrupt30`.

The result JSON format is identical to the existing `results_session_{id}.json`, with one additional field inside `config_metadata`:
```json
"corruption": {
  "corrupt_rate": 0.3,
  "random_seed": 42,
  "memories_before": 120,
  "memories_after": 84,
  "memories_deleted": 36
}
```

---

## 4. CLI Interface

`--snapshot-base` and `--output-dir` are automatically derived from `--model`, `--config`, `--subset`, and `--corrupt-rate`, so the interface mirrors `run_experiment.py`.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python run_corruption.py \
  --start-session 0 --end-session 499 \
  --subset opposed \
  --model Qwen/Qwen3-1.7B \
  --corrupt-rate 0.3 \
  --tensor-parallel 1 --gpu-memory 0.34 --max-model-len 30000 \
  --batch-size 25 --config config_0 \
  > nohup/nohup_corrupt30_1.7b_0_499.out 2>&1 &
```

Automatically derived paths (example with `--corrupt-rate 0.3`, `config_0`, `Qwen3-1.7B`, `opposed`):
```
snapshot_base → {module}/config_0_outputs_Qwen3-1.7B_opposed/
output_dir    → {module}/corrupt30_config_0_outputs_Qwen3-1.7B_opposed/
```

| Argument | Description |
|---|---|
| `--corrupt-rate` | Fraction to delete (0.0–1.0) |
| `--start-session` | Start session ID |
| `--end-session` | End session ID, inclusive |
| `--subset` | opposed / supportive |
| `--model` | LLM model path |
| `--tensor-parallel` | Tensor parallel size |
| `--gpu-memory` | GPU memory utilization (0.0–1.0) |
| `--max-model-len` | Max model context length (optional) |
| `--batch-size` | Sessions per batch |
| `--config` | Config module name (default: config_0) |
| `--save-corrupted-snapshot` | Save corrupted snapshot to output directory (optional) |

---

## 5. Per-Module Corruption Logic

### Common rules
- `n_delete = floor(N * corrupt_rate)`
- Selecting items to delete: `random.Random(42).sample(population, n_delete)`
- List-based: delete by index → remove same indices from embeddings
- Dict-based: sort keys first, then sample → guarantees deterministic order
- `embeddings.npy` is always updated in sync with the memory items it is aligned to

---

### 5.1 amem

**Snapshot files:**
```
memories.json       ← Dict[uuid_str, MemoryNote]          ← corrupt target
retriever.json      ← {"model_name": str, "corpus": [...]}
embeddings.npy      ← shape (N, dim), rows aligned with corpus
metadata.json       ← preserve (overwrite num_memories only)
```

**Corruption steps:**
1. Load `memories.json` → sort keys → sample 30% → delete those keys
2. Record the original insertion order to identify `deleted_idx` (position set)
3. Remove `deleted_idx` rows from `retriever.json["corpus"]`
4. Remove `deleted_idx` rows from `embeddings.npy` via `np.delete`
5. Fix `links` field (list of int indices into `list(memories.values())`) in each surviving note:
   - Drop references to deleted indices
   - Remap surviving indices to their new positions (build old→new mapping dict first)

**Restoring in-memory state:**
```python
memories = {k: MemoryNote(**v) for k, v in memories_json.items()}
retriever.corpus = retriever_json["corpus"]
retriever.embeddings = np.load("embeddings.npy")
```
`MemoryNote` accepts `**kwargs` in its constructor. Assign directly to `AgenticMemorySystem.memories` and `.retriever`.

---

### 5.2 memorybank

**Snapshot files:**
```
entries.json        ← List[MemoryEntry]                    ← corrupt target
corpus.json         ← List[str], 1:1 aligned with entries  ← sync with entries deletion
embeddings.npy      ← shape (N, dim)                       ← sync with entries deletion
summaries.json      ← {daily_event_summaries, daily_personality_summaries, ...}
metadata.json       ← preserve
```

**`summaries.json` structure:**
```json
{
  "daily_event_summaries":       {"0": "...", "2": "..."},   ← corrupt target (key = str(conv_id))
  "daily_personality_summaries": {"0": "...", "2": "..."},   ← corrupt target
  "global_event_summary":        "...",                      ← preserve
  "global_user_portrait":        "..."                       ← preserve
}
```

**Corruption steps:**
1. `entries.json`: delete 30% of indices → remove same indices from `corpus.json` and `embeddings.npy`
2. `daily_event_summaries`: sort keys → sample 30% → delete those keys
3. `daily_personality_summaries`: same (independent sample)
4. `global_event_summary`, `global_user_portrait`: no change

**Restoring in-memory state:**
```python
mb.entries = [MemoryEntry(**d) for d in entries_json]
mb.retriever.corpus = corpus_json
mb.retriever.embeddings = np.load("embeddings.npy")
mb.daily_event_summaries = {int(k): v for k, v in summaries["daily_event_summaries"].items()}
mb.daily_personality_summaries = {int(k): v for k, v in summaries["daily_personality_summaries"].items()}
mb.global_event_summary = summaries["global_event_summary"]
mb.global_user_portrait = summaries["global_user_portrait"]
# Reconstruct daily_summary_entries and daily_summary_retriever
mb._rebuild_daily_summary_index()
```

---

### 5.3 theanine

**Snapshot files:**
```
nodes.json          ← Dict[node_id_str, NodeDict]          ← corrupt target
embedding_ids.json  ← List[str], aligned with embeddings   ← remove deleted node_ids
embeddings.npy      ← shape (N, dim)                       ← remove same rows
```

**`links` field in each node:**
```json
"links": {"c1-m0": "temporal_successor", "c0-m2": "semantic_similar"}
```
A dict keyed by node_id string.

**Corruption steps:**
1. Sort `nodes.json` keys → sample 30% → build `deleted_ids` set → delete those nodes
2. From each surviving node's `links` dict, remove any key in `deleted_ids`
3. Remove `deleted_ids` entries from `embedding_ids.json` (preserve order)
4. Remove the corresponding rows from `embeddings.npy`

**Restoring in-memory state:**
```python
emb_map = {nid: embeddings[i] for i, nid in enumerate(embedding_ids)}
graph.nodes = {
    nid: MemoryNode(**{**d, "embedding": emb_map[nid]})
    for nid, d in nodes_json.items()
}
```
`MemoryNode.embedding` is not stored in `nodes.json`; inject it from `emb_map`.

---

### 5.4 gmem4

**Snapshot files:**
```
module_state.json          ← preserve (load as-is)
graph/graph.json           ← nodes, creation_order, src_out, evid_out  ← corrupt target
graph/graph_embeddings.npy ← shape (N, dim), rows aligned with creation_order ← sync
graph/graph_metadata.json  ← preserve
cache/context_cache.json   ← preserve (load as-is)
```

**`graph.json` structure:**
```json
{
  "creation_order": ["uuid1", "uuid2", ...],
  "nodes": {"uuid1": {...}, "uuid2": {...}},
  "src_out":  {"uuid1": ["uuid3", "uuid4"], ...},
  "evid_out": {"uuid1": {"uuid3": "subtype"}, ...}
}
```

**Corruption steps:**
1. Sort `nodes` keys → sample 30% → build `deleted_uuids` set → delete from `nodes`
2. Remove `deleted_uuids` from `creation_order` (preserve order)
3. Remove corresponding rows from `graph_embeddings.npy` (based on `creation_order` positions)
4. Clean `src_out`:
   - Drop entries whose key is in `deleted_uuids`
   - Remove `deleted_uuids` members from each value list
5. Clean `evid_out`:
   - Drop entries whose key is in `deleted_uuids`
   - Remove `deleted_uuids` members from each value dict

**Restoring in-memory state:**
```python
emb_map = {uid: embeddings[i] for i, uid in enumerate(creation_order)}
graph._nodes = {uid: Node(**{**d, "embedding": emb_map[uid]}) for uid, d in nodes.items()}
graph._creation_order = creation_order
graph._src_out = {k: set(v) for k, v in src_out.items()}
graph._evid_out = {k: dict(v) for k, v in evid_out.items()}
# Rebuild reverse edges (_src_in, _evid_in) from src_out / evid_out
graph._rebuild_reverse_edges()
module._global_turn = module_state["global_turn"]
module._current_conv_id = module_state["current_conv_id"]
module._current_turn_id = module_state["current_turn_id"]
module._context_cache = ContextCache.from_snapshot(cache_json)
```
Check whether `_rebuild_reverse_edges()` already exists in `HeterogeneousGraph`; if not, implement inline in the script (see Section 8).

---

### 5.5 ldagent

**Snapshot files:**
```
long_term_memory.json  ← {"documents": [...], "metadatas": [...]}       ← corrupt target
ltm_embeddings.npy     ← shape (N, 384)                                 ← sync with LTM deletion
short_term_memory.json ← List[Dict]                                      ← corrupt target
personas.json          ← {"user_traits": [...], "agent_traits": [...]}  ← corrupt target
memory_state.json      ← preserve (load as-is)
```

Note: STM is **not** cleared after the Phase 2 flush (`clear_stm_after_store=False`). The snapshot is saved after Phase 2, so STM retains the last dialogue group and is non-empty in general.

**Corruption steps:**
1. LTM: `documents[i]` and `metadatas[i]` are 1:1 aligned → sample 30% of indices → delete from both lists and remove corresponding rows from `ltm_embeddings.npy`
2. STM: sample 30% of indices from `short_term_memory.json` → delete. Skip if empty.
3. `user_traits`: sample 30% of list indices → delete
4. `agent_traits`: sample 30% of list indices → delete (independent sample from user_traits)

**`EventMemory.load_snapshot()` and `Personas.load_snapshot()` are already implemented** — call them directly on the freshly created agent:

```python
agent.memory_bank.load_snapshot(snapshot_dir)
agent.personas.load_snapshot(snapshot_dir)
```

After loading, apply `corrupt_agent(agent, corrupt_rate, rng)` which modifies `_ltm_metadata`, `_ltm_documents`, `_ltm_embeddings`, `short_term_memory`, `user_traits`, `agent_traits` in-place.

---

### 5.6 dense

**Snapshot files:**
```
memories.json   ← List[{"content", "embedding", "session_id", "conv_id", "turn_id"}]  ← corrupt target
```
Embeddings are stored inline as float lists. No separate `.npy` file.

**Corruption steps:**
1. Sample 30% of list indices → delete those entries

**Restoring in-memory state:**
```python
store._memories = [
    MemoryUnit(
        content=d["content"],
        embedding=np.array(d["embedding"], dtype=np.float32),
        session_id=d["session_id"],
        conv_id=d["conv_id"],
        turn_id=d["turn_id"]
    )
    for d in memories_json
]
```
`DenseMemoryStore` has no `load_snapshot()` — handle entirely inside the script.

---

## 6. Script Architecture (`run_corruption.py`)

Each module has its own `run_corruption.py` inside its directory. The script inherits from the existing `BatchedRunner` class in `run_experiment.py`, overriding only `run_batch()`. This avoids duplicating Phase 2 logic.

```
run_corruption.py  (per module, e.g. ldagent/run_corruption.py)
│
├── find_snapshot_dir(snapshot_base, session_id) → Path
│     Iterates session_{start}_{end}/ subdirs to locate memory_snapshots/session_{id}/
│
├── corrupt_agent(agent, corrupt_rate, rng) → Dict
│     Deletes floor(N * corrupt_rate) items in-place from each memory structure.
│     Returns corruption metadata dict added to config_metadata.
│
├── class CorruptionRunner(BatchedLDAgentRunner)   ← inherits Phase 2 logic
│   ├── __init__: calls super().__init__(), then overrides output path attributes
│   └── run_batch():
│         for each session:
│           1. create empty LDAgentModule
│           2. agent.memory_bank.load_snapshot(snapshot_dir)
│           3. agent.personas.load_snapshot(snapshot_dir)
│           4. corrupt_agent(agent, corrupt_rate, Random(42 + session_id))
│           5. record memory_at_qa_start
│         call inherited _run_phase2_batched()
│         assemble results with corruption metadata
│
└── main()
      auto-derive snapshot_base and output_dir from --model/--config/--subset/--corrupt-rate
      initialize LLM client, spaCy, SentenceTransformer
      inject cfg and logger into run_experiment namespace (_re.cfg, _re.logger)
      run batched loop identical to run_experiment.py
```

**cfg sharing:** `_run_phase2_batched()` references the module-level `cfg` in `run_experiment.py`. Setting `_re.cfg = cfg` before instantiating the runner ensures the inherited method sees the correct config.

---

## 7. Randomness & Reproducibility

```python
import random, math

# Per-session RNG — seed is 42 + session_id for full reproducibility
# regardless of batch boundaries or run order
session_rng = random.Random(42 + session_id)

# dict-based (amem memories, gmem4 nodes, theanine nodes)
all_keys = sorted(data.keys())          # sort first to guarantee deterministic order
n_delete = math.floor(len(all_keys) * corrupt_rate)
to_delete = set(session_rng.sample(all_keys, n_delete))

# list-based (memorybank entries, ldagent LTM/STM/traits, dense memories)
all_indices = list(range(len(data)))
n_delete = math.floor(len(data) * corrupt_rate)
to_delete = set(session_rng.sample(all_indices, n_delete))
```

Within the same session, each structure (LTM, STM, user_traits, agent_traits for ldagent; entries, daily_event_summaries, daily_personality_summaries for memorybank) calls `session_rng.sample()` independently in sequence, so each ends up with a different set of deleted items.

---

## 8. Implementation Notes

### amem — links index remap
`MemoryNote.links` stores integer indices into `list(memories.values())`. After deletion, positions shift and links become stale. Remap before assigning:

```python
old_to_new = {}
new_idx = 0
for old_idx, key in enumerate(original_order):
    if key not in deleted_keys:
        old_to_new[old_idx] = new_idx
        new_idx += 1

for note in surviving_memories.values():
    note.links = [old_to_new[i] for i in note.links if i in old_to_new]
```

### gmem4 — rebuilding reverse edges
`_src_in` and `_evid_in` are not persisted in the snapshot; they are derived from `src_out`/`evid_out` on load. Check whether `HeterogeneousGraph` already has a `_rebuild_reverse_edges()` method. If not, implement in the script:

```python
from collections import defaultdict

src_in = defaultdict(set)
for src, targets in src_out.items():
    for tgt in targets:
        src_in[tgt].add(src)

evid_in = defaultdict(dict)
for src, targets in evid_out.items():
    for tgt, subtype in targets.items():
        evid_in[tgt][src] = subtype
```

### memorybank — rebuilding daily_summary_entries
`daily_summary_entries: List[DailySummaryEntry]` and `daily_summary_retriever` are not stored in the snapshot directly. They are reconstructed by `_rebuild_daily_summary_index()`, which reads `daily_event_summaries`, builds the entry list, and re-encodes embeddings. This method must be called after restoring state for Phase 2 retrieval to work correctly.

### dense — no re-encoding needed
Embeddings are stored inline in `memories.json` as float lists. Restore with `np.array(d["embedding"], dtype=np.float32)` — no re-encoding step required.

### ldagent — STM is non-empty after flush
`flush_stm()` uses `clear_stm_after_store=False`, so STM is retained after the Phase 2 flush. The snapshot (saved after Phase 2) will generally have STM content. Corruption applies normally; skip only if the list happens to be empty.

### Snapshot lookup across parallel run directories
Existing experiments split sessions across multiple `session_{start}_{end}/` subdirectories. To find the snapshot for a given `session_id`:

```python
def find_snapshot_dir(snapshot_base: Path, session_id: int) -> Path:
    for session_dir in sorted(snapshot_base.iterdir()):
        if not session_dir.name.startswith("session_"):
            continue
        snap_path = session_dir / "memory_snapshots" / f"session_{session_id}"
        if snap_path.exists():
            return snap_path
    raise FileNotFoundError(f"snapshot for session {session_id} not found")
```

### LLM client initialization
`run_corruption.py` accepts the same LLM arguments as `run_experiment.py`: `--model`, `--tensor-parallel`, `--gpu-memory`, `--max-model-len`. The client is initialized identically.

---

## 9. Example Commands

```bash
# ldagent, opposed, 30% corruption, sessions 0–499
CUDA_VISIBLE_DEVICES=0 nohup python run_corruption.py \
  --start-session 0 --end-session 499 \
  --subset opposed \
  --model Qwen/Qwen3-1.7B \
  --corrupt-rate 0.3 \
  --tensor-parallel 1 --gpu-memory 0.34 --max-model-len 30000 \
  --batch-size 25 --config config_0 \
  > nohup/nohup_corrupt30_1.7b_0_499.out 2>&1 &

# ldagent, supportive, 50% corruption, sessions 0–499
CUDA_VISIBLE_DEVICES=1 nohup python run_corruption.py \
  --start-session 0 --end-session 499 \
  --subset supportive \
  --model Qwen/Qwen3-1.7B \
  --corrupt-rate 0.5 \
  --tensor-parallel 1 --gpu-memory 0.34 --max-model-len 30000 \
  --batch-size 25 --config config_0 \
  > nohup/nohup_corrupt50_1.7b_supportive_0_499.out 2>&1 &
```

Result merging uses the existing `merge_results.py` without modification (output format is identical).
