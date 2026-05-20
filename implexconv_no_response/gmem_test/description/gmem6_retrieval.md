# GraphMem: Retrieval

> **Scope**: This document defines retrieval only. It covers seed retrieval, graph expansion, final-set assembly, and the prompt-ready retrieval output used before response prompting or QA. Storage schema, labels, and extraction rules are defined in `gmem6_storage_extraction.md`.

---

## 1. Overview

GraphMem retrieval uses a 3-stage pipeline:

1. **Seed Retrieval**
2. **Graph Expansion**
3. **Final Set Assembly**

The resulting final set is serialized and passed to ⑥ QA answering. This variant is QA-only — there is no ① Response prompt call in the code (the retrieval-serialized context is reused later by ⑥).

---

## 2. Step 1: Seed Retrieval

### 2.1 Query Preprocessing

```text
Query = {
    query_context:  str,
    query_keywords: Set[str],
}
```

### 2.2 Overlap Definition

Query-side normalized overlap, scaled to [0, 1]:

```text
overlap_norm(q, n) = |query_keywords(q) ∩ label_set(n)| / max(|query_keywords(q)|, 1)
```

Label set depends on node type:

```text
label_set(c) = keywords(c)
label_set(e) = keywords(e) ∪ domain_label(e)
label_set(s) = keywords(s) ∪ domain_label(s)
label_set(t) = keywords(t) ∪ domain_label(t)
```

Notes:
- `sem(q, n)` is cosine similarity of embeddings, effectively in [0, 1].
- `overlap_norm` is in [0, 1] by construction.
- Both components share the same scale, so weights directly reflect relative importance.
- **Change 3 (gmem6) — single-token labels.** Every keyword and every domain label is a single token (no whitespace). The set intersection above is therefore a token-level comparison; multi-word phrases like `"machine learning"` or `"vegetarian food"` cannot occur on either side. Lowercase normalization and whitespace filtering are applied uniformly on both query and node side. See `gmem6_storage_extraction.md` §4.5 / §4.6 for the storage-side enforcement.

### 2.3 Context Node (`c`)

Context nodes do not have `scope`. They use a fixed weight pair:

```text
seed_score(c, q) = w_sem_c · sem(q, c)
                 + w_ov_c  · overlap_norm(q, c)
```

Select top-`k_c`.

### 2.4 Episode Node (`e`)

```text
seed_score(e, q) = w_sem(e.scope) · sem(q, e)
                 + w_ov(e.scope)  · overlap_norm(q, e)
```

Select top-`k_e`.

### 2.5 State Node (`s`)

```text
seed_score(s, q) = w_sem(s.scope) · sem(q, s)
                 + w_ov(s.scope)  · overlap_norm(q, s)
```

State seed candidates are scored over **all states except APS members** (APS is constructed first; see §2.7). Select top-`k_s` from the resulting `s_seed` pool.

Disjointness: `s_aps` and `s_seed` are mutually exclusive. HIGH recall-priority states that are not selected for APS (because more than `k_aps` HIGH candidates exist) **fall through** to the `s_seed` pool and compete normally there.

### 2.6 Trait Node (`t`)

```text
seed_score(t, q) = w_sem(t.scope) · sem(q, t)
                 + w_ov(t.scope)  · overlap_norm(q, t)
```

Select top-`k_t`.

Notes:
- Trait scoring uses the same scope-dependent weights as states and episodes.
- Trait validation (stable vs. challenged) handles "how established is this trait" at final-set assembly; seed scoring focuses only on query relevance.

### 2.7 Active Persona Set (`aps`)

`aps` is constructed by:

1. taking all states with `recall_priority = HIGH`;
2. excluding states that are sources of a `SHIFT_TO` edge (`APS_EXCLUDE_SHIFT_SOURCE`);
3. ranking the remaining candidates by `seed_score(s, query)` — the same scope-dependent score defined in §2.5 (NARROW and BROAD HIGH candidates are scored with their respective weights and ranked together);
4. selecting the top `k_aps`.

Membership criterion is `recall_priority = HIGH` only; **`scope` is not a membership criterion** (scope continues to drive seed-score weights).

Assumption:
- `recall_priority = HIGH` is assigned conservatively at extraction time;
- `HIGH` should be reserved for states whose omission would likely harm the immediate next response.

Disjointness with `s_seed`:
- APS members are removed from the `s_seed` candidate pool (§2.5);
- HIGH states ranked outside the APS top-`k_aps` are **not discarded** — they fall through to the `s_seed` pool and compete by normal seed scoring.

#### Change 1 (gmem6) — APS is not an expansion origin

APS nodes are added to the post-seed pool so they remain available as evidence in `_cross_type_sup_con` (they continue to contribute SUP / CON signals to other nodes' support ratios). However, **APS nodes are excluded from `origin_ids` in Step 2a** — they no longer drive query-independent sign propagation. APS is a query-independent persona slot that is always serialized as *Current Constraints*; letting APS members also act as expansion seeds would conflate "always-included persona context" with "query-driven retrieval driver". Pool size shrinks modestly; expansion becomes strictly query-driven.

#### APS Filtering (single pass)
APS construction is single-pass at retrieval time. Expansion does not create new edges, so any state that survives `APS_EXCLUDE_SHIFT_SOURCE` filtering remains a `SHIFT_TO`-terminal node throughout the retrieval call; no post-expansion re-filter is needed.

#### Edge cases
- If no HIGH candidates exist after `SHIFT_TO`-source filtering, APS is empty (correct and intended).
- If only one HIGH candidate exists with very low `seed_score`, it still enters APS — this is the implicit-persona-reasoning case the design supports.

### 2.8 `retrieval_count` Update

After final-set assembly, increment `retrieval_count += 1` for every node in the final set.

`retrieval_count` is not used in seed scoring. It is retained as an analytical field for debugging and post-hoc analysis.

### 2.9 Pool After Seed Retrieval

```text
{c_seed, e_seed, s_seed, t_seed, s_aps}
```

---

## 3. Step 2: Graph Expansion

### 3.1 Purpose

Seed retrieval captures only directly similar nodes. Expansion discovers additional related nodes through evidence edges and temporal transitions, building a richer pool for final-set assembly.

### 3.2 Expansion Procedure

| Step | Process | Pool |
|------|---------|------|
| 0 | after seed retrieval | `{c_seed, e_seed, s_seed, t_seed, s_aps}` |
| 1 | `c_seed` → `SOURCE` → source-derived `e`, `s`, **`t`**; remove `c` from pool | `{e_seed, s_seed, t_seed, s_aps, s_src, e_src, t_src}` |
| 1.5 | **Change 2 (gmem6)** — for each node in `s_seed ∪ e_seed`, pull S/E nodes whose `(conv_id, turn_id)` differs by exactly ±1 turn within the same `conv_id` into the pool as plain members | pool ∪ `{s_turn, e_turn}` |
| 2 | expand over `EVIDENCE` edges using Rules A, B (Rule B now folded into Rule A — see §3.3) | enlarged pool |
| 3 | deduplicate | final expanded pool |

Note: traits CAN be SOURCE-children of context nodes (④ adds
`SOURCE` edges from each chunk's context nodes to the new trait). Step 1
must therefore admit `t` along with `e` and `s`; otherwise traits with no
direct seed and only `SOURCE` connection from c are missed.

**Change 2 (gmem6) — ±1 turn neighbor inclusion.** State extraction is sparse (`STATE_MAX_COUNT=1`, `CHUNK_SIZE_CONV=1`), so a single user situation often spans several consecutive turns. When only one fragment scores above the seed threshold, the rest is lost. Pulling ±1 turn neighbors of `s_seed` and `e_seed` into the pool acts as a recall safety net for implicit relations the evidence extractor did not (or could not) capture. Pool growth is modest (≤ 2× per S/E seed, often less due to extraction sparsity). Constraints:
- The lookup is restricted to S and E types — `c_seed` is not touched (its SOURCE children already span the surrounding turn at extraction time), and traits do not have meaningful per-turn locality.
- The lookup never crosses `conv_id` boundaries; a `conv_id` jump corresponds to ~12 hours under the time model.
- Turn-neighbors are added to the pool **only** — they are **not** added to `origin_ids` (they are evidence/recall reinforcement, not expansion drivers).

### 3.3 Expansion Rules

#### Rule A: Support-chain expansion with 1-hop contradiction pull (SHIFT_TO included)

From a pool node `A`:
- **1-hop**: follow `SUPPORT`, `CONTRADICT`, and `SHIFT_TO` edges. Add discovered nodes to the pool.
- **2-hop and beyond**: follow `SUPPORT`-typed edges only (this includes outgoing `SHIFT_TO`, see Change 4 below). Stop at any non-support edge.

1-hop `CONTRADICT` nodes are added to the pool but do **not** serve as expansion origins for further hops.

##### Change 4 (gmem6) — SHIFT_TO unified into sign propagation

In gmem5, `SHIFT_TO` was handled by a separate forward-only BFS (Rule B) and a bidirectional bonus at final-set assembly. Both layers are removed in gmem6 in favor of a single rule that lives inside the ordinary BFS. The actual mapping in `graph_store.py:_ordinary_evidence_neighbors` splits same-type and cross-type SHIFT_TO:

```text
At a frontier node X:
  outgoing SHIFT_TO  X → Y, Y same-type as X  → emit edge as SUP
                                                (flow forward to the newer node)
  outgoing SHIFT_TO  X → Y, Y cross-type      → emit edge as CON
                                                (source invalidates the cross-type target)
  incoming SHIFT_TO  Z → X (any types)        → emit edge as CON
                                                (record older version, stop here)
```

Ordinary evidence edges (`SUPPORT`, `CONTRADICT`, `IRRELEVANT`) follow the per-node-type outgoing-direction whitelist `_EVID_EXPAND_OUT`:

```text
C → ∅           (context nodes do not emit ordinary evidence)
E → {S, E, T}
S → {S, T}
T → {T}
```

Cross-type ordinary **incoming** edges are also dropped: only same-type incoming ordinary edges are followed back at a frontier node. (Same-type incoming SHIFT_TO is still emitted as CON above.)

Consequences:
- Same-type outgoing SHIFT_TO is SUP-typed, so the BFS continues forward to the newer version exactly as it would for a SUP-evidence neighbor.
- Cross-type outgoing SHIFT_TO is CON-typed and stops traversal.
- Incoming SHIFT_TO is CON-typed, so the SUP × CON → CON-stop rule records the older node and halts there. The old node is never used as an expansion driver.
- The dedicated forward BFS (gmem5 Rule B) and the final-set bidirectional bonus block (gmem5 §3.5) are both removed; the same signal is now carried entirely by `signed_cache`.

##### SHIFT_TO chain depth (caveat)

Because outgoing SHIFT_TO is now CON-typed once you have already taken a SHIFT_TO step (SUP × CON → CON-stop), **SHIFT_TO chains collapse to depth 1 in propagation**. For a chain `A →SHIFT_TO→ B →SHIFT_TO→ C` rooted at A, only B is reached via signed propagation; C is not. If long SHIFT_TO chains turn out to be frequent in ImplexConv, mitigate by hoisting the existing `_shift_chain_collapse` logic to retrieval entry (keep only the latest node in each chain). This mitigation is **not enabled by default** — it is conditional on ablation.

#### Rule C: Contradiction preservation

Contradiction-like evidence is preserved in the pool for use during final-set assembly:
- direct `CONTRADICT` neighbors pulled in by Rule A (1-hop),
- contradiction reached after support-chain expansion,
- `SHIFT_TO`-incoming nodes (older versions) reached via the in=CON mapping above.

These nodes are not removed during expansion; they participate in support-ratio computation at final-set assembly.

### 3.4 Sign Table

The sign table governs multi-hop traversal over evidence edges. After Change 4 (gmem6), `SHIFT_TO` participates in the same table via the same-type / cross-type / direction-dependent edge mapping (§3.3). `IRRELEVANT` is carried as a third sign in `signed_cache`; it does **not** match `SUP` for support-ratio purposes and it does **not** trigger the `SUP × CON → CON-stop` rule.

| Current cumulative sign | Edge sign (after SHIFT_TO mapping) | Result | Action |
|-------------------------|-------------------------------------|--------|--------|
| `SUP` | `SUP`         | `SUP` | continue |
| `SUP` | `CON`         | `CON` | **stop** |
| `SUP` | `IRRELEVANT`  | `IRRELEVANT` | stop (recorded as IRR, not folded into SUP/CON) |
| `CON` | —             | —     | **stop** (CON terminates immediately) |
| `IRRELEVANT` | —      | —     | stop |

Key properties:
- CON at any point stops traversal immediately. There is no multi-hop contradiction propagation.
- SHIFT_TO is no longer a separate channel; it is mapped to SUP/CON via the rule in §3.3 and then handled by this same table.
- IRRELEVANT is stored as a third sign so that 1-hop IRRELEVANT direct edges are distinguishable from "no evidence". Support-ratio uses only SUP/CON counts; an IRRELEVANT-only neighbor leaves a node in the "no evidence → 0.5" default branch (§4.1).

### 3.5 Per-Origin BFS Dedup (Change 5 / spec only)

Within a single `get_ordinary_signed_reachable(start)` call, a node `X` is visited at most once. If `X` is reached at hop `h`, then in any later hop `h' > h` the edges out of `X` are not traversed for the purposes of this BFS; `X` is skipped.

This is **per-origin** dedup only. Different origin BFSes are independent and may each visit `X`. Sharing visited sets across origins would make results origin-order dependent and is explicitly **not** part of this spec.

### 3.6 Multiple Paths Between Nodes — CON wins on ties (Change 6)

If multiple paths exist between two nodes:
1. **Shortest path takes priority** (preserved across hop levels).
2. **Among equal-length paths, CON takes priority over SUP.** (gmem5 had the opposite rule.)

Rationale: under the gmem5 "SUP wins on ties" rule, equal-length CON evidence was silently swallowed. For the ImplexConv *opposed* subset — where the user's situation must override a generic answer — losing CON signal is the costlier error. CON-wins better surfaces contradiction-bearing nodes for challenged-trait detection and persona override. Combined with Change 4, CON-wins also makes outgoing SHIFT_TO edges (now SUP-typed) lose to equally short CON paths, which is consistent with the intended "outdatedness is sticky" semantics.

### 3.7 Expansion Constraints

- Expansion is hop-capped by `SIGN_PROP_HOP_CAP`.
- Already-seen nodes are not re-added (deduplication).
- Retrieval does not create new nodes.
- **Change 7 (gmem6) — caching.** `get_ordinary_signed_reachable(start_id, hop_cap)` is memoized for the duration of a single `retrieve()` call (per-retrieve cache, reset at every call). The cache is keyed on `(start_id, hop_cap)` and is safe because the underlying graph is immutable for the duration of a retrieval call. This removes the duplicate computation between expansion (one BFS per origin) and `signed_cache` construction (one BFS per pool member) for nodes that are both. No effect on retrieved content.

### 3.8 Pool After Expansion

```text
{e_seed, s_seed, t_seed, s_aps, s_src, e_src, s_exp, e_exp, t_exp}
```

This is the **final expanded pool** used for final-set assembly.

---

## 4. Step 3: Final Set Assembly

### 4.1 Common Scoring Structure

All node types (traits, states, episodes) use the same final scoring formula:

```text
support_ratio(n) = sup_w / (sup_w + con_w)
final_score(n, q) = w_sr · support_ratio(n) + (1 - w_sr) · seed_score(n, q)
```

Rules:
- `sup_w` and `con_w` are computed from pooled evidence reachable under the directional traversal in §3.3 / §3.4. Evidence is **not** symmetric across all type pairs: ordinary outgoing edges follow the `_EVID_EXPAND_OUT` whitelist (C:∅; E→{S,E,T}; S→{S,T}; T→{T}); ordinary incoming edges are followed only same-type; SHIFT_TO follows the same-type / cross-type / direction split. The previous wording "every node type serves as evidence for every other node type" was a simplification — the actual flow respects these constraints.
- If a node has **no evidence** under the rules above (sup_w = con_w = 0), `support_ratio` defaults to **0.5**.
- For expansion nodes not in the original seed set, `seed_score(n, q)` is computed at assembly time using the same formula as seed retrieval.
- `w_sr` is a configurable parameter (default 0.5).

### 4.2 `retrieval_relation(x, y)` for Support Ratio

For each pooled node `y` being evaluated, collect evidence from other pooled nodes `x`:

#### Single-channel relation (Change 4 / gmem6)

Compute multi-hop relation over `SUPPORT / CONTRADICT / IRRELEVANT / SHIFT_TO` edges using the sign table (Section 3.4) with the SHIFT_TO mapping (out → SUP, in → CON). The result is either `SUP`, `CON`, or no reachable path.

The gmem5 separate "SHIFT_TO bidirectional" channel at support-ratio time is **removed**: keeping it alongside the unified propagation would double-count direct 1-hop SHIFT_TO edges relative to 2-hop chains, producing a distance-dependent asymmetry. The bidirectional intuition (old node accumulates con, new node accumulates sup) survives — it is now carried by the in=CON / out=SUP edge mapping inside `signed_cache`.

### 4.3 Trait Selection

Target: all trait nodes in the final expanded pool.

```text
For each trait t:
    sup_w, con_w = 0, 0

    # Change 4 (gmem6): single channel — SHIFT_TO contribution is already
    # included in signed_cache via the in=CON / out=SUP edge mapping.
    For each pooled node n (n ≠ t, n ∈ {s, e, t}):
        sign = signed_cache[n][t]   # multi-hop sign over SUP/CON/SHIFT_TO
        if sign == SUP:  sup_w += 1
        elif sign == CON: con_w += 1

    support_ratio(t) = sup_w / (sup_w + con_w)    # default 0.5 if no evidence
    final_score(t, q) = w_sr · support_ratio(t) + (1 - w_sr) · seed_score(t, q)
```

Select top-`k_t_final` → preliminary `t_f`.

#### Trait classification within `t_f`

After selecting top-k traits, classify each (see `retriever.py:classify_traits`):

```text
has_shifted     = ∃ trait t' in t_f reachable from t via outgoing SHIFT_TO
                  (t's "newer version" sits in the same selected pool)
has_contradict  = evidence_count > 0 and ratio ≤ τ
challenged      = has_shifted OR has_contradict
stable          = not challenged  (includes evidence_count == 0 with no SHIFT_TO out)
```

Both branches feed into the same `[Challenged Traits]` section. A trait with `has_shifted = True` but no CONTRADICT evidence is still serialized as challenged, with a `↳ shifted to: ...` sub-line naming the newer trait(s) — this is the channel that surfaces same-type SHIFT_TO drift even when no contradictory state/episode was extracted.

#### Conflict collection for challenged traits

For each challenged trait, `retriever.py` populates `conflict_per_trait[trait.node_id] = {"shifted_to": [...], "states": [...], "episodes": [...]}`:

- `shifted_to`: newer same-type traits in `t_f` reachable from the trait via outgoing SHIFT_TO. Used to render the `↳ shifted to:` sub-line.
- `states`: pooled states whose multi-hop sign at the trait is `CON` (direct `CONTRADICT` edges and incoming SHIFT_TO from older same-type sources both surface here automatically).
- `episodes`: same as above, restricted to pooled episodes.

These three lists drive the nested serialization under each challenged trait.

#### Shift-chain collapse for traits

If a SHIFT_TO chain exists among selected traits:
```text
A →SHIFT_TO→ B →SHIFT_TO→ C
```
- if A, B, C are all in `t_f`, keep only C;
- if A and C are in `t_f`, drop A.

### 4.4 State Selection

Target: pooled states excluding APS members (APS / `s_seed` disjointness, see §2.5 / §2.7) and states used as challenged-trait conflict evidence.

```text
For each candidate state n:
    sup_w, con_w = 0, 0

    # Change 4 (gmem6): single channel — SHIFT_TO contribution is already
    # included in signed_cache via the in=CON / out=SUP edge mapping.
    For each pooled node p (p ≠ n, p ∈ {s, e, t}):
        sign = signed_cache[p][n]
        if sign == SUP:  sup_w += 1
        elif sign == CON: con_w += 1

    support_ratio(n) = sup_w / (sup_w + con_w)    # default 0.5 if no evidence
    final_score(n, q) = w_sr · support_ratio(n) + (1 - w_sr) · seed_score(n, q)
```

Select top-`k_sf` → preliminary `s_f`.

#### Shift-chain collapse for `s_f`

Same logic as trait collapse:
```text
A →SHIFT_TO→ B →SHIFT_TO→ C
```
- if A, B, C are all in `s_f`, keep only C;
- if A and C are in `s_f`, drop A.

### 4.5 Episode Selection

Target: all pooled episodes (seed + expansion), excluding episodes used as challenged-trait conflict evidence.

```text
For each candidate episode n:
    sup_w, con_w = 0, 0

    For each pooled node p (p ≠ n, p ∈ {s, e, t}):
        sign = signed_cache[p][n]
        if sign == SUP:  sup_w += 1
        elif sign == CON: con_w += 1

    # Episode nodes do not have SHIFT_TO edges (SHIFT_TO is same-type only,
    # restricted to s↔s and t↔t), so the in=CON / out=SUP mapping never
    # fires on an episode.

    support_ratio(n) = sup_w / (sup_w + con_w)    # default 0.5 if no evidence
    final_score(n, q) = w_sr · support_ratio(n) + (1 - w_sr) · seed_score(n, q)
```

For expansion episodes not in `e_seed`, `seed_score(n, q)` is computed at this point using the same formula as seed retrieval (Section 2.4).

Select top-`k_e_final` → `e_f`.

---

## 5. Final Set

```text
{t_final, s_final, e_final, s_aps}
```

---

## 6. Serialization

The final set is serialized into prompt-ready sections:

```text
[Current Constraints]
[Traits]
[Challenged Traits]
[Relevant States]
[Relevant Episodes]
[Recent Conversation]
```

### Section Sources

| Section | Source |
|---------|--------|
| Current Constraints | `s_aps` |
| Traits | stable traits from `t_final` |
| Challenged Traits | challenged traits from `t_final` with nested conflict evidence |
| Relevant States | `s_final` |
| Relevant Episodes | `e_final` |
| Recent Conversation | context cache |

### Notes

- This variant is QA-only — there is no ① Response prompt call. `[Recent Conversation]` exists only on the QA path.
- For QA answering, `[Recent Conversation]` is controlled by `INCLUDE_RECENT_CONVERSATION_FOR_QA`. The current default in `config_0.py` is `True` (on).
- Shift-chain collapse is applied when forming `t_final`, `s_final`, and `s_aps`.
- Within `[Challenged Traits]`, each challenged trait may render a `↳ shifted to: <newer trait content>` sub-line in addition to the `states` / `episodes` conflict evidence, when a same-type SHIFT_TO out-edge to another selected trait exists.

### Retrieval output contract (`RetrievalResult`)

`GraphRetriever.retrieve(...)` returns a `RetrievalResult` dataclass (see `retriever.py`). The serialized text above is the `serialized` field. Other fields useful to callers:

| Field | Type | Description |
|-------|------|-------------|
| `active_persona` | `List[StateNode]` | the materialized APS members (rendered under `[Current Constraints]`). |
| `traits_stable` | `List[TraitNode]` | stable traits in `t_final` (rendered under `[Traits]`). |
| `traits_challenged` | `List[TraitNode]` | challenged traits in `t_final` (rendered under `[Challenged Traits]`). |
| `states_conflict` | `List[StateNode]` | states surfaced as challenged-trait conflict evidence. |
| `episodes_conflict` | `List[EpisodeNode]` | episodes surfaced as challenged-trait conflict evidence. |
| `states_relevant` | `List[StateNode]` | `s_final` minus conflict-evidence states (rendered under `[Relevant States]`). |
| `episodes_relevant` | `List[EpisodeNode]` | `e_final` minus conflict-evidence episodes (rendered under `[Relevant Episodes]`). |
| `conflict_per_trait` | `Dict[node_id, {"shifted_to": [...], "states": [...], "episodes": [...]}]` | per-challenged-trait conflict breakdown driving the nested serialization (`↳ shifted to:` sub-line is sourced from `shifted_to`). |
| `seed_contexts`, `seed_episodes`, `seed_states`, `seed_traits` | seed-stage outputs per type, retained for debugging. |
| `pool_nodes` | `Set[node_id]` | the full expanded pool at the moment of final-set scoring. |
| `node_scores` | `Dict[node_id, score]` | per-node `final_score` recorded at assembly time. |
| `all_final_nodes` | `List[Node]` | union of APS + traits + states + episodes used for `retrieval_count += 1` post-assembly. |

Note: the dataclass field is `active_persona` (not `s_aps`) and the conflict evidence is split into `states_conflict` and `episodes_conflict` at the top level in addition to the per-trait breakdown.

---

## 7. Retrieval Summary

GraphMem v6 retrieval combines:
- **unified seed scoring** with semantic similarity and query-side normalized overlap, using scope-dependent weights, with **single-token keywords and domain labels** (Change 3) on both sides of the set intersection,
- **1-hop contradiction pull** to surface direct conflicting evidence,
- **SHIFT_TO unified into sign propagation** (Change 4): outgoing SHIFT_TO emits a SUP edge (continue forward), incoming SHIFT_TO emits a CON edge (record old version, stop). The dedicated forward BFS and the final-set bidirectional bonus from gmem5 are removed,
- **CON wins on ties** in multi-path conflict resolution (Change 6),
- **uniform final scoring** across all node types: `w_sr · support_ratio + (1 - w_sr) · seed_score`,
- **directional cross-type evidence**: ordinary edges follow `_EVID_EXPAND_OUT` (C:∅; E→{S,E,T}; S→{S,T}; T→{T}) and only same-type incoming ordinary edges; SHIFT_TO follows the same-type / cross-type / direction split in §3.3,
- **shift-chain collapse** for traits, states, and APS,
- **APS** for implicit persona reasoning: top-`k_aps` HIGH recall-priority, non-`SHIFT_TO`-source states ranked by `seed_score`. APS and `s_seed` are disjoint partitions of the state pool. **APS members are evidence-only — they do not act as expansion origins** (Change 1),
- **±1 turn neighbor inclusion** for `s_seed` and `e_seed` (Change 2): same-`conv_id` adjacent S/E nodes are pulled into the pool (not as origins) as a recall safety net for sparse extraction,
- **per-retrieve memoization** of `get_ordinary_signed_reachable` (Change 7) — engineering only.

This yields:
- query-relevant HIGH recall-priority states through APS, with HIGH non-APS states still able to surface in `s_seed`,
- conflict-aware trait handling with stable/challenged classification,
- newer state and trait recovery from outdated seeds,
- evidence-scored episode selection from the full pool,
- and controlled pruning without globally deleting older evidence.
