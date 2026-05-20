# GraphMem: Retrieval

> **Scope**: This document defines retrieval only. It covers seed retrieval, graph expansion, final-set assembly, and the prompt-ready retrieval output used before response prompting or QA. Storage schema, labels, and extraction rules are defined in `gmem5_storage_extraction.md`.

---

## 1. Overview

GraphMem retrieval uses a 3-stage pipeline:

1. **Seed Retrieval**
2. **Graph Expansion**
3. **Final Set Assembly**

The resulting final set is serialized and passed to:
- ① response prompt construction
- ⑥ QA answering

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
label_set(m) = keywords(m) ∪ domain_label(m)
label_set(s) = keywords(s) ∪ domain_label(s)
label_set(t) = keywords(t) ∪ domain_label(t)
```

Notes:
- `sem(q, n)` is cosine similarity of embeddings, effectively in [0, 1].
- `overlap_norm` is in [0, 1] by construction.
- Both components share the same scale, so weights directly reflect relative importance.

### 2.3 Context Node (`c`)

Context nodes do not have `scope`. They use a fixed weight pair:

```text
seed_score(c, q) = w_sem_c · sem(q, c)
                 + w_ov_c  · overlap_norm(q, c)
```

Select top-`k_c`.

### 2.4 Memory Node (`m`)

```text
seed_score(m, q) = w_sem(m.scope) · sem(q, m)
                 + w_ov(m.scope)  · overlap_norm(q, m)
```

Select top-`k_m`.

### 2.5 State Node (`s`)

```text
seed_score(s, q) = w_sem(s.scope) · sem(q, s)
                 + w_ov(s.scope)  · overlap_norm(q, s)
```

State seed candidates are scored over **all states except APS members** (APS is constructed first; see §2.7). Select top-`k_s` from the resulting `s_seed` pool.

Disjointness: `s_aps` and `s_seed` are mutually exclusive. HIGH-impact states that are not selected for APS (because more than `k_aps` HIGH candidates exist) **fall through** to the `s_seed` pool and compete normally there.

### 2.6 Trait Node (`t`)

```text
seed_score(t, q) = w_sem(t.scope) · sem(q, t)
                 + w_ov(t.scope)  · overlap_norm(q, t)
```

Select top-`k_t`.

Notes:
- Trait scoring uses the same scope-dependent weights as states and memories.
- Trait validation (stable vs. challenged) handles "how established is this trait" at final-set assembly; seed scoring focuses only on query relevance.

### 2.7 Active Persona Set (`aps`)

`aps` is constructed by:

1. taking all states with `current_decision_impact = HIGH`;
2. excluding states that are sources of a `SHIFT_TO` edge (`APS_EXCLUDE_SHIFT_SOURCE`);
3. ranking the remaining candidates by `seed_score(s, query)` — the same scope-dependent score defined in §2.5 (NARROW and BROAD HIGH candidates are scored with their respective weights and ranked together);
4. selecting the top `k_aps`.

Membership criterion is `current_decision_impact = HIGH` only; **`scope` is not a membership criterion** (scope continues to drive seed-score weights).

Assumption:
- `current_decision_impact = HIGH` is assigned conservatively at extraction time;
- `HIGH` should be reserved for states whose omission would likely harm the immediate next response.

Disjointness with `s_seed`:
- APS members are removed from the `s_seed` candidate pool (§2.5);
- HIGH states ranked outside the APS top-`k_aps` are **not discarded** — they fall through to the `s_seed` pool and compete by normal seed scoring.

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
{c_seed, m_seed, s_seed, t_seed, s_aps}
```

---

## 3. Step 2: Graph Expansion

### 3.1 Purpose

Seed retrieval captures only directly similar nodes. Expansion discovers additional related nodes through evidence edges and temporal transitions, building a richer pool for final-set assembly.

### 3.2 Expansion Procedure

| Step | Process | Pool |
|------|---------|------|
| 0 | after seed retrieval | `{c_seed, m_seed, s_seed, t_seed, s_aps}` |
| 1 | `c_seed` → `SOURCE` → source-derived `m`, `s`, **`t`**; remove `c` from pool | `{m_seed, s_seed, t_seed, s_aps, s_src, m_src, t_src}` |
| 2 | expand over `EVIDENCE` edges using Rules A, B, C | enlarged pool |
| 3 | deduplicate | final expanded pool |

Note: traits CAN be SOURCE-children of context nodes (④ adds
`SOURCE` edges from each chunk's context nodes to the new trait). Step 1
must therefore admit `t` along with `m` and `s`; otherwise traits with no
direct seed and only `SOURCE` connection from c are missed.

### 3.3 Expansion Rules

#### Rule A: Support-chain expansion with 1-hop contradiction pull

From a pool node `A`:
- **1-hop**: follow both `SUPPORT` and `CONTRADICT` edges. Add discovered nodes to the pool.
- **2-hop and beyond**: follow `SUPPORT` edges only. Stop at any non-support edge.

1-hop `CONTRADICT` nodes are added to the pool but do **not** serve as expansion origins for further hops.

#### Rule B: SHIFT_TO forward traversal

`SHIFT_TO` edges are traversed **only in the stored direction** (old → new):

```text
A →SHIFT_TO→ B →SHIFT_TO→ C →SUPPORT→ D
```

- If `A` is in the pool, then `B`, `C`, and `D` are all discoverable.
- If `C` is in the pool, `A` and `B` are **not** discoverable via reverse traversal.

SHIFT_TO traversal is purely for **discovery** — it adds nodes to the pool but does not assign cumulative signs. The sign implications of SHIFT_TO are handled entirely at final-set assembly time.

SHIFT_TO chains can be followed consecutively, and from any SHIFT_TO arrival node, support-chain expansion (Rule A) may resume.

#### Rule C: Contradiction preservation

Contradiction-like evidence is preserved in the pool for use during final-set assembly:
- direct `CONTRADICT` neighbors pulled in by Rule A (1-hop),
- contradiction reached after support-chain expansion.

These nodes are not removed during expansion; they participate in support-ratio computation at final-set assembly.

### 3.4 Sign Table

The sign table governs multi-hop traversal over ordinary evidence edges.

| Current cumulative sign | Edge sign | Result | Action |
|-------------------------|-----------|--------|--------|
| `SUP` | `SUP` | `SUP` | continue |
| `SUP` | `CON` | `CON` | **stop** |
| `SUP` | `IRRELEVANT` | — | stop |
| `CON` | — | — | **stop** (CON terminates immediately) |

Key simplifications from prior design:
- CON at any point stops traversal immediately. There is no multi-hop contradiction propagation.
- SHIFT_TO does **not** participate in the sign table during expansion. It is used only for forward traversal (Rule B).

### 3.5 SHIFT_TO in Sign Computation (Final-Set Assembly Only)

For support-ratio computation during final-set assembly, `SHIFT_TO(A → B)` contributes **bidirectionally**:

```text
Storage:       A →SHIFT_TO→ B    (A = old, B = new)
Sign reading:  bidirectional
```

This means:
- `A`'s support_ratio: receives `con_w += 1` (penalized as outdated);
- `B`'s support_ratio: receives `sup_w += 1` (boosted as current).

The bidirectional contribution applies **only at support-ratio computation time** (final-set assembly). Expansion traversal of `SHIFT_TO` remains forward-only (Rule B); `SHIFT_TO` is still excluded from the sign table during expansion.

Edge case: the rule requires **both endpoints in the pool**. If A is in the pool but B is not (or vice versa), this edge contributes nothing.

#### Worked Example
Chain `A → SHIFT_TO → B → SHIFT_TO → C`, all three in the pool, no other evidence:

| Node | sup_w | con_w | support_ratio | Interpretation |
|------|-------|-------|---------------|----------------|
| A    | 0     | 1 (out→B) | 0.0       | fully outdated |
| B    | 1 (in←A) | 1 (out→C) | 0.5    | once new, now superseded |
| C    | 1 (in←B) | 0     | 1.0           | currently valid |

Shift-chain collapse rules in §4.3 / §4.4 interact naturally: with the new scoring, B and C are more likely to outrank A in top-k, so collapse rules either drop A explicitly or A simply does not make it in.

### 3.6 Multiple Paths Between Nodes

If multiple paths exist between two nodes:
1. **Shortest path takes priority.**
2. **Among equal-length paths, SUP takes priority over CON.**

### 3.7 Expansion Constraints

- Expansion is hop-capped by `SIGN_PROP_HOP_CAP`.
- Already-seen nodes are not re-added (deduplication).
- Retrieval does not create new nodes.

### 3.8 Pool After Expansion

```text
{m_seed, s_seed, t_seed, s_aps, s_src, m_src, s_exp, m_exp, t_exp}
```

This is the **final expanded pool** used for final-set assembly.

---

## 4. Step 3: Final Set Assembly

### 4.1 Common Scoring Structure

All node types (traits, states, memories) use the same final scoring formula:

```text
support_ratio(n) = sup_w / (sup_w + con_w)
final_score(n, q) = w_sr · support_ratio(n) + (1 - w_sr) · seed_score(n, q)
```

Rules:
- `sup_w` and `con_w` are computed from **all node types** in the final expanded pool. Every node type can serve as evidence for every other node type.
- If a node has **no evidence** (sup_w = con_w = 0), `support_ratio` defaults to **0.5**.
- For expansion nodes not in the original seed set, `seed_score(n, q)` is computed at assembly time using the same formula as seed retrieval.
- `w_sr` is a configurable parameter (default 0.5).

### 4.2 `retrieval_relation(x, y)` for Support Ratio

For each pooled node `y` being evaluated, collect evidence from other pooled nodes `x`:

#### Ordinary relation

Compute multi-hop relation over `SUPPORT / CONTRADICT / IRRELEVANT` edges using the sign table (Section 3.4). The result is either `SUP`, `CON`, or no reachable path.

#### SHIFT_TO relation (bidirectional)

For `SHIFT_TO(A → B)` where both `A` and `B` are in the pool:
- When evaluating `A`: `con_w += 1` (A is the old node, penalized);
- When evaluating `B`: `sup_w += 1` (B is the new node, boosted).

This is the bidirectional interpretation described in Section 3.5.

### 4.3 Trait Selection

Target: all trait nodes in the final expanded pool.

```text
For each trait t:
    sup_w, con_w = 0, 0

    For each pooled node n (n ≠ t, n ∈ {s, m, t}):
        ordinary = multi-hop sign(n, t)
        if ordinary == SUP:  sup_w += 1
        elif ordinary == CON: con_w += 1

    # SHIFT_TO bidirectional contribution
    if ∃ SHIFT_TO(t → z) where z in pool:   # t is old
        con_w += 1                            # penalized as outdated
    if ∃ SHIFT_TO(z → t) where z in pool:   # t is new
        sup_w += 1                            # boosted as current

    support_ratio(t) = sup_w / (sup_w + con_w)    # default 0.5 if no evidence
    final_score(t, q) = w_sr · support_ratio(t) + (1 - w_sr) · seed_score(t, q)
```

Select top-`k_t_final` → preliminary `t_f`.

#### Trait classification within `t_f`

After selecting top-k traits, classify each:
- `evidence_count > 0` and `ratio ≤ τ` → **challenged** (serialized with nested conflict evidence)
- otherwise → **stable** (includes evidence_count == 0)

#### Conflict collection for challenged traits

For each challenged trait, collect pooled states and memories that contribute `CON` signal, including:
- direct CONTRADICT evidence,
- SHIFT_TO sources (the old trait itself has outgoing SHIFT_TO).

These are nested under the challenged trait in serialization.

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

    For each pooled node p (p ≠ n, p ∈ {s, m, t}):
        ordinary = multi-hop sign(p, n)
        if ordinary == SUP:  sup_w += 1
        elif ordinary == CON: con_w += 1

    # SHIFT_TO bidirectional contribution
    if ∃ SHIFT_TO(n → z) where z in pool:   # n is old
        con_w += 1                            # penalized as outdated
    if ∃ SHIFT_TO(z → n) where z in pool:   # n is new
        sup_w += 1                            # boosted as current

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

### 4.5 Memory Selection

Target: all pooled memories (seed + expansion), excluding memories used as challenged-trait conflict evidence.

```text
For each candidate memory n:
    sup_w, con_w = 0, 0

    For each pooled node p (p ≠ n, p ∈ {s, m, t}):
        ordinary = multi-hop sign(p, n)
        if ordinary == SUP:  sup_w += 1
        elif ordinary == CON: con_w += 1

    # Memory nodes do not have SHIFT_TO edges — no shift signal.

    support_ratio(n) = sup_w / (sup_w + con_w)    # default 0.5 if no evidence
    final_score(n, q) = w_sr · support_ratio(n) + (1 - w_sr) · seed_score(n, q)
```

For expansion memories not in `m_seed`, `seed_score(n, q)` is computed at this point using the same formula as seed retrieval (Section 2.4).

Select top-`k_m_final` → `m_f`.

---

## 5. Final Set

```text
{t_final, s_final, m_final, s_aps}
```

---

## 6. Serialization

The final set is serialized into prompt-ready sections:

```text
[Current Constraints]
[Traits]
[Challenged Traits]
[Relevant States]
[Relevant Memories]
[Recent Conversation]
```

### Section Sources

| Section | Source |
|---------|--------|
| Current Constraints | `s_aps` |
| Traits | stable traits from `t_final` |
| Challenged Traits | challenged traits from `t_final` with nested conflict evidence |
| Relevant States | `s_final` |
| Relevant Memories | `m_final` |
| Recent Conversation | context cache |

### Notes

- `Recent Conversation` is always available for response prompt construction.
- For QA answering, `Recent Conversation` is controlled by config and is off by default.
- Shift-chain collapse is applied when forming `t_final`, `s_final`, and `s_aps`.

---

## 7. Retrieval Summary

GraphMem retrieval combines:
- **unified seed scoring** with semantic similarity and query-side normalized overlap, using scope-dependent weights,
- **1-hop contradiction pull** to surface direct conflicting evidence,
- **SHIFT_TO forward traversal** from older nodes to newer nodes during expansion,
- **bidirectional SHIFT_TO** in support-ratio computation: old node receives `con_w += 1` (penalized), new node receives `sup_w += 1` (boosted),
- **uniform final scoring** across all node types: `w_sr · support_ratio + (1 - w_sr) · seed_score`,
- **cross-type evidence**: all node types serve as evidence for all other node types,
- **shift-chain collapse** for traits, states, and APS,
- **APS** for implicit persona reasoning: top-`k_aps` HIGH-impact, non-`SHIFT_TO`-source states ranked by `seed_score`. APS and `s_seed` are disjoint partitions of the state pool.

This yields:
- query-relevant high-impact states through APS, with HIGH non-APS states still able to surface in `s_seed`,
- conflict-aware trait handling with stable/challenged classification,
- newer state and trait recovery from outdated seeds,
- evidence-scored memory selection from the full pool,
- and controlled pruning without globally deleting older evidence.
