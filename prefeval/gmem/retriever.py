"""Retriever for GraphMem."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from graph_store import (
    HeterogeneousGraph,
    Node,
    NODE_C,
    NODE_E,
    NODE_S,
    NODE_T,
    EVID_SUPPORT,
    EVID_CONTRADICT,
    EVID_IRRELEVANT,
    SCOPE_BROAD,
    SCOPE_NARROW,
    PRIORITY_HIGH,
    format_elapsed_str,
    embed_text,
    extract_kw_nouns,
)


@dataclass
class RetrievalResult:
    active_persona: List[Node]
    traits_stable: List[Node]
    traits_challenged: List[Node]
    states_conflict: List[Node]
    episodes_conflict: List[Node]
    states_relevant: List[Node]
    episodes_relevant: List[Node]
    conflict_per_trait: Dict[str, Dict[str, Any]]
    serialized: str
    all_final_nodes: List[Node]
    seed_contexts: List[Node]
    seed_episodes: List[Node]
    seed_states: List[Node]
    seed_traits: List[Node]
    pool_nodes: List[Node]
    node_scores: Dict[str, float]  # node_id → seed score (0.0 for expansion-only nodes)


class GraphRetriever:
    """Implements the GraphMem retrieval pipeline."""

    def __init__(self, graph: HeterogeneousGraph, embed_model, spacy_nlp, config):
        self._graph = graph
        self._embed = embed_model
        self._nlp = spacy_nlp
        self._cfg = config
        # Per-retrieve memoization of get_ordinary_signed_reachable; cleared
        # at the top of every retrieve() (graph is immutable during a call).
        self._signed_reach_cache: Dict[Tuple[str, int], Dict[str, str]] = {}

    def set_graph(self, graph: HeterogeneousGraph) -> None:
        self._graph = graph

    def _get_signed_reachable(self, start_id: str, hop_cap: int) -> Dict[str, str]:
        key = (start_id, hop_cap)
        cached = self._signed_reach_cache.get(key)
        if cached is not None:
            return cached
        result = self._graph.get_ordinary_signed_reachable(start_id, hop_cap)
        self._signed_reach_cache[key] = result
        return result

    def retrieve(
        self,
        query: str,
        global_turn: int,
        context_cache_str: str,
        current_conv_id: int,
        current_turn_id: int,
        for_qa: bool = False,
    ) -> RetrievalResult:
        self._signed_reach_cache = {}

        q_emb = self._embed_text(query)
        q_kw = self._extract_kw(query)
        q_kw_set = {kw.strip().lower() for kw in q_kw if isinstance(kw, str) and kw.strip()}

        seed_c, seed_m, seed_s, seed_t, aps, node_scores = self._seed_retrieval(q_emb, q_kw_set)
        pool = self._graph_expansion(seed_c, seed_m, seed_s, seed_t, aps)

        signed_cache = {
            nid: self._get_signed_reachable(nid, self._cfg.SIGN_PROP_HOP_CAP)
            for nid in pool
        }

        result = self._final_set_assembly(
            pool=pool,
            signed_cache=signed_cache,
            seed_m=seed_m,
            aps=aps,
            q_emb=q_emb,
            q_kw_set=q_kw_set,
            context_cache_str=context_cache_str,
            current_conv_id=current_conv_id,
            current_turn_id=current_turn_id,
            seed_c=seed_c,
            seed_s=seed_s,
            seed_t=seed_t,
            node_scores=node_scores,
            for_qa=for_qa,
        )

        self._graph.increment_retrieval_count([n.node_id for n in result.all_final_nodes])
        return result

    # ------------------------------------------------------------------
    # Step 1: seed retrieval
    # ------------------------------------------------------------------

    def _seed_retrieval(
        self,
        q_emb: np.ndarray,
        q_kw_set: Set[str],
    ) -> Tuple[List[Node], List[Node], List[Node], List[Node], List[Node], Dict[str, float]]:
        cfg = self._cfg

        all_c = self._graph.get_nodes_by_type(NODE_C)
        scored_c = [(self._seed_score(n, q_emb, q_kw_set), n) for n in all_c]
        seed_c = _top_k(scored_c, cfg.K_CONTEXT)

        all_m = self._graph.get_nodes_by_type(NODE_E)
        scored_m = [(self._seed_score(n, q_emb, q_kw_set), n) for n in all_m]
        seed_m = _top_k(scored_m, cfg.K_EPISODE)

        all_t = self._graph.get_nodes_by_type(NODE_T)
        scored_t = [(self._seed_score(n, q_emb, q_kw_set), n) for n in all_t]
        seed_t = _top_k(scored_t, cfg.K_TRAIT)

        # APS = top-K_APS HIGH-priority, non-SHIFT_TO-source states by seed_score.
        # s_seed is drawn from the remaining states; HIGH states beyond rank
        # K_APS fall through into s_seed.
        all_s = self._graph.get_nodes_by_type(NODE_S)
        scored_s = [(self._seed_score(n, q_emb, q_kw_set), n) for n in all_s]
        seed_s, aps = self._partition_states_into_aps_and_seed(scored_s)

        node_scores: Dict[str, float] = {}
        for s, n in scored_c + scored_m + scored_s + scored_t:
            node_scores[n.node_id] = s

        return seed_c, seed_m, seed_s, seed_t, aps, node_scores

    def _partition_states_into_aps_and_seed(
        self,
        scored_s: List[Tuple[float, Node]],
    ) -> Tuple[List[Node], List[Node]]:
        """Build APS (top-K_APS HIGH-priority, non-SHIFT_TO-source states by
        seed_score) and s_seed (top-K_STATE from the rest). Disjoint partitions."""
        cfg = self._cfg

        if cfg.K_APS > 0:
            aps_candidates = [
                (score, n) for score, n in scored_s
                if n.recall_priority == PRIORITY_HIGH
                and (
                    not cfg.APS_EXCLUDE_SHIFT_SOURCE
                    or not self._graph.get_shift_to_out(n.node_id)
                )
            ]
            aps_candidates.sort(key=lambda x: x[0], reverse=True)
            aps = [n for _, n in aps_candidates[: cfg.K_APS]]
        else:
            aps = []

        aps_ids = {n.node_id for n in aps}
        seed_pool = [(score, n) for score, n in scored_s if n.node_id not in aps_ids]
        seed_s = _top_k(seed_pool, cfg.K_STATE)
        return seed_s, aps

    def _seed_score(self, node: Node, q_emb: np.ndarray, q_kw_set: Set[str]) -> float:
        w_sem, w_ov = self._weights_for(node)
        sem = _cos_sim(q_emb, node.embedding)
        ov = _overlap_norm(q_kw_set, _label_set(node))
        return w_sem * sem + w_ov * ov

    def _weights_for(self, node: Node) -> Tuple[float, float]:
        cfg = self._cfg
        if node.node_type == NODE_C:
            return cfg.W_SEM_C, cfg.W_OV_C
        if node.scope == SCOPE_NARROW:
            return cfg.W_SEM_NARROW, cfg.W_OV_NARROW
        return cfg.W_SEM_BROAD, cfg.W_OV_BROAD

    # ------------------------------------------------------------------
    # Step 2: graph expansion
    # ------------------------------------------------------------------

    def _graph_expansion(
        self,
        seed_c: List[Node],
        seed_m: List[Node],
        seed_s: List[Node],
        seed_t: List[Node],
        aps: List[Node],
    ) -> Dict[str, Node]:
        pool: Dict[str, Node] = {}
        for n in seed_c + seed_m + seed_s + seed_t + aps:
            pool[n.node_id] = n

        # Step 1: seed c → SOURCE-derived m/s/t; then drop c nodes from the pool
        for c_node in seed_c:
            for child in self._graph.get_source_children(c_node.node_id):
                if child.node_type in {NODE_E, NODE_S, NODE_T} and child.node_id not in pool:
                    pool[child.node_id] = child
        for c_node in seed_c:
            pool.pop(c_node.node_id, None)

        # Pull ± SEED_TURN_NEIGHBOR_DELTA turn neighbors of seed_s/seed_m
        # (S/M, same conv_id) into the pool as plain members — recall safety
        # net since state extraction is sparse (≤1 per turn). These are NOT
        # expansion drivers.
        turn_neighbor_ids: Set[str] = set()
        if getattr(self._cfg, "SEED_TURN_NEIGHBOR_ENABLED", True):
            delta = getattr(self._cfg, "SEED_TURN_NEIGHBOR_DELTA", 1)
            if delta > 0:
                for seed_node in list(seed_s) + list(seed_m):
                    for neighbor in self._graph.get_turn_neighbors(
                        seed_node.node_id, types={NODE_S, NODE_E}, delta=delta
                    ):
                        if neighbor.node_id not in pool:
                            pool[neighbor.node_id] = neighbor
                            turn_neighbor_ids.add(neighbor.node_id)

        # Expansion origins exclude APS members (query-independent persona
        # slot) and turn-neighbors (recall safety net only); both still
        # contribute SUP/CON signals to other nodes through _cross_type_sup_con.
        aps_ids = {n.node_id for n in aps}
        origin_ids = [
            nid for nid in pool.keys()
            if nid not in aps_ids and nid not in turn_neighbor_ids
        ]
        for nid in origin_ids:
            self._expand_from(nid, pool)

        return pool

    def _expand_from(self, origin_id: str, pool: Dict[str, Node]) -> None:
        """Add nodes reachable from origin_id under sign propagation to the pool."""
        signed = self._get_signed_reachable(origin_id, self._cfg.SIGN_PROP_HOP_CAP)
        for reached_id, _sign in signed.items():
            node = self._graph.get_node(reached_id)
            if node is None or node.node_type not in {NODE_E, NODE_S, NODE_T}:
                continue
            if reached_id not in pool:
                pool[reached_id] = node

    # ------------------------------------------------------------------
    # Step 3: final set assembly
    # ------------------------------------------------------------------

    def _final_set_assembly(
        self,
        pool: Dict[str, Node],
        signed_cache: Dict[str, Dict[str, str]],
        seed_m: List[Node],
        aps: List[Node],
        q_emb: np.ndarray,
        q_kw_set: Set[str],
        context_cache_str: str,
        current_conv_id: int,
        current_turn_id: int,
        seed_c: List[Node],
        seed_s: List[Node],
        seed_t: List[Node],
        node_scores: Dict[str, float],
        for_qa: bool = False,
    ) -> RetrievalResult:
        cfg = self._cfg
        evidence_nodes = [n for n in pool.values() if n.node_type in {NODE_S, NODE_E, NODE_T}]
        pool_traits = [n for n in pool.values() if n.node_type == NODE_T]
        pool_states = [n for n in pool.values() if n.node_type == NODE_S]
        pool_episodes = [n for n in pool.values() if n.node_type == NODE_E]

        # ---- Trait selection: top-k from pooled traits using unified final score ----
        scored_traits: List[Tuple[float, int, Node]] = []  # (final_score, evidence_count, trait)
        trait_scores: Dict[str, Tuple[float, int]] = {}
        for trait in pool_traits:
            sup, con = self._cross_type_sup_con(trait, evidence_nodes, signed_cache)
            ev_count = sup + con
            ratio = sup / ev_count if ev_count > 0 else 0.5
            seed = self._seed_score_cached(trait, q_emb, q_kw_set, node_scores)
            final = cfg.W_SR * ratio + (1.0 - cfg.W_SR) * seed
            scored_traits.append((final, ev_count, trait))
            trait_scores[trait.node_id] = (ratio, ev_count)
            node_scores[trait.node_id] = final

        scored_traits.sort(key=lambda x: x[0], reverse=True)
        t_final = [t for _, _, t in scored_traits[: cfg.K_T_FINAL]]

        # Trait classification within selected top-k.
        # A trait is challenged if either (a) it has CONTRADICT evidence with
        # ratio ≤ τ, or (b) it has an outgoing SHIFT_TO into a pooled trait
        # (it has been replaced). Both signals surface as separate sub-bullets
        # in serialization.
        traits_stable: List[Node] = []
        traits_challenged: List[Node] = []
        # conflict_per_trait[trait_id] = {
        #     "shifted_to": [Node],         # newer traits this trait shifted into
        #     "states":     [(Node, rel)],  # CONTRADICT state evidence
        #     "episodes":   [(Node, rel)],  # CONTRADICT episode evidence
        # }
        conflict_per_trait: Dict[str, Dict[str, Any]] = {}
        states_conflict_map: Dict[str, Node] = {}
        episodes_conflict_map: Dict[str, Node] = {}
        pool_trait_ids = {t.node_id for t in pool_traits}

        for trait in t_final:
            ratio, ev_count = trait_scores[trait.node_id]
            shifted_to_ids = [
                tid for tid in self._graph.get_shift_to_out(trait.node_id)
                if tid in pool_trait_ids
            ]
            shifted_to_nodes = [self._graph.get_node(tid) for tid in shifted_to_ids]
            shifted_to_nodes = [n for n in shifted_to_nodes if n is not None]

            has_shifted = len(shifted_to_nodes) > 0
            has_contradict = ev_count > 0 and ratio <= cfg.TRAIT_VALIDATION_TAU

            if has_shifted or has_contradict:
                traits_challenged.append(trait)
                trait_state_conflicts: List[Tuple[Node, str]] = []
                trait_episode_conflicts: List[Tuple[Node, str]] = []
                for s_node in pool_states:
                    rel = signed_cache.get(s_node.node_id, {}).get(trait.node_id)
                    if rel == EVID_CONTRADICT:
                        trait_state_conflicts.append((s_node, rel))
                        states_conflict_map[s_node.node_id] = s_node
                for e_node in pool_episodes:
                    rel = signed_cache.get(e_node.node_id, {}).get(trait.node_id)
                    if rel == EVID_CONTRADICT:
                        trait_episode_conflicts.append((e_node, rel))
                        episodes_conflict_map[e_node.node_id] = e_node
                conflict_per_trait[trait.node_id] = {
                    "shifted_to": shifted_to_nodes,
                    "states": trait_state_conflicts,
                    "episodes": trait_episode_conflicts,
                }
            else:
                traits_stable.append(trait)

        # ---- State selection ----
        aps_ids = {n.node_id for n in aps}
        state_conflict_ids = set(states_conflict_map)
        candidate_states = [
            n for n in pool_states
            if n.node_id not in aps_ids and n.node_id not in state_conflict_ids
        ]
        scored_states: List[Tuple[float, Node]] = []
        for n in candidate_states:
            sup, con = self._cross_type_sup_con(n, evidence_nodes, signed_cache)
            ratio = sup / (sup + con) if (sup + con) > 0 else 0.5
            seed = self._seed_score_cached(n, q_emb, q_kw_set, node_scores)
            final = cfg.W_SR * ratio + (1.0 - cfg.W_SR) * seed
            scored_states.append((final, n))
            node_scores[n.node_id] = final
        states_relevant_raw = _top_k(scored_states, cfg.K_SF)

        if cfg.ENABLE_SHIFT_CHAIN_PRUNING:
            states_relevant = self._shift_chain_collapse(states_relevant_raw)
            aps = self._shift_chain_collapse(aps)
        else:
            states_relevant = states_relevant_raw

        # ---- Episode selection ----
        episode_conflict_ids = set(episodes_conflict_map)
        candidate_episodes = [
            n for n in pool_episodes
            if n.node_id not in episode_conflict_ids
        ]
        scored_episodes: List[Tuple[float, Node]] = []
        for n in candidate_episodes:
            sup, con = self._cross_type_sup_con(n, evidence_nodes, signed_cache)
            ratio = sup / (sup + con) if (sup + con) > 0 else 0.5
            seed = self._seed_score_cached(n, q_emb, q_kw_set, node_scores)
            final = cfg.W_SR * ratio + (1.0 - cfg.W_SR) * seed
            scored_episodes.append((final, n))
            node_scores[n.node_id] = final
        episodes_relevant = _top_k(scored_episodes, cfg.K_EPISODE_FINAL)

        if cfg.ENABLE_SHIFT_CHAIN_PRUNING:
            t_final_collapsed = self._shift_chain_collapse(t_final)
            t_final_ids_kept = {t.node_id for t in t_final_collapsed}
            traits_stable = [t for t in traits_stable if t.node_id in t_final_ids_kept]
            traits_challenged = [t for t in traits_challenged if t.node_id in t_final_ids_kept]
            conflict_per_trait = {tid: cv for tid, cv in conflict_per_trait.items() if tid in t_final_ids_kept}

        serialized = self._serialize(
            aps=aps,
            traits_stable=traits_stable,
            traits_challenged=traits_challenged,
            states_relevant=states_relevant,
            episodes_relevant=episodes_relevant,
            conflict_per_trait=conflict_per_trait,
            context_cache_str=context_cache_str,
            current_conv_id=current_conv_id,
            current_turn_id=current_turn_id,
            for_qa=for_qa,
        )

        all_final_nodes = _dedupe_nodes(
            aps
            + traits_stable
            + traits_challenged
            + list(states_conflict_map.values())
            + list(episodes_conflict_map.values())
            + states_relevant
            + episodes_relevant
        )

        return RetrievalResult(
            active_persona=aps,
            traits_stable=traits_stable,
            traits_challenged=traits_challenged,
            states_conflict=list(states_conflict_map.values()),
            episodes_conflict=list(episodes_conflict_map.values()),
            states_relevant=states_relevant,
            episodes_relevant=episodes_relevant,
            conflict_per_trait=conflict_per_trait,
            serialized=serialized,
            all_final_nodes=all_final_nodes,
            seed_contexts=seed_c,
            seed_episodes=seed_m,
            seed_states=seed_s,
            seed_traits=seed_t,
            pool_nodes=list(pool.values()),
            node_scores=node_scores,
        )

    def _cross_type_sup_con(
        self,
        target: Node,
        evidence_nodes: List[Node],
        signed_cache: Dict[str, Dict[str, str]],
    ) -> Tuple[int, int]:
        """Count (sup, con) signals reaching `target` from pooled S/M/T evidence.
        SHIFT_TO contributions are already folded into signed_cache via
        _ordinary_evidence_neighbors, so they are not double-counted here."""
        sup = 0
        con = 0
        for n in evidence_nodes:
            if n.node_id == target.node_id:
                continue
            rel = signed_cache.get(n.node_id, {}).get(target.node_id)
            if rel == EVID_SUPPORT:
                sup += 1
            elif rel == EVID_CONTRADICT:
                con += 1
        return sup, con

    def _seed_score_cached(
        self,
        node: Node,
        q_emb: np.ndarray,
        q_kw_set: Set[str],
        node_scores: Dict[str, float],
    ) -> float:
        cached = node_scores.get(node.node_id)
        if cached is not None:
            return cached
        score = self._seed_score(node, q_emb, q_kw_set)
        node_scores[node.node_id] = score
        return score

    def _shift_chain_collapse(self, nodes: List[Node]) -> List[Node]:
        if not nodes:
            return nodes
        node_ids = {n.node_id for n in nodes}
        to_drop: Set[str] = set()
        for node in nodes:
            forward = self._graph.get_shift_forward_reachable(node.node_id)
            if forward & node_ids:
                to_drop.add(node.node_id)
        return [n for n in nodes if n.node_id not in to_drop]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize(
        self,
        aps: List[Node],
        traits_stable: List[Node],
        traits_challenged: List[Node],
        states_relevant: List[Node],
        episodes_relevant: List[Node],
        conflict_per_trait: Dict[str, Dict[str, Any]],
        context_cache_str: str,
        current_conv_id: int,
        current_turn_id: int,
        for_qa: bool = False,
    ) -> str:
        sections: List[str] = []
        seen_content: Set[str] = set()

        def add_lines(title: str, lines: List[str]) -> None:
            # Skip empty sections entirely to reduce prompt noise.
            if not lines:
                return
            section_lines = [f"[{title}]"]
            section_lines.extend(lines)
            sections.append("\n".join(section_lines))

        aps_lines = []
        for node in aps:
            key = node.content.strip()
            if key in seen_content:
                continue
            seen_content.add(key)
            aps_lines.append(f"[{self._elapsed_str(node, current_conv_id, current_turn_id)}] {node.content}")
        add_lines("Current Constraints", aps_lines)

        trait_lines = []
        for node in traits_stable:
            key = node.content.strip()
            if key in seen_content:
                continue
            seen_content.add(key)
            trait_lines.append(f"[{self._elapsed_str(node, current_conv_id, current_turn_id)}] {node.content}")
        add_lines("Traits", trait_lines)

        challenged_lines = []
        for trait in traits_challenged:
            key = trait.content.strip()
            if key in seen_content:
                continue
            seen_content.add(key)
            challenged_lines.append(
                f"[{self._elapsed_str(trait, current_conv_id, current_turn_id)}] {trait.content}"
            )
            conflicts = conflict_per_trait.get(
                trait.node_id, {"shifted_to": [], "states": [], "episodes": []}
            )
            local_seen: Set[str] = set()
            # Sub-bullet 1: shifted_to (newer trait that replaces this one)
            for newer in conflicts.get("shifted_to", []):
                snippet = newer.content.strip()
                if snippet in local_seen:
                    continue
                local_seen.add(snippet)
                challenged_lines.append(
                    f"  ↳ shifted to: [{self._elapsed_str(newer, current_conv_id, current_turn_id)}] {newer.content}"
                )
            # Sub-bullet 2: conflicting evidence (states + episodes with CONTRADICT)
            for node, _relation in conflicts.get("states", []) + conflicts.get("episodes", []):
                snippet = node.content.strip()
                if snippet in local_seen:
                    continue
                local_seen.add(snippet)
                challenged_lines.append(
                    f"  ↳ conflicting evidence: [{self._elapsed_str(node, current_conv_id, current_turn_id)}] {node.content}"
                )
        add_lines("Challenged Traits", challenged_lines)

        state_lines = []
        for node in states_relevant:
            key = node.content.strip()
            if key in seen_content:
                continue
            seen_content.add(key)
            state_lines.append(
                f"[{self._elapsed_str(node, current_conv_id, current_turn_id)}] {node.content}"
            )
        add_lines("Relevant States", state_lines)

        episode_lines = []
        for node in episodes_relevant:
            key = node.content.strip()
            if key in seen_content:
                continue
            seen_content.add(key)
            episode_lines.append(f"[{self._elapsed_str(node, current_conv_id, current_turn_id)}] {node.content}")
        add_lines("Relevant Episodes", episode_lines)

        if not for_qa or self._cfg.INCLUDE_RECENT_CONVERSATION_FOR_QA:
            convo_lines = context_cache_str.splitlines() if context_cache_str else []
            add_lines("Recent Conversation", convo_lines)

        return "\n\n".join(sections)

    def _elapsed_str(self, node: Node, current_conv_id: int, current_turn_id: int) -> str:
        return format_elapsed_str(
            node.conv_id, node.turn_id,
            current_conv_id, current_turn_id,
            self._cfg.TIME_PER_CONV_ID_HOURS,
            self._cfg.TIME_PER_TURN_MINUTES,
        )

    def _embed_text(self, text: str) -> np.ndarray:
        return embed_text(self._embed, text)

    def _extract_kw(self, text: str) -> List[str]:
        return extract_kw_nouns(self._nlp, text)


def _top_k(scored: List[Tuple[float, Node]], k: int) -> List[Node]:
    if k <= 0:
        return []
    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    return [node for _, node in scored_sorted[:k]]


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.dot(a, b))


def _label_set(node: Node) -> Set[str]:
    if node.node_type == NODE_C:
        base = node.keywords
    else:
        base = list(node.keywords) + list(node.domain_label)
    return {x.strip().lower() for x in base if isinstance(x, str) and x.strip()}


def _overlap_norm(q_kw_set: Set[str], label_set: Set[str]) -> float:
    if not q_kw_set:
        return 0.0
    inter = len(q_kw_set & label_set)
    return inter / max(len(q_kw_set), 1)


def _dedupe_nodes(nodes: List[Node]) -> List[Node]:
    seen = set()
    result = []
    for node in nodes:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        result.append(node)
    return result
