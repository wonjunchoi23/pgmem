"""Graph storage for PGMem."""

import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# Constants
# =============================================================================

NODE_C = "c"
NODE_E = "e"
NODE_S = "s"
NODE_T = "t"

EDGE_SOURCE = "SOURCE"
EDGE_EVIDENCE = "EVIDENCE"

EVID_SUPPORT = "SUPPORT"
EVID_CONTRADICT = "CONTRADICT"
EVID_SHIFT_TO = "SHIFT_TO"
EVID_IRRELEVANT = "IRRELEVANT"

SCOPE_BROAD = "BROAD"
SCOPE_NARROW = "NARROW"

PRIORITY_HIGH = "HIGH"
PRIORITY_LOW = "LOW"

NODE_TYPES = {NODE_C, NODE_E, NODE_S, NODE_T}
EVID_SUBTYPES = {EVID_SUPPORT, EVID_CONTRADICT, EVID_SHIFT_TO, EVID_IRRELEVANT}
SCOPES = {SCOPE_BROAD, SCOPE_NARROW}
PRIORITY_LEVELS = {PRIORITY_HIGH, PRIORITY_LOW}

EVID_ORDINARY = {EVID_SUPPORT, EVID_CONTRADICT, EVID_IRRELEVANT}

_EVID_EXPAND_OUT: Dict[str, Set[str]] = {
    NODE_C: set(),
    NODE_E: {NODE_S, NODE_E, NODE_T},
    NODE_S: {NODE_S, NODE_T},
    NODE_T: {NODE_T},
}


# =============================================================================
# Data classes
# =============================================================================

@dataclass(eq=False)
class Node:
    node_id: str
    node_type: str
    content: str
    keywords: List[str]
    embedding: np.ndarray
    created_at: int
    session_id: int
    conv_id: int
    turn_id: int
    domain_label: List[str] = field(default_factory=list)
    scope: Optional[str] = None
    recall_priority: Optional[str] = None
    retrieval_count: int = 1

    def __hash__(self) -> int:
        return hash(self.node_id)


class HeterogeneousGraph:
    """In-memory graph for one PGMem session."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._creation_order: List[str] = []
        self._src_out: Dict[str, Set[str]] = {}
        self._src_in: Dict[str, Set[str]] = {}
        self._evid_out: Dict[str, Dict[str, str]] = {}
        self._evid_in: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def new_node_id() -> str:
        return str(uuid.uuid4())

    def add_node(self, node: Node) -> None:
        node.keywords, node.domain_label = deduplicate_labels(
            node.keywords, node.domain_label
        )
        nid = node.node_id
        self._nodes[nid] = node
        self._creation_order.append(nid)
        self._src_out[nid] = set()
        self._src_in[nid] = set()
        self._evid_out[nid] = {}
        self._evid_in[nid] = {}

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def all_nodes(self) -> List[Node]:
        return [self._nodes[nid] for nid in self._creation_order if nid in self._nodes]

    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        return [n for n in self.all_nodes() if n.node_type == node_type]

    def get_turn_neighbors(
        self,
        node_id: str,
        types: Set[str],
        delta: int = 1,
    ) -> List[Node]:
        """Nodes of `types` within ≤ `delta` turns of `node_id` in the same `conv_id`.
        Excludes the anchor itself; never crosses `conv_id` boundaries (a
        conv_id jump corresponds to ~12h under the time model).
        """
        anchor = self._nodes.get(node_id)
        if anchor is None or delta <= 0:
            return []
        conv_id = anchor.conv_id
        turn_id = anchor.turn_id
        result: List[Node] = []
        for nid in self._creation_order:
            if nid == node_id:
                continue
            n = self._nodes.get(nid)
            if n is None or n.node_type not in types:
                continue
            if n.conv_id != conv_id:
                continue
            if abs(n.turn_id - turn_id) > delta or n.turn_id == turn_id:
                continue
            result.append(n)
        return result

    def get_recent_domain_labels(self, limit: int = 5) -> List[str]:
        seen = set()
        result: List[str] = []
        for node in reversed(self.all_nodes()):
            for label in reversed(node.domain_label):
                norm = label.strip().lower()
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                result.append(norm)
                if len(result) >= limit:
                    return result
        return result

    def add_source_edge(self, src_id: str, dst_id: str) -> None:
        if src_id not in self._nodes or dst_id not in self._nodes:
            return
        self._src_out[src_id].add(dst_id)
        self._src_in[dst_id].add(src_id)

    def get_source_children(self, node_id: str) -> List[Node]:
        return [self._nodes[nid] for nid in self._src_out.get(node_id, set()) if nid in self._nodes]

    def get_source_parents(self, node_id: str) -> List[Node]:
        return [self._nodes[nid] for nid in self._src_in.get(node_id, set()) if nid in self._nodes]

    def add_evidence_edge(self, src_id: str, dst_id: str, subtype: str) -> None:
        if src_id not in self._nodes or dst_id not in self._nodes:
            return
        if subtype not in EVID_SUBTYPES:
            return
        if dst_id in self._evid_out.get(src_id, {}):
            return

        self._evid_out[src_id][dst_id] = subtype
        self._evid_in[dst_id][src_id] = subtype

    def has_direct_edge(self, node_a: str, node_b: str) -> bool:
        """Return True if any direct evidence edge exists between the two nodes (either direction)."""
        return (
            node_b in self._evid_out.get(node_a, {})
            or node_a in self._evid_out.get(node_b, {})
        )

    def get_evidence_out_raw(self, node_id: str) -> Dict[str, str]:
        return dict(self._evid_out.get(node_id, {}))

    def get_evidence_in_raw(self, node_id: str) -> Dict[str, str]:
        return dict(self._evid_in.get(node_id, {}))

    def get_evidence_subtype(self, src_id: str, dst_id: str) -> Optional[str]:
        return self._evid_out.get(src_id, {}).get(dst_id)

    def get_shift_to_out(self, node_id: str) -> List[str]:
        """Return node IDs that node_id points to via SHIFT_TO (node_id is the old/source node)."""
        return [
            dst_id for dst_id, sub in self._evid_out.get(node_id, {}).items()
            if sub == EVID_SHIFT_TO
        ]

    def get_shift_to_in(self, node_id: str) -> List[str]:
        """Return node IDs that point to node_id via SHIFT_TO (node_id is the new/target node)."""
        return [
            src_id for src_id, sub in self._evid_in.get(node_id, {}).items()
            if sub == EVID_SHIFT_TO
        ]

    def get_shift_forward_reachable(self, start_id: str) -> Set[str]:
        """Traverse SHIFT_TO edges in the stored (old → new) direction and return all reachable node IDs."""
        if start_id not in self._nodes:
            return set()
        visited: Set[str] = set()
        queue = deque(self.get_shift_to_out(start_id))
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(t for t in self.get_shift_to_out(nid) if t not in visited)
        return visited

    def _ordinary_evidence_neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """Return (neighbor_id, subtype) pairs reachable via evidence edges.

        SHIFT_TO is folded into sign propagation:
        - Same-type outgoing SHIFT_TO → SUP (flow forward to the newer node).
        - Cross-type outgoing SHIFT_TO → CON (source invalidates the target).
        - Any incoming SHIFT_TO (same- or cross-type) → CON (halt under SUP×CON rule).
        """
        node = self._nodes.get(node_id)
        if node is None:
            return []

        valid_out_types = _EVID_EXPAND_OUT.get(node.node_type, set())
        seen: Set[str] = set()
        result: List[Tuple[str, str]] = []

        for dst_id, subtype in self._evid_out.get(node_id, {}).items():
            dst_node = self._nodes.get(dst_id)
            if dst_node is None:
                continue
            if subtype in EVID_ORDINARY:
                if dst_node.node_type not in valid_out_types:
                    continue
                mapped = subtype
            elif subtype == EVID_SHIFT_TO:
                if dst_node.node_type == node.node_type:
                    mapped = EVID_SUPPORT   # same-type: forward to newer
                else:
                    mapped = EVID_CONTRADICT  # cross-type: source invalidates target
            else:
                continue
            if dst_id in seen:
                continue
            seen.add(dst_id)
            result.append((dst_id, mapped))

        for src_id, subtype in self._evid_in.get(node_id, {}).items():
            src_node = self._nodes.get(src_id)
            if src_node is None:
                continue
            if subtype in EVID_ORDINARY:
                if src_node.node_type != node.node_type:
                    continue
                mapped = subtype
            elif subtype == EVID_SHIFT_TO:
                mapped = EVID_CONTRADICT  # same-type: older invalidated; cross-type: invalidator reached
            else:
                continue
            if src_id in seen:
                continue
            seen.add(src_id)
            result.append((src_id, mapped))

        return result

    def get_ordinary_signed_reachable(self, start_id: str, hop_cap: int) -> Dict[str, str]:
        """Multi-hop sign propagation over evidence edges.

        Sign table: SUP×SUP → SUP (continue), SUP×CON → CON (stop), CON nodes
        do not expand further. Shortest path wins; on ties at the same hop,
        CON wins over SUP. Within one call, a node is visited at most once.

        Returns {node_id: relation} where relation ∈ {SUPPORT, CONTRADICT}.
        """
        if start_id not in self._nodes or hop_cap <= 0:
            return {}

        result: Dict[str, str] = {}
        frontier: List[Tuple[str, str]] = [(start_id, EVID_SUPPORT)]  # (node_id, cum_sign)

        for hop in range(hop_cap):
            if not frontier:
                break

            sup_candidates: Dict[str, str] = {}
            con_candidates: Dict[str, str] = {}

            for node_id, cum_sign in frontier:
                if cum_sign != EVID_SUPPORT:
                    continue
                for neighbor_id, edge_sub in self._ordinary_evidence_neighbors(node_id):
                    if neighbor_id == start_id or edge_sub == EVID_IRRELEVANT:
                        continue
                    if neighbor_id in result:
                        continue
                    if edge_sub == EVID_SUPPORT:
                        sup_candidates.setdefault(neighbor_id, EVID_SUPPORT)
                    elif edge_sub == EVID_CONTRADICT:
                        con_candidates.setdefault(neighbor_id, EVID_CONTRADICT)

            # CON written first so it wins on ties; CON nodes do not expand.
            next_frontier: List[Tuple[str, str]] = []
            for nid, sign in con_candidates.items():
                result[nid] = sign
            for nid, sign in sup_candidates.items():
                if nid in result:
                    continue
                result[nid] = sign
                next_frontier.append((nid, sign))

            frontier = next_frontier

        return result

    def increment_retrieval_count(self, node_ids: List[str]) -> None:
        for nid in node_ids:
            if nid in self._nodes:
                self._nodes[nid].retrieval_count += 1

    def clear(self) -> None:
        self._nodes.clear()
        self._creation_order.clear()
        self._src_out.clear()
        self._src_in.clear()
        self._evid_out.clear()
        self._evid_in.clear()

    def save_snapshot(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        nodes_data = {}
        for nid in self._creation_order:
            node = self._nodes[nid]
            nodes_data[nid] = {
                "node_type": node.node_type,
                "content": node.content,
                "keywords": node.keywords,
                "domain_label": node.domain_label,
                "scope": node.scope,
                "recall_priority": node.recall_priority,
                "created_at": node.created_at,
                "session_id": node.session_id,
                "conv_id": node.conv_id,
                "turn_id": node.turn_id,
                "retrieval_count": node.retrieval_count,
            }

        graph_data = {
            "creation_order": self._creation_order,
            "nodes": nodes_data,
            "src_out": {k: list(v) for k, v in self._src_out.items()},
            "evid_out": self._evid_out,
        }
        with open(directory / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=True, indent=2)

        if self._creation_order:
            embeddings = np.stack([self._nodes[nid].embedding for nid in self._creation_order])
        else:
            embeddings = np.empty((0,), dtype=np.float32)
        np.save(str(directory / "graph_embeddings.npy"), embeddings)

        metadata = {
            "node_counts": self.node_count_by_type(),
            "source_edges": self.edge_count_source(),
            "evidence_edges": self.edge_count_evidence_by_subtype(),
        }
        with open(directory / "graph_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)

    def load_snapshot(self, directory: Path) -> None:
        directory = Path(directory)
        json_path = directory / "graph.json"
        if not json_path.exists():
            return

        with open(json_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        creation_order = graph_data["creation_order"]
        nodes_data = graph_data["nodes"]
        src_out = graph_data["src_out"]
        evid_out = graph_data["evid_out"]

        emb_path = directory / "graph_embeddings.npy"
        if emb_path.exists() and creation_order:
            embeddings = np.load(str(emb_path))
        else:
            embeddings = np.empty((0, 1), dtype=np.float32)

        self._nodes = {}
        self._creation_order = creation_order
        for idx, nid in enumerate(creation_order):
            d = nodes_data[nid]
            emb = embeddings[idx] if idx < len(embeddings) else np.zeros(1, dtype=np.float32)
            self._nodes[nid] = Node(
                node_id=nid,
                node_type=d["node_type"],
                content=d["content"],
                keywords=d["keywords"],
                domain_label=d.get("domain_label", []),
                embedding=emb,
                created_at=d["created_at"],
                session_id=d["session_id"],
                conv_id=d["conv_id"],
                turn_id=d["turn_id"],
                scope=d.get("scope"),
                recall_priority=d.get("recall_priority"),
                retrieval_count=d.get("retrieval_count", 1),
            )

        self._src_out = {nid: set(targets) for nid, targets in src_out.items()}
        self._evid_out = {nid: dict(targets) for nid, targets in evid_out.items()}

        self._src_in = {nid: set() for nid in self._nodes}
        for src, targets in self._src_out.items():
            for dst in targets:
                self._src_in.setdefault(dst, set()).add(src)

        self._evid_in = {nid: {} for nid in self._nodes}
        for src, targets in self._evid_out.items():
            for dst, subtype in targets.items():
                self._evid_in.setdefault(dst, {})[src] = subtype

    def node_count_by_type(self) -> Dict[str, int]:
        counts = {NODE_C: 0, NODE_E: 0, NODE_S: 0, NODE_T: 0}
        for node in self._nodes.values():
            counts[node.node_type] += 1
        return counts

    def edge_count_source(self) -> int:
        return sum(len(v) for v in self._src_out.values())

    def edge_count_evidence_by_subtype(self) -> Dict[str, int]:
        counts = {
            EVID_SUPPORT: 0,
            EVID_CONTRADICT: 0,
            EVID_SHIFT_TO: 0,
            EVID_IRRELEVANT: 0,
        }
        for dst_map in self._evid_out.values():
            for subtype in dst_map.values():
                counts[subtype] += 1
        return counts


# =============================================================================
# Shared utility functions
# =============================================================================

def _format_compact_duration(elapsed_minutes: float) -> str:
    """Render a duration in compact form: now / Nmin / Nh / Nd / Nmo / Ny.

    Buckets:
      < 1 min            → "now"
      < 60 min           → "{N}min"
      < 24 h             → "{N}h"
      < 30 d             → "{N}d"
      < 12 mo (≈ 365 d)  → "{N}mo"
      ≥ 1 y              → "{N}y"
    """
    if elapsed_minutes < 1:
        return "now"
    if elapsed_minutes < 60:
        return f"{int(elapsed_minutes)}min"
    elapsed_hours = elapsed_minutes / 60.0
    if elapsed_hours < 24:
        return f"{int(elapsed_hours)}h"
    elapsed_days = elapsed_hours / 24.0
    if elapsed_days < 30:
        return f"{int(elapsed_days)}d"
    elapsed_months = elapsed_days / 30.0
    if elapsed_months < 12:
        return f"{int(elapsed_months)}mo"
    elapsed_years = elapsed_days / 365.0
    return f"{int(elapsed_years)}y"


def format_elapsed_str(
    entry_conv_id: int,
    entry_turn_id: int,
    current_conv_id: int,
    current_turn_id: int,
    time_per_conv_id_hours: float,
    time_per_turn_minutes: float,
) -> str:
    """Convert conv/turn delta to a compact relative-time string ("now",
    "Nmin ago", "Nh ago", "Nd ago", "Nmo ago", "Ny ago").
    """
    delta_conv = max(current_conv_id - entry_conv_id, 0)
    delta_turn = max(current_turn_id - entry_turn_id, 0) if delta_conv == 0 else 0
    elapsed_minutes = (
        delta_conv * time_per_conv_id_hours * 60.0
        + delta_turn * time_per_turn_minutes
    )
    compact = _format_compact_duration(elapsed_minutes)
    if compact == "now":
        return "now"
    return f"{compact} ago"


def format_conv_gap(
    conv_id_a: int,
    conv_id_b: int,
    time_per_conv_id_hours: float,
) -> str:
    """Compact-duration string for the absolute gap between two conv_ids.
    Used by ⑤d's "Relation context" rendering. Returns the duration without
    a trailing "ago" (the caller phrases the surrounding text).
    """
    delta_conv = abs(conv_id_a - conv_id_b)
    elapsed_minutes = delta_conv * time_per_conv_id_hours * 60.0
    return _format_compact_duration(elapsed_minutes)


def embed_text(embed_model, text: str) -> np.ndarray:
    """Encode text using the given sentence-transformers model."""
    return embed_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def deduplicate_labels(
    keywords: List[str],
    domain_label: List[str],
) -> Tuple[List[str], List[str]]:
    """Drop any domain_label token that case-insensitively matches a keyword
    (keywords win on overlap). Returns new lists; inputs are not mutated."""
    if not keywords or not domain_label:
        return list(keywords), list(domain_label)
    kw_lower = {k.lower() for k in keywords if isinstance(k, str)}
    filtered = [d for d in domain_label if not (isinstance(d, str) and d.lower() in kw_lower)]
    return list(keywords), filtered


def extract_kw_nouns(nlp, text: str) -> List[str]:
    """Extract lowercased lemma nouns from text using spaCy noun chunks."""
    doc = nlp(text)
    return list({
        chunk.root.lemma_.lower()
        for chunk in doc.noun_chunks
        if chunk.root.lemma_.strip()
    })
