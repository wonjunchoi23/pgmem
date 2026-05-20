"""Updater for GraphMem v6 on LoComo."""

import copy
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from graph_store import (
    HeterogeneousGraph,
    Node,
    NODE_E,
    NODE_S,
    NODE_T,
    EVID_SUPPORT,
    EVID_CONTRADICT,
    EVID_SHIFT_TO,
    EVID_IRRELEVANT,
    SCOPE_BROAD,
    SCOPE_NARROW,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    format_elapsed_str,
    format_session_gap,
    embed_text,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PendingLLMCall:
    call_type: str
    system_prompt: str
    user_prompt: str
    guided_json: Dict
    max_tokens: int
    log_dir: str
    created_at: int
    anchor_session_id: int
    anchor_turn_idx: int
    anchor_timestamp_seconds: float
    session_id: int
    expected_judgment_count: int = 0
    id_map: List[str] = field(default_factory=list)
    id_map_b: List[str] = field(default_factory=list)
    expected_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class TurnRecord:
    speaker: str
    text: str
    session_id: int
    turn_idx: int
    timestamp_seconds: float
    dia_id: str
    context_node_id: str


@dataclass
class ChunkRecord:
    session_id: int
    turns: List[TurnRecord] = field(default_factory=list)
    context_ids: List[str] = field(default_factory=list)
    state_ids: List[str] = field(default_factory=list)
    episode_id: Optional[str] = None
    last_turn_idx: int = 0
    last_timestamp_seconds: float = 0.0


# =============================================================================
# Prompt components
# =============================================================================

_NODE_TYPE_DESC = (
    "Node types:\n"
    "  State:   A speaker-specific condition that is currently or recently valid and may change over time. It captures the speaker's present stance, ongoing goal, constraint, situation, or preference shift. States are time-bounded and context-sensitive.\n"
    "  Trait:   A generalized speaker characteristic that persists across situations and time. It represents recurring dispositions, stable preferences, values, or habitual tendencies. Traits are cross-situational and relatively context-independent.\n"
    "  Episode: A summary of what happened during a recent conversation — concrete events, discussed topics, and actions at a particular time. Not a generalized persona attribute."
)

_EVID_DESC_FULL = (
    "Evidence relationships:\n"
    "  SUPPORT: The two pieces of information are consistent or mutually reinforcing. Use this only when one piece provides independent evidential force for the other.\n"
    "\n"
    "  CONTRADICT: The two pieces of information are in clear tension, but they do not necessarily form a temporal replacement. Both may still be meaningful evidence.\n"
    "\n"
    "  SHIFT_TO: A same-type temporal transition where the older information has changed into newer information and is no longer currently valid. Use only for true old → new replacement in state-state or trait-trait pairs. When emitting SHIFT_TO, the earlier node is the source old node and the later node is the target new node.\n"
    "\n"
    "  IRRELEVANT: The pair was judged and found unrelated, OR the pair is merely topically related without one piece providing independent evidential force for or against the other."
)

_EVID_DESC_REDUCED = (
    "Evidence relationships:\n"
    "  SUPPORT: The two pieces of information are consistent or mutually reinforcing. Use this only when one piece provides independent evidential force for the other.\n"
    "  CONTRADICT: The two pieces of information are in clear tension or conflict. Do not use this for merely different topics, weak associations, or facts that can naturally coexist.\n"
    "  IRRELEVANT: The pair was judged and found unrelated, OR the pair is merely topically related without one piece providing independent evidential force for or against the other."
)

_LABEL_DISCIPLINE_BLOCK = (
    "keywords:\n"
    "  Surface-level tokens from the source text, or close lexical variants.\n"
    "  Use concrete entities, named items, specific actions, constraints, or particular phrases.\n"
    "  Do not use broad topical categories here.\n"
    "\n"
    "domain_label:\n"
    "  Abstract topical or categorical labels at a higher level of abstraction.\n"
    "  Use broader subject areas or mid-level categories that could group related episodes across different surface wording.\n"
    "\n"
    "Hard constraints (both fields):\n"
    "  Each item is one continuous word with no whitespace. Compounds like \"machinelearning\" are fine; items with whitespace are dropped at storage time.\n"
    "  Each label string MUST appear in at most one of keywords or domain_label.\n"
    "  A domain_label may overlap lexically with a keyword only when it expresses a clearly broader topical category, not a simple restatement.\n"
    "  Drop any domain_label that merely restates or narrowly rephrases a keyword."
)

SYS_STATE_EXTRACT = (
    "You are a persona state extraction assistant.\n"
    "Extract a speaker state from the current speaker turn.\n"
    "Do not classify relationships in this call.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC
)

SYS_EPISODE_EXTRACT = (
    "You are an episode extraction assistant.\n"
    "Summarize the recent conversation between two named speakers into one episode.\n"
    "Do not classify relationships in this call.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC
)

SYS_TRAIT_EXTRACT = (
    "You are a persona trait extraction assistant.\n"
    "Infer at most one new long-term trait for one speaker from the accumulated recent evidence.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC
)

SYS_TRAIT_EVIDENCE_5A = (
    "You are an evidence classification assistant.\n"
    "Classify the direct relationship between a newly extracted trait and nearby existing nodes.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_TRAIT_EXTRA_REL_5B = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships between a new trait and additional candidate nodes retrieved from the graph.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_REDUCED
)

SYS_STATE_STATE_5C = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships for candidate state-state pairs.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_STATE_EPISODE_5D = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships for candidate state-episode pairs.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_REDUCED
)

SYS_STATE_NEW_REL = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships involving newly extracted states: pairs among the new states themselves, and pairs between each new state and each previously extracted state.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_EPISODE_NEW_REL = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships involving a newly extracted episode: the pair (new_episode, previous_episode) when a previous episode exists, and one pair (new_episode, chunk_state_i) for each chunk state.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_REDUCED
)


# =============================================================================
# JSON schemas
# =============================================================================

_JUDGMENT_OBJECT = {
    "type": "object",
    "properties": {
        "source_id": {"type": "integer"},
        "target_id": {"type": "integer"},
        "relation": {"type": "string"},
    },
    "required": ["source_id", "target_id", "relation"],
    "additionalProperties": False,
}

_STATE_EPISODE_JUDGMENT_OBJECT = {
    "type": "object",
    "properties": {
        "state_id": {"type": "integer"},
        "episode_id": {"type": "integer"},
        "relation": {"type": "string"},
    },
    "required": ["state_id", "episode_id", "relation"],
    "additionalProperties": False,
}

STATE_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "states": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "domain_label": {"type": "array", "items": {"type": "string"}},
                    "scope": {"type": "string"},
                    "recall_priority": {"type": "string"},
                },
                "required": [
                    "id", "content", "keywords", "domain_label",
                    "scope", "recall_priority",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["states"],
    "additionalProperties": False,
}

EPISODE_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "episode": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "domain_label": {"type": "array", "items": {"type": "string"}},
                "scope": {"type": "string"},
            },
            "required": ["id", "content", "keywords", "domain_label", "scope"],
            "additionalProperties": False,
        },
    },
    "required": ["episode"],
    "additionalProperties": False,
}

TRAIT_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "traits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "speaker": {"type": "string"},
                    "content": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "domain_label": {"type": "array", "items": {"type": "string"}},
                    "scope": {"type": "string"},
                },
                "required": ["id", "speaker", "content", "keywords", "domain_label", "scope"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["traits"],
    "additionalProperties": False,
}

JUDGMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": _JUDGMENT_OBJECT,
        },
    },
    "required": ["judgments"],
    "additionalProperties": False,
}

STATE_EPISODE_JUDGMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": _STATE_EPISODE_JUDGMENT_OBJECT,
        },
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


# =============================================================================
# Updater
# =============================================================================

class GraphUpdater:
    """Updater for GraphMem v5 internal calls ②③④⑤ on LoComo."""

    _STAT_COUNTERS: Tuple[Tuple[str, str], ...] = (
        ("_state_extracted_count", "num_state_nodes_extracted"),
        ("_episode_extracted_count", "num_episode_nodes_extracted"),
        ("_trait_extracted_count", "num_trait_nodes_extracted"),
        ("_5a_call_count", "num_5a_trait_evidence_calls"),
        ("_5b_call_count", "num_5b_trait_extra_rel_calls"),
        ("_5c_call_count", "num_5c_state_state_rel_calls"),
        ("_5d_call_count", "num_5d_state_episode_rel_calls"),
        ("_state_new_rel_call_count", "num_state_new_rel_calls"),
        ("_episode_new_rel_call_count", "num_episode_new_rel_calls"),
    )

    def _reset_stat_counters(self) -> None:
        for attr, _ in self._STAT_COUNTERS:
            setattr(self, attr, 0)

    def __init__(
        self,
        graph: HeterogeneousGraph,
        llm_client,
        embed_model,
        model_path: str,
        config,
        llm_logger=None,
        context_cache=None,
    ):
        self._graph = graph
        self._llm = llm_client
        self._embed_model = embed_model
        self._model_path = model_path
        self._cfg = config
        self._llm_logger = llm_logger
        self._context_cache = context_cache

        self._current_chunk: Optional[ChunkRecord] = None
        self._completed_chunks: deque = deque(maxlen=config.TRAIT_EXTRACTION_CHUNKS)
        self._recent_turns: deque = deque(maxlen=config.STATE_EXTRACTION_H)
        self._previous_state_ids: List[str] = []
        self._previous_episode_id: Optional[str] = None
        self._previous_trait_id: Optional[str] = None

        self._user_turn_count = 0   # counter of registered turns (one speaker turn = one tick)
        self._chunk_count = 0

        self._call_type_tokens: Dict[str, Dict[str, int]] = {}

        self._reset_stat_counters()

        self._pending_trait_node_id: Optional[str] = None
        self._pending_state_new_rel_new_ids: List[str] = []
        self._pending_episode_new_rel: Optional[Dict[str, Any]] = None

        self._pending_new_state_ids_5c: Set[str] = set()
        self._pending_new_state_ids_5d: Set[str] = set()
        self._pending_new_episode_ids_5d: Set[str] = set()

    # ------------------------------------------------------------------
    # Public state helpers
    # ------------------------------------------------------------------

    def ensure_current_chunk(self, session_id: int) -> None:
        if self._current_chunk is None:
            self._current_chunk = ChunkRecord(session_id=session_id)

    def register_turn(
        self,
        speaker: str,
        text: str,
        session_id: int,
        turn_idx: int,
        context_node_id: str,
        timestamp_seconds: float,
        dia_id: str = "",
    ) -> None:
        self.ensure_current_chunk(session_id)
        turn = TurnRecord(
            speaker=speaker,
            text=text,
            session_id=session_id,
            turn_idx=turn_idx,
            timestamp_seconds=timestamp_seconds,
            dia_id=dia_id,
            context_node_id=context_node_id,
        )
        self._current_chunk.turns.append(turn)
        self._current_chunk.context_ids.append(context_node_id)
        self._current_chunk.last_turn_idx = turn_idx
        self._current_chunk.last_timestamp_seconds = timestamp_seconds
        self._recent_turns.append(turn)
        self._user_turn_count += 1

    # ------------------------------------------------------------------
    # Step-wise preparation
    # ------------------------------------------------------------------

    def prepare_pre_turn_call(
        self,
        session_id: int,
        turn_idx: int,
        global_turn: int,
        anchor_timestamp_seconds: float,
    ) -> Optional[PendingLLMCall]:
        # Chunk boundary = session boundary. If the incoming session differs
        # from the current chunk's session, finalize the current chunk via ③.
        if self._current_chunk is None:
            self.ensure_current_chunk(session_id)
            return None
        if session_id == self._current_chunk.session_id:
            return None
        if not self._current_chunk.turns:
            self._current_chunk = ChunkRecord(session_id=session_id)
            return None
        return self._build_episode_call(
            chunk=self._current_chunk,
            session_id=session_id,
            created_at=global_turn,
            anchor_session_id=session_id,
            anchor_turn_idx=turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
        )

    def prepare_post_turn_call(
        self,
        session_id: int,
        global_turn: int,
        current_turn_idx: int,
        current_timestamp_seconds: float,
    ) -> Optional[PendingLLMCall]:
        if self._user_turn_count == 0:
            return None
        if self._user_turn_count % self._cfg.STATE_EXTRACTION_H != 0:
            return None
        if not self._recent_turns:
            return None
        return self._build_state_call(
            session_id=session_id,
            created_at=global_turn,
            anchor_session_id=session_id,
            anchor_turn_idx=current_turn_idx,
            anchor_timestamp_seconds=current_timestamp_seconds,
        )

    def prepare_finalize_call(
        self,
        session_id: int,
        global_turn: int,
        anchor_timestamp_seconds: float,
    ) -> Optional[PendingLLMCall]:
        if self._current_chunk is None or not self._current_chunk.turns:
            return None
        return self._build_episode_call(
            chunk=self._current_chunk,
            session_id=session_id,
            created_at=global_turn,
            anchor_session_id=self._current_chunk.session_id,
            anchor_turn_idx=self._current_chunk.last_turn_idx,
            anchor_timestamp_seconds=self._current_chunk.last_timestamp_seconds,
        )

    def apply_irrelevant_fallback(self, call: PendingLLMCall) -> None:
        if not call.expected_pairs:
            return
        for src_id, dst_id in call.expected_pairs:
            if src_id is None or dst_id is None or src_id == dst_id:
                continue
            if self._graph.get_node(src_id) is None or self._graph.get_node(dst_id) is None:
                continue
            if self._graph.has_direct_edge(src_id, dst_id):
                continue
            self._store_evidence_edge(src_id, dst_id, EVID_IRRELEVANT)

    _RESULT_HANDLERS: Dict[str, str] = {
        "state": "_after_state",
        "state_new_rel": "_after_state_new_rel",
        "episode": "_after_episode",
        "episode_new_rel": "_after_episode_new_rel",
        "trait": "_after_trait",
        "trait_evidence_5a": "_after_trait_evidence_5a",
        "trait_extra_rel_5b": "_after_trait_extra_rel_5b",
        "state_state_5c": "_after_state_state_5c",
        "state_episode_5d": "_after_state_episode_5d",
    }

    def apply_call_result(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        handler_name = self._RESULT_HANDLERS.get(call.call_type)
        if handler_name is None:
            return None
        return getattr(self, handler_name)(call, result)

    def _after_state(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        self._apply_state_result(call, result)
        new_state_ids = list(self._previous_state_ids)
        prev_state_ids = self._get_recent_prev_state_ids(
            exclude_ids=set(new_state_ids),
            k=self._cfg.STATE_NEW_REL_PREV_WINDOW,
        )
        n_new = len(new_state_ids)
        n_prev = len(prev_state_ids)
        if n_new >= 2 or (n_new >= 1 and n_prev >= 1):
            return self._build_state_new_rel_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
                previous_state_ids=prev_state_ids,
                new_state_ids=new_state_ids,
            )
        return None

    def _after_state_new_rel(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        self._apply_state_new_rel_result(result, call.id_map)
        return None

    def _after_episode(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        old_prev_episode_id = self._previous_episode_id
        self._apply_episode_result(call, result)
        new_episode_id = self._previous_episode_id
        if (
            new_episode_id
            and new_episode_id != old_prev_episode_id
            and self._completed_chunks
        ):
            chunk_state_ids = list(self._completed_chunks[-1].state_ids)
            prev_id_for_call = old_prev_episode_id if old_prev_episode_id else None
            if chunk_state_ids or prev_id_for_call:
                return self._build_episode_new_rel_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_session_id=call.anchor_session_id,
                    anchor_turn_idx=call.anchor_turn_idx,
                    anchor_timestamp_seconds=call.anchor_timestamp_seconds,
                    new_episode_id=new_episode_id,
                    previous_episode_id=prev_id_for_call,
                    chunk_state_ids=chunk_state_ids,
                )
        if self._chunk_count > 0 and self._chunk_count % self._cfg.TRAIT_EXTRACTION_CHUNKS == 0:
            return self._build_trait_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        return None

    def _after_episode_new_rel(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        self._apply_episode_new_rel_result(result, call.id_map)
        if self._chunk_count > 0 and self._chunk_count % self._cfg.TRAIT_EXTRACTION_CHUNKS == 0:
            return self._build_trait_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        return None

    def _after_trait(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        created = self._apply_trait_result(call, result)
        if created is not None:
            self._pending_trait_node_id = created
            return self._build_5a_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._pending_new_state_ids_5c:
            return self._build_5c_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and (self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d):
            return self._build_5d_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        return None

    def _after_trait_evidence_5a(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        self._apply_5a_result(result, call.id_map)
        if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._pending_trait_node_id:
            return self._build_5b_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        self._pending_trait_node_id = None
        if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._pending_new_state_ids_5c:
            return self._build_5c_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and (self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d):
            return self._build_5d_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        return None

    def _after_trait_extra_rel_5b(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        self._apply_5b_result(result, call.id_map)
        self._pending_trait_node_id = None
        if self._pending_new_state_ids_5c:
            return self._build_5c_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        if (self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d):
            return self._build_5d_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        return None

    def _after_state_state_5c(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        self._apply_5c_result(result, call.id_map)
        if (self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d):
            return self._build_5d_call(
                session_id=call.session_id,
                created_at=call.created_at,
                anchor_session_id=call.anchor_session_id,
                anchor_turn_idx=call.anchor_turn_idx,
                anchor_timestamp_seconds=call.anchor_timestamp_seconds,
            )
        return None

    def _after_state_episode_5d(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        self._apply_5d_result(result, call.id_map, call.id_map_b)
        return None

    # ------------------------------------------------------------------
    # Sequential wrapper
    # ------------------------------------------------------------------

    def execute_call(self, call: PendingLLMCall) -> Dict:
        max_retry = max(int(getattr(self._cfg, "JUDGMENT_RETRY", 0)), 0)
        last_result: Dict = {}
        for attempt in range(max_retry + 1):
            user_prompt = call.user_prompt
            if attempt > 0 and call.expected_judgment_count > 0:
                hint = (f"\nPrevious attempt returned empty judgments; "
                        f"you MUST output exactly {call.expected_judgment_count} judgments.")
                user_prompt = user_prompt + hint
            try:
                result = self._llm.generate(
                    prompt=user_prompt,
                    system_prompt=call.system_prompt,
                    guided_json=call.guided_json,
                    temperature=self._cfg.TEMPERATURE,
                    max_tokens=call.max_tokens,
                    json_retry=self._cfg.JSON_RETRY,
                    return_usage=True,
                )
                if self._llm_logger is not None:
                    self._llm_logger.log(call.log_dir, call.system_prompt, user_prompt, result)
                prompt_tokens, completion_tokens = _extract_token_counts(result)
                self.accumulate_usage(prompt_tokens, completion_tokens, call_type=call.log_dir)
                last_result = result if isinstance(result, dict) else {}
            except Exception as exc:
                logger.error(f"{call.call_type} call failed: {exc}")
                last_result = {}

            if call.expected_judgment_count <= 0:
                return last_result
            judgments = last_result.get("judgments", []) if isinstance(last_result, dict) else []
            if isinstance(judgments, list) and len(judgments) > 0:
                return last_result
            if attempt < max_retry:
                logger.warning(
                    "%s returned empty judgments (expected %d); retrying (attempt %d/%d)",
                    call.call_type, call.expected_judgment_count,
                    attempt + 1, max_retry,
                )
        return last_result

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def accumulate_usage(self, input_tokens: int, output_tokens: int, call_type: str, llm_calls: int = 1) -> None:
        if call_type not in self._call_type_tokens:
            self._call_type_tokens[call_type] = {"input": 0, "output": 0, "llm_calls": 0}
        self._call_type_tokens[call_type]["input"] += input_tokens
        self._call_type_tokens[call_type]["output"] += output_tokens
        self._call_type_tokens[call_type]["llm_calls"] += llm_calls

    def get_and_reset_internal_tokens(self) -> Dict[str, Dict[str, int]]:
        result = dict(self._call_type_tokens)
        self._call_type_tokens = {}
        return result

    def get_and_reset_internal_stats(self) -> Dict:
        stats = {key: getattr(self, attr) for attr, key in self._STAT_COUNTERS}
        self._reset_stat_counters()
        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._current_chunk = None
        self._completed_chunks.clear()
        self._recent_turns.clear()
        self._previous_state_ids = []
        self._previous_episode_id = None
        self._previous_trait_id = None
        self._pending_trait_node_id = None
        self._user_turn_count = 0
        self._chunk_count = 0
        self._call_type_tokens = {}
        self._reset_stat_counters()
        self._pending_state_new_rel_new_ids = []
        self._pending_episode_new_rel = None
        self._pending_new_state_ids_5c = set()
        self._pending_new_state_ids_5d = set()
        self._pending_new_episode_ids_5d = set()

    def set_graph(self, graph: HeterogeneousGraph) -> None:
        self._graph = graph

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Pending new-node tracking + pair-level top-K
    # ------------------------------------------------------------------

    def _pair_topk_union(
        self,
        pairs: List[Tuple[Node, Node]],
        top_k: int,
    ) -> List[Tuple[Node, Node]]:
        """sem_topK ∪ lex_topK over candidate pairs, deduped (sem-sorted first)."""
        if not pairs or top_k <= 0:
            return []

        def _sem(p: Tuple[Node, Node]) -> float:
            return float(np.dot(p[0].embedding, p[1].embedding))

        def _lex(p: Tuple[Node, Node]) -> int:
            kw_a = set(p[0].keywords) | {l.lower() for l in p[0].domain_label}
            kw_b = set(p[1].keywords) | {l.lower() for l in p[1].domain_label}
            return len(kw_a & kw_b)

        sem_sorted = sorted(pairs, key=_sem, reverse=True)[:top_k]
        lex_sorted = sorted(pairs, key=_lex, reverse=True)[:top_k]
        seen: Set[Tuple[str, str]] = set()
        result: List[Tuple[Node, Node]] = []
        for a, b in sem_sorted + lex_sorted:
            key = (a.node_id, b.node_id)
            if key in seen:
                continue
            seen.add(key)
            result.append((a, b))
        return result

    def _update_ss_reservoir(self, new_state_ids: List[str]) -> None:
        """Record new state IDs; pair top-K is computed at ⑤c flush time."""
        if not self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION:
            return
        self._pending_new_state_ids_5c.update(new_state_ids)

    def _update_sm_reservoir(self, new_state_ids: List[str], new_episode_ids: List[str]) -> None:
        """Record new state/episode IDs; pair top-K is computed at ⑤d flush time."""
        if not self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION:
            return
        self._pending_new_state_ids_5d.update(new_state_ids)
        self._pending_new_episode_ids_5d.update(new_episode_ids)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_state_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
    ) -> PendingLLMCall:
        recent_turns = list(self._recent_turns)
        if not recent_turns:
            current_turn_block = "None"
            current_turn = None
            current_session_id = anchor_session_id
            current_turn_idx = anchor_turn_idx
            speaker_label = "the speaker"
        else:
            current_turn = recent_turns[-1]
            current_turn_block = self._format_turns([current_turn], anchor_timestamp_seconds)
            current_session_id = current_turn.session_id
            current_turn_idx = current_turn.turn_idx
            speaker_label = current_turn.speaker

        prior_turns = self._context_cache.get_prior_turns(
            current_session_id=current_session_id,
            current_turn_idx=current_turn_idx,
            n=self._cfg.STATE_REF_CONTEXT_TURNS,
        ) if self._context_cache is not None else []
        prior_context_block = self._format_prior_turns(prior_turns, anchor_timestamp_seconds)

        prompt = (
            "A state does not have to be permanently stable. If the current turn reveals\n"
            "a useful persona-relevant signal but there is not yet enough evidence to call\n"
            "it a long-term trait, extract it as a state. Later stages may generalize\n"
            "repeated or stable states into traits.\n"
            "\n"
            "Do not extract if the information is only:\n"
            "  - a greeting, thanks, or conversational behavior,\n"
            "  - a description of the other speaker rather than the current speaker,\n"
            "  - a transient emotion with no effect on an active task, decision, constraint, or safety issue.\n"
            "\n"
            "Task:\n"
            f"Extract up to {self._cfg.STATE_MAX_COUNT} persona state(s) revealed by {speaker_label} in the CURRENT TURN.\n"
            f"If the current turn contains no new persona-relevant signal from {speaker_label}, return an empty states list.\n"
            "Do not invent or speculate.\n"
            "\n"
            "Use the prior context only as read-only background for resolving references in the current turn.\n"
            f"Do NOT extract a state if the condition is stated only by another speaker and is not expressed or clearly implied by {speaker_label}.\n"
            "\n"
            "Each state must:\n"
            f"- begin with \"{speaker_label}\",\n"
            "- be a single concise sentence,\n"
            "- avoid raw episodic narration unless it directly functions as a current condition or useful persona signal.\n"
            "\n"
            "State metadata:\n"
            "scope:\n"
            f"  BROAD  : a currently valid persona signal, condition, value-driven stance, health-related limitation, lifestyle constraint, role, or preference of {speaker_label} that may affect decisions across unrelated topics.\n"
            f"  NARROW : a currently valid preference, goal, condition, or constraint of {speaker_label} tied mainly to the current task, topic, or short-term situation.\n"
            "\n"
            "If uncertain between BROAD and NARROW, choose NARROW.\n"
            "\n"
            "recall_priority:\n"
            f"  HIGH : the assistant must actively remember this state about {speaker_label} right now.\n"
            f"         Use HIGH only when BOTH are true:\n"
            f"         (1) {speaker_label} would reasonably expect this to be remembered without re-stating it,\n"
            "         (2) ignoring it would cause a response that is clearly wrong, unsafe,\n"
            "             or noticeably frustrating.\n"
            "         Prefer HIGH for persistent constraints, hard restrictions, safety-relevant\n"
            "         conditions, urgent deadlines, or explicit standing expectations.\n"
            "         Be conservative with short-lived turn-specific preferences.\n"
            "  LOW  : useful persona context, but the response would still be appropriate\n"
            "         and acceptable without it.\n"
            "\n"
            "If uncertain between HIGH and LOW, choose LOW.\n"
            "\n"
            f"{_LABEL_DISCIPLINE_BLOCK}\n"
            "\n"
            "Also provide for each state:\n"
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific single-token nouns (no whitespace, no multi-word phrases)\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} single-token topical labels (no whitespace, no multi-word phrases)\n"
            "\n"
            "[CURRENT TURN — extract state(s) about the current speaker from this turn only]\n"
            f"{current_turn_block}\n"
            "\n"
            "[PRIOR CONTEXT — for disambiguation only.\n"
            " These turns may be from earlier in this session or earlier sessions.\n"
            " Use them only to clarify what the CURRENT TURN refers to.\n"
            " DO NOT extract states from these turns.]\n"
            f"{prior_context_block}\n"
            "\n"
            "Assign placeholder ids: new_0, new_1, ... in order of extraction."
        )
        schema = copy.deepcopy(STATE_EXTRACT_SCHEMA)
        schema["properties"]["states"]["maxItems"] = self._cfg.STATE_MAX_COUNT
        return PendingLLMCall(
            call_type="state",
            system_prompt=SYS_STATE_EXTRACT,
            user_prompt=prompt,
            guided_json=schema,
            max_tokens=self._cfg.MAX_TOKENS_STATE,
            log_dir="call_2_state",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=0,
        )

    def _build_episode_call(
        self,
        chunk: ChunkRecord,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
    ) -> PendingLLMCall:
        prompt = (
            "Create exactly one episode node summarizing what happened in the recent conversation between the two speakers.\n"
            "The episode must:\n"
            "- be 1–2 sentences,\n"
            "- summarize concrete events, discussed topics, actions, and developments,\n"
            "- mention both speakers by name when both participated,\n"
            "- describe the episode itself, not generalized persona traits.\n"
            "\n"
            "Do NOT include chunk-level state extractions or restate persona traits unless\n"
            "they are part of the episode itself. The state nodes are extracted by a\n"
            "separate pipeline; this call summarizes only the conversation.\n"
            "\n"
            "Episode metadata:\n"
            "scope:\n"
            "  BROAD  : the episode reveals or confirms a cross-topic characteristic of one or both speakers (e.g., a health event, a major life decision, a value-revealing exchange, or a standing constraint reaffirmed).\n"
            "  NARROW : the episode is self-contained within the current topic or task — its implications do not extend beyond the current conversation thread.\n"
            "\n"
            "If uncertain between BROAD and NARROW, choose NARROW.\n"
            "\n"
            f"{_LABEL_DISCIPLINE_BLOCK}\n"
            "\n"
            "Also provide:\n"
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific single-token nouns (no whitespace, no multi-word phrases)\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} single-token topical labels (no whitespace, no multi-word phrases)\n"
            "\n"
            "Assign the episode the placeholder id: new_episode.\n"
            "\n"
            "[Recent Conversation]\n"
            f"{self._format_turns(chunk.turns, anchor_timestamp_seconds)}"
        )
        return PendingLLMCall(
            call_type="episode",
            system_prompt=SYS_EPISODE_EXTRACT,
            user_prompt=prompt,
            guided_json=EPISODE_EXTRACT_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_EPISODE,
            log_dir="call_3_episode",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=0,
        )

    def _build_trait_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
    ) -> PendingLLMCall:
        recent_chunks = list(self._completed_chunks)
        turns: List[TurnRecord] = []
        for chunk in recent_chunks:
            turns.extend(chunk.turns)

        conversations_block = self._format_turns(turns, anchor_timestamp_seconds)

        prompt = (
            "Use the \"in general\" test: a trait should still be true if you asked the speaker about themselves \"in general\" with no specific time, place, or context attached.\n"
            "\n"
            "Extract 0 or 1 trait about ONE of the two speakers from the recent conversations below.\n"
            "Each trait must:\n"
            "- begin with the speaker's name (e.g., \"Caroline ...\"),\n"
            "- be 2–3 complete sentences,\n"
            "- be inferable from the recent conversations as a likely persistent pattern, not just a one-time event or momentary feeling.\n"
            "\n"
            "If the recent conversations do not reveal a clear new persistent pattern for either speaker, output 0 traits.\n"
            "\n"
            "Trait metadata:\n"
            "speaker: the name of the speaker the trait describes.\n"
            "scope:\n"
            "  BROAD  : the trait applies across all domains of the speaker's life — it would shape the speaker's approach regardless of the subject being discussed\n"
            "  NARROW : a domain-specific tendency or preference mainly expressed within a particular area (e.g., a consistent style in one hobby or activity)\n"
            "\n"
            "If uncertain between BROAD and NARROW, choose NARROW.\n"
            "\n"
            f"{_LABEL_DISCIPLINE_BLOCK}\n"
            "\n"
            "Also provide:\n"
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific single-token nouns (no whitespace, no multi-word phrases)\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} single-token topical labels (no whitespace, no multi-word phrases)\n"
            "\n"
            "Assign the trait the placeholder id: new_trait.\n"
            "\n"
            "[Recent Two Conversations]\n"
            f"{conversations_block}"
        )
        schema = copy.deepcopy(TRAIT_EXTRACT_SCHEMA)
        schema["properties"]["traits"]["maxItems"] = self._cfg.TRAIT_MAX_COUNT
        return PendingLLMCall(
            call_type="trait",
            system_prompt=SYS_TRAIT_EXTRACT,
            user_prompt=prompt,
            guided_json=schema,
            max_tokens=self._cfg.MAX_TOKENS_TRAIT,
            log_dir="call_4_trait",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
        )

    def _build_5a_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
    ) -> PendingLLMCall:
        recent_chunks = list(self._completed_chunks)
        state_ids: List[str] = []
        episode_ids: List[str] = []
        for chunk in recent_chunks:
            state_ids.extend(chunk.state_ids)
            if chunk.episode_id:
                episode_ids.append(chunk.episode_id)

        trait = self._graph.get_node(self._pending_trait_node_id) if self._pending_trait_node_id else None
        new_trait_block, trait_ids = self._format_node_list_indexed(
            [trait.node_id] if trait else [], anchor_timestamp_seconds, start_idx=0,
        )
        recent_states_block, s_ids = self._format_node_list_indexed(
            state_ids, anchor_timestamp_seconds, start_idx=len(trait_ids),
        )
        recent_episodes_block, e_ids = self._format_node_list_indexed(
            episode_ids, anchor_timestamp_seconds, start_idx=len(trait_ids) + len(s_ids),
        )
        previous_trait_block, pt_ids = self._format_node_list_indexed(
            [self._previous_trait_id] if self._previous_trait_id else [],
            anchor_timestamp_seconds,
            start_idx=len(trait_ids) + len(s_ids) + len(e_ids),
        )
        id_map = trait_ids + s_ids + e_ids + pt_ids
        new_trait_idx = 0
        prev_trait_idx = len(trait_ids) + len(s_ids) + len(e_ids) if pt_ids else None
        expected_5a = len(s_ids) + len(e_ids) + (1 if pt_ids else 0)

        prompt = (
            "You will judge direct evidence relationships between a newly extracted trait and nearby existing nodes.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"Index {new_trait_idx} is the new trait. "
            f"Indices {len(trait_ids)}..{len(trait_ids)+len(s_ids)-1} are recent states. "
            f"Indices {len(trait_ids)+len(s_ids)}..{len(trait_ids)+len(s_ids)+len(e_ids)-1} are recent episodes. "
            + (f"Index {prev_trait_idx} is the previous trait.\n" if prev_trait_idx is not None else "\n")
            + "\n"
            "Judge:\n"
            "- each (recent state, new_trait) pair, in the listed order;\n"
            "- each (recent episode, new_trait) pair, in the listed order;\n"
            "- the (previous_trait, new_trait) pair, when a previous trait exists.\n"
            "\n"
            "Direction rule:\n"
            f"- For (recent_state, new_trait) and (recent_episode, new_trait): source_id = state/episode index, target_id = {new_trait_idx}.\n"
            + (f"- For (previous_trait, new_trait) SHIFT_TO: source_id MUST be {prev_trait_idx} (previous trait) and target_id MUST be {new_trait_idx} (new trait).\n"
               if prev_trait_idx is not None else "")
            + "\n"
            "Relation rules:\n"
            "- For state↔trait and episode↔trait: use only SUPPORT | CONTRADICT | IRRELEVANT.\n"
            "- For previous_trait↔new_trait: use SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT. SHIFT_TO is allowed only for a true old_trait → new_trait replacement.\n"
            "\n"
            "SUPPORT calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- For (state, new_trait): use SUPPORT only when the state describes a current/recent condition that the new trait would naturally generalize from or predict. Sharing a domain or keyword is not enough.\n"
            "- For (episode, new_trait): use SUPPORT only when the episode captures concrete speaker behavior that the trait would predict; the episode must add evidential force beyond mere topic overlap.\n"
            "- For (previous_trait, new_trait): use SUPPORT only when the two traits independently describe overlapping or compatible persistent characteristics.\n"
            "- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "CONTRADICT calibration:\n"
            "- Use CONTRADICT only when the candidate clearly conflicts with the new trait at the persona level.\n"
            "- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the trait.\n"
            "- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "SHIFT_TO caution (only for previous_trait↔new_trait):\n"
            "- Use SHIFT_TO only when the previous trait is no longer applicable because the new trait replaces it at the same dimension.\n"
            "- Do not use SHIFT_TO when the new trait only adds a related disposition without invalidating the previous one.\n"
            "- When uncertain between SHIFT_TO and CONTRADICT, choose CONTRADICT.\n"
            "\n"
            f"Output exactly {expected_5a} judgments — one for each listed pair, in the listed order.\n"
            "Do not skip or duplicate pairs.\n"
            "\n"
            "[New Trait]\n"
            f"{new_trait_block}\n"
            "\n"
            "[Recent States]\n"
            f"{recent_states_block}\n"
            "\n"
            "[Recent Episodes]\n"
            f"{recent_episodes_block}\n"
            "\n"
            "[Previous Trait]\n"
            f"{previous_trait_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {expected_5a} judgments — one for each listed pair, in the listed order.\n"
            "Use the integer indices shown above for source_id and target_id.\n"
            "Return strict JSON only."
        )
        new_trait_id = trait_ids[0] if trait_ids else None
        expected_pairs: List[Tuple[str, str]] = []
        if new_trait_id is not None:
            for sid in s_ids:
                expected_pairs.append((sid, new_trait_id))
            for eid in e_ids:
                expected_pairs.append((eid, new_trait_id))
            for pt in pt_ids:
                expected_pairs.append((pt, new_trait_id))
        return PendingLLMCall(
            call_type="trait_evidence_5a",
            system_prompt=SYS_TRAIT_EVIDENCE_5A,
            user_prompt=prompt,
            guided_json=JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_TRAIT_EVIDENCE_5A,
            log_dir="call_5a_trait_evidence",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=expected_5a,
            id_map=id_map,
            expected_pairs=expected_pairs,
        )

    def _build_5b_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
    ) -> Optional[PendingLLMCall]:
        if not self._pending_trait_node_id:
            return None
        trait = self._graph.get_node(self._pending_trait_node_id)
        if trait is None:
            return None

        recent_ids: Set[str] = set()
        for chunk in self._completed_chunks:
            recent_ids.update(chunk.state_ids)
            if chunk.episode_id:
                recent_ids.add(chunk.episode_id)

        k_state = self._cfg.TRAIT_EXTRA_REL_TOPK_STATE
        k_ep = self._cfg.TRAIT_EXTRA_REL_TOPK_EPISODE

        candidate_states = self._5b_candidates(trait, NODE_S, recent_ids, k_state)
        candidate_episodes = self._5b_candidates(trait, NODE_E, recent_ids, k_ep)

        if not candidate_states and not candidate_episodes:
            return None

        trait_block, trait_ids = self._format_node_list_indexed(
            [trait.node_id], anchor_timestamp_seconds, start_idx=0,
        )
        cand_state_block, cs_ids = self._format_node_list_indexed(
            [n.node_id for n in candidate_states], anchor_timestamp_seconds,
            start_idx=len(trait_ids),
        )
        cand_ep_block, ce_ids = self._format_node_list_indexed(
            [n.node_id for n in candidate_episodes], anchor_timestamp_seconds,
            start_idx=len(trait_ids) + len(cs_ids),
        )
        id_map = trait_ids + cs_ids + ce_ids
        trait_idx = 0
        num_candidates = len(cs_ids) + len(ce_ids)

        prompt = (
            "You will judge direct evidence relationships between a newly extracted trait and additional candidate nodes that are currently unconnected to it in the graph.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"Index {trait_idx} is the new trait. "
            f"Indices {len(trait_ids)}..{len(trait_ids)+len(cs_ids)-1} are candidate states. "
            f"Indices {len(trait_ids)+len(cs_ids)}..{len(trait_ids)+len(cs_ids)+len(ce_ids)-1} are candidate episodes.\n"
            "\n"
            "Judge each listed candidate directly against the new trait.\n"
            "\n"
            "Direction rule:\n"
            f"- source_id = candidate index, target_id = {trait_idx} (new trait) for every judgment.\n"
            "\n"
            "Relation rules:\n"
            "- state ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT\n"
            "- episode ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT\n"
            "- SHIFT_TO is NOT allowed in this call. Cross-type pairs (state↔trait, episode↔trait) cannot be temporal replacements.\n"
            "\n"
            "The listed candidates are surfaced by global similarity mining over unconnected state↔trait and episode↔trait pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each candidate's content directly.\n"
            "\n"
            "SUPPORT calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- For (state, new_trait): use SUPPORT only when the state describes a current/recent condition that would be expected GIVEN the new trait, or that the trait would naturally generalize from. Sharing a domain or keyword is not enough.\n"
            "- For (episode, new_trait): use SUPPORT only when the episode captures concrete speaker behavior that the trait would predict; the episode must add evidential force beyond mere topic overlap.\n"
            "- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "CONTRADICT calibration:\n"
            "- Use CONTRADICT only when the candidate clearly conflicts with the new trait at the persona level.\n"
            "- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the trait.\n"
            "- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            f"Output exactly {num_candidates} judgments — one for each listed candidate, in the listed order.\n"
            "Do not skip or duplicate candidates.\n"
            "\n"
            "[New Trait]\n"
            f"{trait_block}\n"
            "\n"
            "[Additional Candidate States]\n"
            f"{cand_state_block}\n"
            "\n"
            "[Additional Candidate Episodes]\n"
            f"{cand_ep_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {num_candidates} judgments — one for each listed candidate, in the listed order.\n"
            "Use the integer indices shown above for source_id and target_id.\n"
            "Return strict JSON only."
        )
        new_trait_id = trait_ids[0]
        expected_pairs: List[Tuple[str, str]] = [
            (cid, new_trait_id) for cid in (cs_ids + ce_ids)
        ]
        return PendingLLMCall(
            call_type="trait_extra_rel_5b",
            system_prompt=SYS_TRAIT_EXTRA_REL_5B,
            user_prompt=prompt,
            guided_json=JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_TRAIT_EXTRA_REL_5B,
            log_dir="call_5b_trait_extra_rel",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=num_candidates,
            id_map=id_map,
            expected_pairs=expected_pairs,
        )

    def _5b_candidates(
        self,
        trait: Node,
        node_type: str,
        exclude_ids: Set[str],
        top_k: int,
    ) -> List[Node]:
        """Retrieve extra candidates for ⑤b via sem_topk ∪ lex_topk (no rerank, no cap)."""
        all_nodes = [
            n for n in self._graph.get_nodes_by_type(node_type)
            if n.node_id not in exclude_ids
            and not self._graph.has_direct_edge(trait.node_id, n.node_id)
        ]
        if not all_nodes:
            return []

        sem_scored = sorted(
            all_nodes,
            key=lambda n: float(np.dot(trait.embedding, n.embedding)),
            reverse=True,
        )[:top_k]

        trait_lex = set(trait.keywords) | {l.lower() for l in trait.domain_label}
        lex_scored = sorted(
            all_nodes,
            key=lambda n: len(trait_lex & (set(n.keywords) | {l.lower() for l in n.domain_label})),
            reverse=True,
        )[:top_k]

        seen: Set[str] = set()
        union: List[Node] = []
        for n in sem_scored + lex_scored:
            if n.node_id not in seen:
                seen.add(n.node_id)
                union.append(n)
        return union

    def _build_5c_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
    ) -> Optional[PendingLLMCall]:
        if not self._pending_new_state_ids_5c:
            return None

        # At flush time: build candidate pair pool from pending new state IDs
        # vs. all states (unconnected only), then apply sem_topK ∪ lex_topK
        # over pairs.
        pending_new = self._pending_new_state_ids_5c
        self._pending_new_state_ids_5c = set()

        all_states = self._graph.get_nodes_by_type(NODE_S)
        node_by_id = {n.node_id: n for n in all_states}
        candidate_pairs: List[Tuple[Node, Node]] = []
        seen_pair_keys: Set[Tuple[str, str]] = set()
        for sid in pending_new:
            a = node_by_id.get(sid)
            if a is None:
                continue
            for b in all_states:
                if a.node_id == b.node_id:
                    continue
                if self._graph.has_direct_edge(a.node_id, b.node_id):
                    continue
                key = tuple(sorted((a.node_id, b.node_id)))
                if key in seen_pair_keys:
                    continue
                seen_pair_keys.add(key)
                candidate_pairs.append((a, b))

        if not candidate_pairs:
            return None

        selected_pairs = self._pair_topk_union(
            candidate_pairs, self._cfg.STATE_STATE_EXTRA_REL_TOPK,
        )
        if not selected_pairs:
            return None

        id_map: List[str] = []
        id_to_idx: Dict[str, int] = {}

        def _5c_get_idx(nid: str) -> int:
            if nid not in id_to_idx:
                id_to_idx[nid] = len(id_map)
                id_map.append(nid)
            return id_to_idx[nid]

        pair_lines = []
        ss_expected_pairs: List[Tuple[str, str]] = []
        for pair_num, (node_a, node_b) in enumerate(selected_pairs):
            if node_a.created_at <= node_b.created_at:
                old_node, new_node = node_a, node_b
            else:
                old_node, new_node = node_b, node_a
            old_id, new_id = old_node.node_id, new_node.node_id
            old_idx = _5c_get_idx(old_id)
            new_idx = _5c_get_idx(new_id)
            ss_expected_pairs.append((old_id, new_id))
            old_elapsed = self._elapsed_str(old_node.timestamp_seconds, anchor_timestamp_seconds)
            new_elapsed = self._elapsed_str(new_node.timestamp_seconds, anchor_timestamp_seconds)
            pair_lines.append(
                f"Pair {pair_num}\n"
                f"- older_state: [{old_idx}] [{old_elapsed}] (scope={old_node.scope}): {old_node.content}\n"
                f"- newer_state: [{new_idx}] [{new_elapsed}] (scope={new_node.scope}): {new_node.content}"
            )

        if not pair_lines:
            return None

        pair_block = "\n\n".join(pair_lines)
        num_pairs = len(pair_lines)
        prompt = (
            "You will judge direct evidence relationships for candidate state-state pairs.\n"
            "Each pair is currently unconnected in the graph.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            "\n"
            "The listed pairs are surfaced by global similarity mining over unconnected state-state pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each pair's content directly.\n"
            "\n"
            "The pair is shown in chronological order: older_state first, newer_state second.\n"
            "\n"
            "Direction rule:\n"
            "- For SUPPORT, CONTRADICT, IRRELEVANT: source_id = older_state index, target_id = newer_state index.\n"
            "- For SHIFT_TO: source_id MUST be the older_state index, target_id MUST be the newer_state index.\n"
            "\n"
            "Relation rules:\n"
            "- SUPPORT: the two states are consistent or mutually reinforcing.\n"
            "- CONTRADICT: the two states are in tension but can coexist as evidence.\n"
            "- SHIFT_TO: the older state has changed into the newer state and is no longer currently valid.\n"
            "- IRRELEVANT: the two states are unrelated, OR are merely topically similar without evidential force in either direction.\n"
            "\n"
            "SUPPORT calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- Temporal proximity alone is NOT SUPPORT.\n"
            "- Use SUPPORT only when one state provides independent evidential force for the other (e.g., one state would be expected GIVEN the other, or both reinforce the same persona signal from independent angles).\n"
            "- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "CONTRADICT calibration:\n"
            "- Use CONTRADICT only when the two states are in clear tension at the persona level.\n"
            "- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the state.\n"
            "- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "SHIFT_TO caution:\n"
            "- Be conservative with SHIFT_TO.\n"
            "- A short-term or narrow preference does not replace a broader condition, value, role, lifestyle constraint, or safety condition unless one state explicitly invalidates the other.\n"
            "- Do not use SHIFT_TO when the newer state only adds detail without invalidating the older state.\n"
            "- Do not use SHIFT_TO for a topic change without explicit replacement.\n"
            "- When uncertain between SHIFT_TO and CONTRADICT, choose CONTRADICT.\n"
            "\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order.\n"
            "Do not skip or duplicate pairs.\n"
            "\n"
            "[Candidate State-State Pairs]\n"
            f"{pair_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order.\n"
            "Use the integer indices shown above for source_id and target_id.\n"
            "Return strict JSON only."
        )
        return PendingLLMCall(
            call_type="state_state_5c",
            system_prompt=SYS_STATE_STATE_5C,
            user_prompt=prompt,
            guided_json=JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_STATE_STATE_5C,
            log_dir="call_5c_state_state_rel",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=num_pairs,
            id_map=id_map,
            expected_pairs=ss_expected_pairs,
        )

    def _build_5d_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
    ) -> Optional[PendingLLMCall]:
        if not (self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d):
            return None

        # At flush time: build candidate state-episode pairs from:
        #   (pending new states) × (all episodes) ∪ (pending new episodes) × (all states)
        # then apply sem_topK ∪ lex_topK over pairs. State is always pos 0, episode pos 1.
        pending_new_s = self._pending_new_state_ids_5d
        pending_new_e = self._pending_new_episode_ids_5d
        self._pending_new_state_ids_5d = set()
        self._pending_new_episode_ids_5d = set()

        all_states = self._graph.get_nodes_by_type(NODE_S)
        all_episodes = self._graph.get_nodes_by_type(NODE_E)
        state_by_id = {n.node_id: n for n in all_states}
        episode_by_id = {n.node_id: n for n in all_episodes}

        candidate_pairs: List[Tuple[Node, Node]] = []
        seen_pair_keys: Set[Tuple[str, str]] = set()

        def _add_pair(s_node: Node, e_node: Node) -> None:
            if self._graph.has_direct_edge(s_node.node_id, e_node.node_id):
                return
            key = (s_node.node_id, e_node.node_id)
            if key in seen_pair_keys:
                return
            seen_pair_keys.add(key)
            candidate_pairs.append((s_node, e_node))

        for sid in pending_new_s:
            s_node = state_by_id.get(sid)
            if s_node is None:
                continue
            for e_node in all_episodes:
                _add_pair(s_node, e_node)

        for eid in pending_new_e:
            e_node = episode_by_id.get(eid)
            if e_node is None:
                continue
            for s_node in all_states:
                _add_pair(s_node, e_node)

        if not candidate_pairs:
            return None

        selected_pairs = self._pair_topk_union(
            candidate_pairs, self._cfg.STATE_EPISODE_EXTRA_REL_TOPK,
        )
        if not selected_pairs:
            return None

        id_map: List[str] = []
        id_map_b: List[str] = []
        s_to_idx: Dict[str, int] = {}
        e_to_idx: Dict[str, int] = {}

        def _get_s_idx(nid: str) -> int:
            if nid not in s_to_idx:
                s_to_idx[nid] = len(id_map)
                id_map.append(nid)
            return s_to_idx[nid]

        def _get_e_idx(nid: str) -> int:
            if nid not in e_to_idx:
                e_to_idx[nid] = len(id_map_b)
                id_map_b.append(nid)
            return e_to_idx[nid]

        pair_lines = []
        se_expected_pairs: List[Tuple[str, str]] = []
        for pair_num, (s_node, e_node) in enumerate(selected_pairs):
            s_id = s_node.node_id
            e_id = e_node.node_id
            s_idx = _get_s_idx(s_id)
            e_idx = _get_e_idx(e_id)
            se_expected_pairs.append((e_id, s_id))
            elapsed_s = self._elapsed_str(s_node.timestamp_seconds, anchor_timestamp_seconds)
            elapsed_e = self._elapsed_str(e_node.timestamp_seconds, anchor_timestamp_seconds)
            same_session = (s_node.session_id == e_node.session_id)
            if same_session:
                relation_context = "same session."
            else:
                gap_str = format_session_gap(s_node.timestamp_seconds, e_node.timestamp_seconds)
                relation_context = f"different sessions, {gap_str} apart."
            pair_lines.append(
                f"Pair {pair_num}\n"
                f"- state:   [{s_idx}] [{elapsed_s}]: {s_node.content}\n"
                f"- episode: [{e_idx}] [{elapsed_e}]: {e_node.content}\n"
                f"  Relation context: {relation_context}"
            )

        if not pair_lines:
            return None

        pair_block = "\n\n".join(pair_lines)
        num_pairs = len(pair_lines)
        prompt = (
            "You will judge direct evidence relationships for candidate state-episode pairs.\n"
            "Each pair is currently unconnected in the graph.\n"
            "\n"
            "State nodes use one integer index space (state_id); episode nodes use a separate integer index space (episode_id).\n"
            "\n"
            "The listed pairs are surfaced by global similarity mining over unconnected state-episode pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each pair's content directly.\n"
            "\n"
            "Direction rule:\n"
            "- This call uses keyed identifiers `state_id` and `episode_id` (not `source_id`/`target_id`). Output one judgment per listed pair, naming each pair by its state and episode integer indices.\n"
            "\n"
            "Relation rules:\n"
            "- SUPPORT: the episode provides concrete episodic evidence that independently grounds or confirms the state.\n"
            "- CONTRADICT: the episode provides concrete episodic evidence that conflicts with the state.\n"
            "- IRRELEVANT: merely topically related without evidential force, or unrelated.\n"
            "\n"
            "SUPPORT calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- Same-session co-extraction or temporal adjacency alone is NOT SUPPORT.\n"
            "- Use SUPPORT only when the episode provides concrete episodic evidence that independently grounds or confirms the state. The episode must add an episodic fact that would still ground the state if read in isolation.\n"
            "- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "CONTRADICT calibration:\n"
            "- Use CONTRADICT only when the episode contains an episodic fact that directly invalidates or conflicts with the state's claim.\n"
            "- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the state.\n"
            "- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "Relation context interpretation:\n"
            "Each pair carries a \"Relation context\" line indicating whether the state and episode come from the same session.\n"
            "- \"same session\" pairs deserve extra scrutiny: same-session co-extraction alone is NOT SUPPORT. Look for an episodic fact in the episode that would still ground the state outside that session.\n"
            "- \"different sessions\" pairs come from temporally distinct episodes; SUPPORT here is most justified when the episode's events directly evidence the state.\n"
            "\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order.\n"
            "Do not skip or duplicate pairs.\n"
            "\n"
            "[Candidate State-Episode Pairs]\n"
            f"{pair_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order.\n"
            "Use the integer indices shown above for state_id and episode_id.\n"
            "Return strict JSON only."
        )
        return PendingLLMCall(
            call_type="state_episode_5d",
            system_prompt=SYS_STATE_EPISODE_5D,
            user_prompt=prompt,
            guided_json=STATE_EPISODE_JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_STATE_EPISODE_5D,
            log_dir="call_5d_state_episode_rel",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=num_pairs,
            id_map=id_map,
            id_map_b=id_map_b,
            expected_pairs=se_expected_pairs,
        )

    def _get_recent_prev_state_ids(self, exclude_ids: Set[str], k: int) -> List[str]:
        states = [
            n for n in self._graph.get_nodes_by_type(NODE_S)
            if n.node_id not in exclude_ids
        ]
        states.sort(key=lambda n: n.created_at, reverse=True)
        return [n.node_id for n in states[:k]]

    def _build_state_new_rel_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
        previous_state_ids: List[str],
        new_state_ids: List[str],
    ) -> PendingLLMCall:
        self._pending_state_new_rel_new_ids = list(new_state_ids)
        prev_block, prev_ids = self._format_state_list_with_scope_indexed(
            previous_state_ids, anchor_timestamp_seconds, start_idx=0,
        )
        new_block, new_ids = self._format_state_list_with_scope_indexed(
            new_state_ids, anchor_timestamp_seconds, start_idx=len(prev_ids),
        )
        id_map = prev_ids + new_ids
        n_prev_valid = len(prev_ids)
        n_new_valid = len(new_ids)
        prompt = (
            "You will judge direct evidence relationships involving newly extracted states.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"Indices 0..{n_prev_valid - 1} are previous states. "
            f"Indices {n_prev_valid}..{n_prev_valid + n_new_valid - 1} are new states.\n"
            "\n"
            "Judge:\n"
            "- every previous↔new pair; and\n"
            "- every unordered pair of new states, only when more than one new state was extracted.\n"
            "  Judge each new↔new pair exactly once; do not output both (A,B) and (B,A).\n"
            "\n"
            "Direction rules:\n"
            "- For previous↔new pairs: previous state = source_id, new state = target_id.\n"
            "- For new↔new SHIFT_TO: source_id = older state index (as described in content), target_id = newer replacement state index.\n"
            "- For new↔new SUPPORT, CONTRADICT, IRRELEVANT: earlier-listed new state = source_id, later-listed new state = target_id.\n"
            "\n"
            "new↔new pair caution:\n"
            "- The two states were extracted from the SAME current turn. Co-extracted states usually coexist.\n"
            "- Use SHIFT_TO for a new↔new pair only if the CURRENT TURN explicitly states a temporal replacement between them.\n"
            "- Do not infer SHIFT_TO merely because the two states differ, contrast, or appear emotionally opposed.\n"
            "\n"
            "Calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- Temporal proximity alone does not create evidential force.\n"
            "- A short-term or narrow preference does not replace a broader condition, value, role, lifestyle constraint, or safety condition unless one state explicitly invalidates the other.\n"
            "- Do not use SHIFT_TO when the newer state only adds detail without invalidating the older state.\n"
            "- Do not use SHIFT_TO for a topic change without explicit replacement.\n"
            "- When uncertain between SHIFT_TO and CONTRADICT, choose CONTRADICT.\n"
            "- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "[Previous States — extracted before the current turn]\n"
            f"{prev_block}\n"
            "\n"
            "[New States — extracted from the current turn]\n"
            f"{new_block}"
        )
        n_new = n_new_valid
        n_prev = n_prev_valid
        expected = (n_new * (n_new - 1)) // 2 + n_new * n_prev
        expected_pairs: List[Tuple[str, str]] = []
        for p in prev_ids:
            for n in new_ids:
                expected_pairs.append((p, n))
        for i in range(len(new_ids)):
            for j in range(i + 1, len(new_ids)):
                expected_pairs.append((new_ids[i], new_ids[j]))
        return PendingLLMCall(
            call_type="state_new_rel",
            system_prompt=SYS_STATE_NEW_REL,
            user_prompt=prompt,
            guided_json=JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_STATE_NEW_REL,
            log_dir="call_2b_state_new_rel",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=expected,
            id_map=id_map,
            expected_pairs=expected_pairs,
        )

    def _build_episode_new_rel_call(
        self,
        session_id: int,
        created_at: int,
        anchor_session_id: int,
        anchor_turn_idx: int,
        anchor_timestamp_seconds: float,
        new_episode_id: str,
        previous_episode_id: Optional[str],
        chunk_state_ids: List[str],
    ) -> PendingLLMCall:
        self._pending_episode_new_rel = {
            "new_episode_id": new_episode_id,
            "previous_episode_id": previous_episode_id,
            "chunk_state_ids": list(chunk_state_ids),
        }

        new_block, new_ids = self._format_node_list_indexed(
            [new_episode_id], anchor_timestamp_seconds, start_idx=0,
        )
        prev_block, prev_ids = self._format_node_list_indexed(
            [previous_episode_id] if previous_episode_id else [],
            anchor_timestamp_seconds, start_idx=len(new_ids),
        )
        chunk_states_block, state_ids = self._format_node_list_indexed(
            chunk_state_ids, anchor_timestamp_seconds,
            start_idx=len(new_ids) + len(prev_ids),
        )
        id_map = new_ids + prev_ids + state_ids

        n_prev = len(prev_ids)
        n_states = len(state_ids)
        expected = n_prev + n_states

        prev_idx = len(new_ids) if n_prev else None
        state_start = len(new_ids) + n_prev

        prompt = (
            "You will judge direct evidence relationships involving the newly extracted episode.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            "Index 0 is the new episode. "
            + (f"Index {prev_idx} is the previous episode. " if prev_idx is not None else "")
            + (f"Indices {state_start}..{state_start + n_states - 1} are chunk states.\n" if n_states else "\n")
            + "\n"
            "Judge:\n"
            "- the (new_episode, previous_episode) pair, when a previous episode exists; and\n"
            "- one (new_episode, chunk_state_i) pair for each listed chunk state, in the listed order.\n"
            "\n"
            "Direction rules (FIXED):\n"
            "- source_id MUST be 0 (new episode index) for every judgment.\n"
            "- target_id is the previous_episode index or chunk_state index.\n"
            "\n"
            "Relation rules:\n"
            "\n"
            "For (new_episode, previous_episode):\n"
            "  Use SUPPORT | CONTRADICT | IRRELEVANT.\n"
            "  Temporal adjacency alone is NOT SUPPORT.\n"
            "  Use SUPPORT only when the new episode continues, confirms, or concretely reinforces the previous episode.\n"
            "\n"
            "For (new_episode, chunk_state_i):\n"
            "  Use SUPPORT | CONTRADICT | IRRELEVANT.\n"
            "  Use SUPPORT only when the episode provides concrete episodic evidence that independently grounds or confirms the state.\n"
            "  Same-session co-extraction alone is NOT sufficient evidential force.\n"
            "  The episode must add an episodic fact that would still ground the state if read in isolation.\n"
            "\n"
            "When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            f"Output exactly {expected} judgments in this order:\n"
            "  1) the (new_episode, previous_episode) judgment, if a previous episode exists;\n"
            "  2) one (new_episode, chunk_state_i) judgment for each listed chunk state, in the listed order.\n"
            "Do not invent states, episodes, or judgments.\n"
            "\n"
            "[New Episode]\n"
            f"{new_block}\n"
            "\n"
            "[Previous Episode]\n"
            f"{prev_block}\n"
            "\n"
            "[Chunk States — listed in extraction order]\n"
            f"{chunk_states_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {expected} judgments in the order specified above (previous_episode first if present, then chunk states in listed order).\n"
            "Use the integer indices shown above for source_id and target_id.\n"
            "Return strict JSON only."
        )
        expected_pairs: List[Tuple[str, str]] = []
        if prev_ids:
            expected_pairs.append((new_ids[0], prev_ids[0]))
        for s in state_ids:
            expected_pairs.append((new_ids[0], s))
        return PendingLLMCall(
            call_type="episode_new_rel",
            system_prompt=SYS_EPISODE_NEW_REL,
            user_prompt=prompt,
            guided_json=JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_EPISODE_NEW_REL,
            log_dir="call_3b_episode_new_rel",
            created_at=created_at,
            anchor_session_id=anchor_session_id,
            anchor_turn_idx=anchor_turn_idx,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            session_id=session_id,
            expected_judgment_count=expected,
            id_map=id_map,
            expected_pairs=expected_pairs,
        )

    # ------------------------------------------------------------------
    # Result appliers
    # ------------------------------------------------------------------

    def _apply_state_result(self, call: PendingLLMCall, result: Dict) -> None:
        raw_states = result.get("states", []) if isinstance(result, dict) else []
        new_state_ids: List[str] = []
        recent_turns = list(self._recent_turns)
        anchor_turn = recent_turns[-1] if recent_turns else None

        for item in raw_states[: self._cfg.STATE_MAX_COUNT]:
            content = _validate_content(item)
            if not content:
                continue
            keywords = _validate_keywords(item.get("keywords"), self._cfg.MAX_KEYWORDS)
            domain_label = _validate_domain_labels(
                item.get("domain_label"),
                keywords,
                self._cfg.MIN_DOMAIN_LABELS,
                self._cfg.MAX_DOMAIN_LABELS,
            )
            speaker = anchor_turn.speaker if anchor_turn else ""
            session_id = anchor_turn.session_id if anchor_turn else call.anchor_session_id
            turn_idx = anchor_turn.turn_idx if anchor_turn else call.anchor_turn_idx
            ts = anchor_turn.timestamp_seconds if anchor_turn else call.anchor_timestamp_seconds
            dia_id = anchor_turn.dia_id if anchor_turn else ""
            node = Node(
                node_id=HeterogeneousGraph.new_node_id(),
                node_type=NODE_S,
                content=content,
                keywords=keywords,
                domain_label=domain_label,
                embedding=self._embed_text(content),
                created_at=call.created_at,
                session_id=session_id,
                conv_id=session_id,
                turn_id=turn_idx,
                scope=_validate_scope(item.get("scope")),
                recall_priority=_validate_recall_priority(item.get("recall_priority")),
                retrieval_count=1,
                timestamp_seconds=ts,
                dia_id=dia_id,
                speaker=speaker,
            )
            self._graph.add_node(node)
            new_state_ids.append(node.node_id)

            for turn in recent_turns:
                self._graph.add_source_edge(turn.context_node_id, node.node_id)
            if self._current_chunk is not None:
                self._current_chunk.state_ids.append(node.node_id)

        self._previous_state_ids = new_state_ids
        self._state_extracted_count += len(new_state_ids)

        self._update_ss_reservoir(new_state_ids)
        self._update_sm_reservoir(new_state_ids, [])

    def _apply_episode_result(self, call: PendingLLMCall, result: Dict) -> None:
        episode_obj = result.get("episode", {}) if isinstance(result, dict) else {}
        content = _validate_content(episode_obj)
        if not content:
            self._finalize_chunk_record()
            return

        chunk = self._current_chunk
        if chunk is None:
            return

        keywords = _validate_keywords(episode_obj.get("keywords"), self._cfg.MAX_KEYWORDS)
        domain_label = _validate_domain_labels(
            episode_obj.get("domain_label"),
            keywords,
            self._cfg.MIN_DOMAIN_LABELS,
            self._cfg.MAX_DOMAIN_LABELS,
        )
        episode_node = Node(
            node_id=HeterogeneousGraph.new_node_id(),
            node_type=NODE_E,
            content=content,
            keywords=keywords,
            domain_label=domain_label,
            embedding=self._embed_text(content),
            created_at=call.created_at,
            session_id=chunk.session_id,
            conv_id=chunk.session_id,
            turn_id=chunk.last_turn_idx,
            scope=_validate_scope(episode_obj.get("scope")),
            retrieval_count=1,
            timestamp_seconds=chunk.last_timestamp_seconds,
        )
        self._graph.add_node(episode_node)
        chunk.episode_id = episode_node.node_id
        for context_id in chunk.context_ids:
            self._graph.add_source_edge(context_id, episode_node.node_id)

        self._previous_episode_id = episode_node.node_id
        self._episode_extracted_count += 1
        self._finalize_chunk_record()

        self._update_sm_reservoir([], [episode_node.node_id])

    def _apply_trait_result(self, call: PendingLLMCall, result: Dict) -> Optional[str]:
        raw_traits = result.get("traits", []) if isinstance(result, dict) else []
        if not raw_traits:
            return None

        item = raw_traits[0]
        content = _validate_content(item)
        if not content:
            return None

        speaker = ""
        if isinstance(item, dict):
            sp = item.get("speaker")
            if isinstance(sp, str):
                speaker = sp.strip()

        keywords = _validate_keywords(item.get("keywords"), self._cfg.MAX_KEYWORDS)
        domain_label = _validate_domain_labels(
            item.get("domain_label"),
            keywords,
            self._cfg.MIN_DOMAIN_LABELS,
            self._cfg.MAX_DOMAIN_LABELS,
        )
        recent_chunks = list(self._completed_chunks)
        latest_session = recent_chunks[-1].session_id if recent_chunks else call.anchor_session_id
        latest_turn = recent_chunks[-1].last_turn_idx if recent_chunks else call.anchor_turn_idx
        latest_ts = recent_chunks[-1].last_timestamp_seconds if recent_chunks else call.anchor_timestamp_seconds

        trait_node = Node(
            node_id=HeterogeneousGraph.new_node_id(),
            node_type=NODE_T,
            content=content,
            keywords=keywords,
            domain_label=domain_label,
            embedding=self._embed_text(content),
            created_at=call.created_at,
            session_id=latest_session,
            conv_id=latest_session,
            turn_id=latest_turn,
            scope=_validate_scope(item.get("scope")),
            retrieval_count=1,
            timestamp_seconds=latest_ts,
            speaker=speaker,
        )
        self._graph.add_node(trait_node)

        for chunk in recent_chunks:
            for context_id in chunk.context_ids:
                self._graph.add_source_edge(context_id, trait_node.node_id)
            for state_id in chunk.state_ids:
                self._graph.add_source_edge(state_id, trait_node.node_id)
            if chunk.episode_id:
                self._graph.add_source_edge(chunk.episode_id, trait_node.node_id)

        self._trait_extracted_count += 1
        return trait_node.node_id

    def _apply_5a_result(self, result: Dict, id_map: List[str]) -> None:
        if self._pending_trait_node_id is None:
            return
        new_trait_id = self._pending_trait_node_id

        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            src_idx = item.get("source_id")
            dst_idx = item.get("target_id")
            relation = item.get("relation", "")
            src = _resolve_id(id_map, src_idx)
            dst = _resolve_id(id_map, dst_idx)
            if src is None or dst is None or src == dst:
                continue

            src_node = self._graph.get_node(src)
            dst_node = self._graph.get_node(dst)
            if src_node is None or dst_node is None:
                continue

            src_t, dst_t = src_node.node_type, dst_node.node_type

            if src_t == NODE_T and dst_t == NODE_T:
                rel = _validate_relation(relation)
                if rel is None:
                    continue
                if rel == EVID_SHIFT_TO:
                    if src_node.created_at <= dst_node.created_at:
                        self._store_evidence_edge(src, dst, EVID_SHIFT_TO)
                    else:
                        self._store_evidence_edge(dst, src, EVID_SHIFT_TO)
                else:
                    self._store_evidence_edge(src, dst, rel)
            else:
                rel = _validate_relation_reduced(relation)
                if rel is None:
                    continue
                if dst == new_trait_id:
                    self._store_evidence_edge(src, dst, rel)
                else:
                    self._store_evidence_edge(dst, src, rel)

        self._previous_trait_id = new_trait_id
        self._5a_call_count += 1

    def _apply_5b_result(self, result: Dict, id_map: List[str]) -> None:
        if self._pending_trait_node_id is None:
            return
        new_trait_id = self._pending_trait_node_id
        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            src_idx = item.get("source_id")
            dst_idx = item.get("target_id")
            src = _resolve_id(id_map, src_idx)
            dst = _resolve_id(id_map, dst_idx)
            rel = _validate_relation_reduced(item.get("relation"))
            if src is None or dst is None or src == dst or rel is None:
                continue
            if dst == new_trait_id:
                self._store_evidence_edge(src, new_trait_id, rel)
            else:
                self._store_evidence_edge(dst, new_trait_id, rel)
        self._5b_call_count += 1

    def _apply_5c_result(self, result: Dict, id_map: List[str]) -> None:
        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            src_idx = item.get("source_id")
            dst_idx = item.get("target_id")
            src = _resolve_id(id_map, src_idx)
            dst = _resolve_id(id_map, dst_idx)
            if src is None or dst is None or src == dst:
                continue
            rel = _validate_relation(item.get("relation", ""))
            if rel is None:
                continue

            src_node = self._graph.get_node(src)
            dst_node = self._graph.get_node(dst)
            if src_node is None or dst_node is None:
                continue

            if rel == EVID_SHIFT_TO:
                if src_node.created_at <= dst_node.created_at:
                    self._store_evidence_edge(src, dst, EVID_SHIFT_TO)
                else:
                    self._store_evidence_edge(dst, src, EVID_SHIFT_TO)
            else:
                self._store_evidence_edge(src, dst, rel)
        self._5c_call_count += 1

    def _apply_5d_result(self, result: Dict, id_map: List[str], id_map_b: List[str]) -> None:
        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            s_idx = item.get("state_id")
            e_idx = item.get("episode_id")
            s_id = _resolve_id(id_map, s_idx)
            e_id = _resolve_id(id_map_b, e_idx)
            rel = _validate_relation_reduced(item.get("relation"))
            if s_id is None or e_id is None or rel is None:
                continue
            self._store_evidence_edge(e_id, s_id, rel)
        self._5d_call_count += 1

    def _apply_state_new_rel_result(self, result: Dict, id_map: List[str]) -> None:
        new_state_ids = self._pending_state_new_rel_new_ids
        self._pending_state_new_rel_new_ids = []
        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            src_idx = item.get("source_id")
            dst_idx = item.get("target_id")
            src = _resolve_id(id_map, src_idx)
            dst = _resolve_id(id_map, dst_idx)
            relation = _validate_relation(item.get("relation"))
            if src is None or dst is None or src == dst or relation is None:
                continue
            if self._graph.get_node(src) is None or self._graph.get_node(dst) is None:
                continue
            self._add_state_judgment(src, dst, relation, new_state_ids)
        self._state_new_rel_call_count += 1

    def _apply_episode_new_rel_result(self, result: Dict, id_map: List[str]) -> None:
        pending = self._pending_episode_new_rel
        self._pending_episode_new_rel = None
        if pending is None:
            return
        new_id = pending["new_episode_id"]
        prev_id = pending.get("previous_episode_id")
        chunk_state_ids = set(pending.get("chunk_state_ids") or [])

        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            src_idx = item.get("source_id")
            dst_idx = item.get("target_id")
            src = _resolve_id(id_map, src_idx)
            dst = _resolve_id(id_map, dst_idx)
            rel = _validate_relation_reduced(item.get("relation"))
            if src is None or dst is None or src == dst or rel is None:
                continue

            if src == new_id:
                other = dst
            elif dst == new_id:
                other = src
            else:
                continue

            if prev_id and other == prev_id:
                self._store_evidence_edge(new_id, prev_id, rel)
            elif other in chunk_state_ids:
                self._store_evidence_edge(new_id, other, rel)
        self._episode_new_rel_call_count += 1

    def _add_state_judgment(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        new_state_ids: List[str],
    ) -> None:
        if relation == EVID_SHIFT_TO:
            src_new = src_id in new_state_ids
            dst_new = dst_id in new_state_ids
            if src_new and not dst_new:
                self._store_evidence_edge(dst_id, src_id, EVID_SHIFT_TO)
            elif dst_new and not src_new:
                self._store_evidence_edge(src_id, dst_id, EVID_SHIFT_TO)
            else:
                src_node = self._graph.get_node(src_id)
                dst_node = self._graph.get_node(dst_id)
                if src_node and dst_node and src_node.created_at > dst_node.created_at:
                    self._store_evidence_edge(dst_id, src_id, EVID_SHIFT_TO)
                else:
                    self._store_evidence_edge(src_id, dst_id, EVID_SHIFT_TO)
            return

        self._store_evidence_edge(src_id, dst_id, relation)

    def _store_evidence_edge(self, src_id: str, dst_id: str, relation: str) -> None:
        self._graph.add_evidence_edge(src_id, dst_id, relation)

    def _finalize_chunk_record(self) -> None:
        if self._current_chunk is None:
            return
        finished = self._current_chunk
        self._completed_chunks.append(finished)
        self._chunk_count += 1
        self._current_chunk = None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_turns(self, turns: List[TurnRecord], anchor_timestamp_seconds: float) -> str:
        if not turns:
            return "(none)"
        lines = []
        for turn in turns:
            elapsed = self._elapsed_str(turn.timestamp_seconds, anchor_timestamp_seconds)
            lines.append(f"[{elapsed}] {turn.speaker} says: {turn.text}")
        return "\n".join(lines)

    def _format_state_list(self, node_ids: List[Optional[str]], anchor_timestamp_seconds: float) -> str:
        ids = [nid for nid in node_ids if nid]
        if not ids:
            return "(none)"
        lines = []
        for nid in ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            elapsed = self._elapsed_str(node.timestamp_seconds, anchor_timestamp_seconds)
            lines.append(f"[{nid}] [{elapsed}]: {node.content}")
        return "\n".join(lines) if lines else "(none)"

    def _format_node_list_indexed(
        self,
        node_ids: List[Optional[str]],
        anchor_timestamp_seconds: float,
        start_idx: int = 0,
    ) -> Tuple[str, List[str]]:
        """Returns (formatted_block, id_map) using integer indices starting at start_idx.
        Node-type-agnostic — used by ②b, ③b, ⑤a, ⑤b for state/episode/trait blocks."""
        ids = [nid for nid in node_ids if nid and self._graph.get_node(nid)]
        if not ids:
            return "(none)", []
        lines = []
        for i, nid in enumerate(ids):
            node = self._graph.get_node(nid)
            elapsed = self._elapsed_str(node.timestamp_seconds, anchor_timestamp_seconds)
            lines.append(f"[{start_idx + i}] [{elapsed}]: {node.content}")
        return "\n".join(lines), ids

    def _format_state_list_with_scope_indexed(
        self,
        node_ids: List[Optional[str]],
        anchor_timestamp_seconds: float,
        start_idx: int = 0,
    ) -> Tuple[str, List[str]]:
        ids = [nid for nid in node_ids if nid and self._graph.get_node(nid)]
        if not ids:
            return "(none)", []
        lines = []
        for i, nid in enumerate(ids):
            node = self._graph.get_node(nid)
            elapsed = self._elapsed_str(node.timestamp_seconds, anchor_timestamp_seconds)
            lines.append(f"[{start_idx + i}] [{elapsed}] (scope={node.scope}): {node.content}")
        return "\n".join(lines), ids

    def _format_prior_turns(
        self,
        turns: List[Tuple[str, str, int, int, float]],
        anchor_timestamp_seconds: float,
    ) -> str:
        if not turns:
            return "(none)"
        lines = []
        for speaker, text, _sid, _tidx, ts in turns:
            elapsed = self._elapsed_str(ts, anchor_timestamp_seconds)
            lines.append(f"[{elapsed}] {speaker} says: {text}")
        return "\n".join(lines)

    def _elapsed_str(self, entry_timestamp_seconds: float, current_timestamp_seconds: float) -> str:
        return format_elapsed_str(entry_timestamp_seconds, current_timestamp_seconds)

    def _embed_text(self, text: str) -> np.ndarray:
        return embed_text(self._embed_model, text)


# =============================================================================
# Helpers
# =============================================================================

def _extract_token_counts(result) -> Tuple[int, int]:
    if isinstance(result, dict) and "_usage" in result:
        usage = result["_usage"]
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return 0, 0


def _validate_content(item) -> str:
    if not isinstance(item, dict):
        return ""
    value = item.get("content", item.get("text", ""))
    if not isinstance(value, str):
        return ""
    return value.strip()[:1000]


def _validate_keywords(val, max_kw: int) -> List[str]:
    if not isinstance(val, list):
        return []
    result = []
    seen = set()
    for item in val:
        if not isinstance(item, str):
            continue
        norm = item.strip().lower()
        if not norm or norm in seen:
            continue
        # Drop multi-word items so overlap scoring stays token-level.
        if any(ch.isspace() for ch in norm):
            continue
        seen.add(norm)
        result.append(norm)
        if len(result) >= max_kw:
            break
    return result


def _validate_domain_labels(val, keywords: List[str], min_count: int, max_count: int) -> List[str]:
    kw_lower = {k.lower() for k in keywords if isinstance(k, str)}
    result: List[str] = []
    seen: Set[str] = set()
    if isinstance(val, list):
        for item in val:
            if not isinstance(item, str):
                continue
            norm = item.strip().lower()
            if not norm or norm in seen:
                continue
            if norm in kw_lower:
                continue
            if any(ch.isspace() for ch in norm):
                continue
            seen.add(norm)
            result.append(norm)
            if len(result) >= max_count:
                break
    while len(result) < min_count:
        filler = "general" if "general" not in seen else f"general_{len(result)}"
        seen.add(filler)
        result.append(filler)
    return result[:max_count]


def _validate_scope(val) -> str:
    if isinstance(val, str) and val.strip().upper() in {SCOPE_BROAD, SCOPE_NARROW}:
        return val.strip().upper()
    return SCOPE_NARROW


def _validate_recall_priority(val) -> str:
    if isinstance(val, str) and val.strip().upper() in {PRIORITY_HIGH, PRIORITY_LOW}:
        return val.strip().upper()
    return PRIORITY_LOW


def _validate_relation(val) -> Optional[str]:
    if not isinstance(val, str):
        return None
    norm = val.strip().upper()
    if norm in {EVID_SUPPORT, EVID_CONTRADICT, EVID_SHIFT_TO, EVID_IRRELEVANT}:
        return norm
    return None


def _validate_relation_reduced(val) -> Optional[str]:
    if not isinstance(val, str):
        return None
    norm = val.strip().upper()
    if norm in {EVID_SUPPORT, EVID_CONTRADICT, EVID_IRRELEVANT}:
        return norm
    return None


def _resolve_id(id_map: List[str], idx) -> Optional[str]:
    if isinstance(idx, bool):
        return None
    if isinstance(idx, str):
        s = idx.strip().lstrip('[').rstrip(']').strip()
        try:
            idx = int(s)
        except (ValueError, TypeError):
            return None
    if not isinstance(idx, int) or idx < 0 or idx >= len(id_map):
        return None
    return id_map[idx]
