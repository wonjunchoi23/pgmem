"""Updater for PGMem internal calls ②③④⑤."""

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
    format_conv_gap,
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
    anchor_conv_id: int
    anchor_turn_id: int
    session_id: int
    # Expected judgment count; 0 means the call legitimately produces none
    # (extraction-only) and the retry wrapper skips it.
    expected_judgment_count: int = 0
    # Integer-index → node_id maps. id_map_b is only used by ⑤d (episodes).
    id_map: List[str] = field(default_factory=list)
    id_map_b: List[str] = field(default_factory=list)
    # Canonical (src, dst) pairs the call is expected to judge. Used by the
    # IRRELEVANT-fallback after retry exhaustion. Direction per call:
    #   ②b: prev_state → new_state ; new_state(earlier) → new_state(later)
    #   ③b: new_episode → previous_episode ; new_episode → chunk_state
    #   ⑤a: state/episode → new_trait ; previous_trait → new_trait
    #   ⑤b: candidate → new_trait
    #   ⑤c: older_state → newer_state
    #   ⑤d: episode → state
    expected_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class TurnRecord:
    user_utterance: str
    gt_response: str
    conv_id: int
    turn_id: int
    context_node_id: str


@dataclass
class ChunkRecord:
    conv_id: int
    turns: List[TurnRecord] = field(default_factory=list)
    context_ids: List[str] = field(default_factory=list)
    state_ids: List[str] = field(default_factory=list)
    episode_id: Optional[str] = None
    last_turn_id: int = 0


# =============================================================================
# Prompt components
# =============================================================================

_NODE_TYPE_DESC = (
    "Node types:\n"
    "  State: A user condition true at the time it is expressed but subject to change — a current stance, ongoing goal, active constraint, present situation, or preference. Time-bounded and context-sensitive.\n"
    "  Trait: A generalized user characteristic that tends to persist across situations and time — recurring disposition, stable preference, value, or habitual tendency. Cross-situational and relatively context-independent, not tied to one episode or temporary situation.\n"
    "  Episode: A summary of what happened in a past conversation — concrete events, topics, and actions at a particular time. Not a generalized persona attribute."
)

_EVID_DESC_FULL = (
    "Evidence relationships:\n"
    "  IRRELEVANT: The pair is unrelated, or only weakly associated. Use this when the two pieces of information do not help interpret, update, support, or challenge each other.\n"
    "  SHIFT_TO: A same-type temporal transition where the older information has changed into the newer information and is no longer currently valid. The earlier node is the source (old); the later node is the target (new).\n"
    "  CONTRADICT: The two pieces of information are in clear tension or conflict, but not necessarily a temporal replacement. Both may still be meaningful evidence.\n"
    "  SUPPORT: The two pieces are consistent or mutually reinforcing. Shared topic with mutual consistency is enough — SUPPORT does not require strong independent evidential force."
)


_LABEL_DISCIPLINE_BLOCK = (
    "keywords:\n"
    "  Specific one-word terms from the source text or close variants.\n"
    "  Use concrete entities, actions, constraints, or important terminology.\n"
    "  Avoid broad topical categories, speaker names, and time references.\n"
    "\n"
    "domain_label:\n"
    "  Broader topical categories that could group this node with other nodes sharing the same subject area.\n"
    "  Use a higher level of abstraction than keywords.\n"
    "  Multiple labels in the same domain are encouraged when they capture different abstraction levels or alternative names.\n"
    "\n"
    "Constraints (both fields):\n"
    "  - Each item is one continuous word (no whitespace). Compounds like \"machinelearning\" are fine; items with whitespace are dropped at storage time.\n"
    "  - A string may appear in only one of the two fields."
)

SYS_STATE_EXTRACT = (
    "You are a persona state extraction assistant.\n"
    "Extract user states from the current user turn.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC
)

SYS_EPISODE_EXTRACT = (
    "You are an episode extraction assistant.\n"
    "Summarize the recent conversation into one episode.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC
)

SYS_TRAIT_EXTRACT = (
    "You are a persona trait extraction assistant.\n"
    "Infer at most one new long-term trait from the accumulated recent evidence.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC
)

SYS_TRAIT_EVIDENCE_5A = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships involving a newly extracted Trait and listed State, Episode, and Trait nodes.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_TRAIT_EXTRA_REL_5B = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships involving a newly extracted Trait and additional listed State/Episode nodes.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_STATE_STATE_5C = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships between listed State-State pairs.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_STATE_EPISODE_5D = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships between listed State-Episode pairs.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_STATE_NEW_REL = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships between listed State nodes.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_EPISODE_NEW_REL = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships involving a newly extracted Episode and listed Episode/State nodes.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
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
    # ② is extraction-only — no `judgments` field. All new-state relation
    # judgments (new↔new + new↔prev) are produced by ②b.
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
    # ③ is extraction-only — no `judgments` field. All new-episode relation
    # judgments (episode↔chunk_states + episode↔previous_episode) are produced
    # by ③b.
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
                    "content": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "domain_label": {"type": "array", "items": {"type": "string"}},
                    "scope": {"type": "string"},
                },
                "required": ["id", "content", "keywords", "domain_label", "scope"],
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
    """Updater for graph-internal calls ②③④⑤."""

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
        # Read-only reference to the rolling context cache; used by ②
        # for the PRIOR CONTEXT block.
        self._context_cache = context_cache

        self._current_chunk: Optional[ChunkRecord] = None
        self._completed_chunks: deque = deque(maxlen=config.TRAIT_EXTRACTION_CHUNKS)
        self._recent_turns: deque = deque(maxlen=config.STATE_EXTRACTION_H)
        self._previous_state_ids: List[str] = []
        self._previous_episode_id: Optional[str] = None
        self._previous_trait_id: Optional[str] = None

        self._user_turn_count = 0
        self._chunk_count = 0

        self._call_type_tokens: Dict[str, Dict[str, int]] = {}

        self._state_extracted_count = 0
        self._episode_extracted_count = 0
        self._trait_extracted_count = 0
        self._5a_call_count = 0
        self._5b_call_count = 0
        self._5c_call_count = 0
        self._5d_call_count = 0
        self._state_new_rel_call_count = 0
        self._episode_new_rel_call_count = 0

        # Pending trait ID carried across ⑤a → ⑤b chain
        self._pending_trait_node_id: Optional[str] = None

        # Pending data carried across state/episode → prev_rel chain
        self._pending_state_new_rel_new_ids: List[str] = []
        # ③b pending data: holds new_episode_id, previous_episode_id (or None),
        # and chunk_state_ids of the just-finalized chunk.
        self._pending_episode_new_rel: Optional[Dict[str, Any]] = None

        # ⑤c / ⑤d pending new-node tracking. At flush time, candidate pairs
        # touching at least one pending new node are scored once
        # (sem_topK ∪ lex_topK over pairs, not per-anchor).
        self._pending_new_state_ids_5c: Set[str] = set()
        self._pending_new_state_ids_5d: Set[str] = set()
        self._pending_new_episode_ids_5d: Set[str] = set()

    # ------------------------------------------------------------------
    # Public state helpers
    # ------------------------------------------------------------------

    def ensure_current_chunk(self, conv_id: int) -> None:
        if self._current_chunk is None:
            self._current_chunk = ChunkRecord(conv_id=conv_id)

    def register_turn(
        self,
        user_utterance: str,
        gt_response: str,
        conv_id: int,
        turn_id: int,
        context_node_id: str,
    ) -> None:
        self.ensure_current_chunk(conv_id)
        turn = TurnRecord(
            user_utterance=user_utterance,
            gt_response=gt_response,
            conv_id=conv_id,
            turn_id=turn_id,
            context_node_id=context_node_id,
        )
        self._current_chunk.turns.append(turn)
        self._current_chunk.context_ids.append(context_node_id)
        self._current_chunk.last_turn_id = turn_id
        self._recent_turns.append(turn)
        self._user_turn_count += 1

    # ------------------------------------------------------------------
    # Step-wise preparation
    # ------------------------------------------------------------------

    def prepare_pre_turn_call(
        self,
        conv_id: int,
        turn_id: int,
        session_id: int,
        global_turn: int,
    ) -> Optional[PendingLLMCall]:
        if self._current_chunk is None:
            self.ensure_current_chunk(conv_id)
            return None
        if conv_id == self._current_chunk.conv_id:
            return None
        if not self._current_chunk.turns:
            self._current_chunk = ChunkRecord(conv_id=conv_id)
            return None
        return self._build_episode_call(
            chunk=self._current_chunk,
            session_id=session_id,
            created_at=global_turn,
            anchor_conv_id=conv_id,
            anchor_turn_id=turn_id,
        )

    def prepare_post_turn_call(
        self,
        session_id: int,
        global_turn: int,
        current_conv_id: int,
        current_turn_id: int,
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
            anchor_conv_id=current_conv_id,
            anchor_turn_id=current_turn_id,
        )

    def prepare_finalize_call(
        self,
        session_id: int,
        global_turn: int,
    ) -> Optional[PendingLLMCall]:
        if self._current_chunk is None or not self._current_chunk.turns:
            return None
        return self._build_episode_call(
            chunk=self._current_chunk,
            session_id=session_id,
            created_at=global_turn,
            anchor_conv_id=self._current_chunk.conv_id,
            anchor_turn_id=self._current_chunk.last_turn_id,
        )

    def apply_irrelevant_fallback(self, call: PendingLLMCall) -> None:
        """Mark every expected pair as IRRELEVANT after retry exhaustion, so
        future ⑤b/⑤c/⑤d candidate selection skips them via has_direct_edge."""
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

    def apply_call_result(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        if call.call_type == "state":
            self._apply_state_result(call, result)
            new_state_ids = list(self._previous_state_ids)
            # Trigger ②b iff there is at least one judgeable pair:
            #   |new| ≥ 2  OR  (|new| ≥ 1 ∧ |prev| ≥ 1).
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
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                    previous_state_ids=prev_state_ids,
                    new_state_ids=new_state_ids,
                )
            return None

        if call.call_type == "state_new_rel":
            self._apply_state_new_rel_result(result, call.id_map)
            return None

        if call.call_type == "episode":
            old_prev_episode_id = self._previous_episode_id
            self._apply_episode_result(call, result)
            new_episode_id = self._previous_episode_id
            # Trigger ③b iff there is at least one judgeable pair:
            #   |chunk_state_ids| ≥ 1  OR  previous_episode exists.
            # The just-finalized chunk lives at _completed_chunks[-1].
            if (
                new_episode_id
                and new_episode_id != old_prev_episode_id
                and self._completed_chunks
            ):
                chunk_state_ids = list(self._completed_chunks[-1].state_ids)
                prev_id_for_call = (
                    old_prev_episode_id if old_prev_episode_id else None
                )
                if chunk_state_ids or prev_id_for_call:
                    return self._build_episode_new_rel_call(
                        session_id=call.session_id,
                        created_at=call.created_at,
                        anchor_conv_id=call.anchor_conv_id,
                        anchor_turn_id=call.anchor_turn_id,
                        new_episode_id=new_episode_id,
                        previous_episode_id=prev_id_for_call,
                        chunk_state_ids=chunk_state_ids,
                    )
            if self._chunk_count > 0 and self._chunk_count % self._cfg.TRAIT_EXTRACTION_CHUNKS == 0:
                return self._build_trait_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "episode_new_rel":
            self._apply_episode_new_rel_result(result, call.id_map)
            if self._chunk_count > 0 and self._chunk_count % self._cfg.TRAIT_EXTRACTION_CHUNKS == 0:
                return self._build_trait_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "trait":
            created = self._apply_trait_result(call, result)
            if created is not None:
                self._pending_trait_node_id = created
                return self._build_5a_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            # No new trait: skip ⑤a/⑤b, proceed directly to ⑤c
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._pending_new_state_ids_5c:
                return self._build_5c_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and (
                self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d
            ):
                return self._build_5d_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "trait_evidence_5a":
            self._apply_5a_result(result, call.id_map)
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._pending_trait_node_id:
                return self._build_5b_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            self._pending_trait_node_id = None
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._pending_new_state_ids_5c:
                return self._build_5c_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and (
                self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d
            ):
                return self._build_5d_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "trait_extra_rel_5b":
            self._apply_5b_result(result, call.id_map)
            self._pending_trait_node_id = None
            if self._pending_new_state_ids_5c:
                return self._build_5c_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            if self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d:
                return self._build_5d_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "state_state_5c":
            self._apply_5c_result(result, call.id_map)
            if self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d:
                return self._build_5d_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "state_episode_5d":
            self._apply_5d_result(result, call.id_map, call.id_map_b)
            return None

        return None

    # ------------------------------------------------------------------
    # Sequential wrapper
    # ------------------------------------------------------------------

    def execute_call(self, call: PendingLLMCall) -> Dict:
        """Execute the LLM call with empty-judgment retry.

        If `expected_judgment_count > 0` and `judgments` is empty, retry up
        to `JUDGMENT_RETRY` times (each retry still allows `JSON_RETRY`
        parse retries). After exhaustion the caller's IRRELEVANT-fallback
        handles missing edges; a non-empty array (even all-IRRELEVANT) is
        not retried.
        """
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
        stats = {
            "num_state_nodes_extracted": self._state_extracted_count,
            "num_episode_nodes_extracted": self._episode_extracted_count,
            "num_trait_nodes_extracted": self._trait_extracted_count,
            "num_5a_trait_evidence_calls": self._5a_call_count,
            "num_5b_trait_extra_rel_calls": self._5b_call_count,
            "num_5c_state_state_rel_calls": self._5c_call_count,
            "num_5d_state_episode_rel_calls": self._5d_call_count,
            "num_state_new_rel_calls": self._state_new_rel_call_count,
            "num_episode_new_rel_calls": self._episode_new_rel_call_count,
        }
        self._state_extracted_count = 0
        self._episode_extracted_count = 0
        self._trait_extracted_count = 0
        self._5a_call_count = 0
        self._5b_call_count = 0
        self._5c_call_count = 0
        self._5d_call_count = 0
        self._state_new_rel_call_count = 0
        self._episode_new_rel_call_count = 0
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
        self._state_extracted_count = 0
        self._episode_extracted_count = 0
        self._trait_extracted_count = 0
        self._5a_call_count = 0
        self._5b_call_count = 0
        self._5c_call_count = 0
        self._5d_call_count = 0
        self._state_new_rel_call_count = 0
        self._episode_new_rel_call_count = 0
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
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> PendingLLMCall:
        # ②: per-turn extraction. The current turn is the most recent registered
        # TurnRecord; PRIOR CONTEXT comes from the context_cache, filtered to
        # exclude the current (conv_id, turn_id).
        recent_turns = list(self._recent_turns)
        if not recent_turns:
            current_user_utterance = "(none)"
            assistant_response_block = "(none)"
            current_conv_id = anchor_conv_id
            current_turn_id = anchor_turn_id
        else:
            current_turn = recent_turns[-1]
            current_user_utterance = current_turn.user_utterance
            assistant_response_block = current_turn.gt_response
            current_conv_id = current_turn.conv_id
            current_turn_id = current_turn.turn_id

        prior_pairs = self._context_cache.get_prior_pairs(
            current_conv_id=current_conv_id,
            current_turn_id=current_turn_id,
            n=self._cfg.STATE_REF_CONTEXT_TURNS,
        )
        prior_context_block = self._format_prior_pairs(
            prior_pairs, anchor_conv_id, anchor_turn_id
        )

        prompt = (
            f"Task: Extract up to {self._cfg.STATE_MAX_COUNT} persona State(s) from the CURRENT USER UTTERANCE only. "
            "If none, return an empty states list. Extract useful persona-relevant signals as States when they are not stable enough to be Traits. "
            "Do not invent or speculate.\n"
            "\n"
            "Use prior context and the assistant response only to resolve references. "
            "Do not extract information stated only outside the current user utterance.\n"
            "\n"
            "Do not extract:\n"
            "  - greetings, thanks, or conversational behavior,\n"
            "  - descriptions of the current query rather than the user,\n"
            "  - assistant-side information,\n"
            "  - transient emotions with no impact on a task, decision, constraint, or safety issue,\n"
            "  - stable Traits (cross-situational disposition handled by a later stage),\n"
            "  - past episodes without a currently valid implication.\n"
            "\n"
            "Each State must begin with \"The user\", be one concise sentence, and express a currently valid condition, constraint, goal, stance, or preference. "
            "Rewrite episodic descriptions as implied current States, not raw narration. "
            "Use placeholder ids new_0, new_1, ... in extraction order.\n"
            "\n"
            "Metadata:\n"
            "scope:\n"
            "  BROAD: likely to affect decisions across multiple future tasks or topics, even when the future query does not explicitly mention this state.\n"
            "  NARROW: mainly affects the current task, topic, or short-term situation.\n"
            "If uncertain, choose NARROW.\n"
            "\n"
            "recall_priority:\n"
            "  HIGH: ignoring this state would make the response clearly wrong, unsafe, inconsistent with an explicit constraint, or noticeably frustrating, and the user would reasonably expect it to be remembered.\n"
            "  LOW: useful context, but not necessary for an appropriate response.\n"
            "If uncertain, choose LOW.\n"
            "\n"
            "Follow the label rules:\n"
            f"{_LABEL_DISCIPLINE_BLOCK}\n"
            "\n"
            "Also provide:\n"
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific single-token nouns (no whitespace, no multi-word phrases)\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} single-token topical labels (no whitespace, no multi-word phrases)\n"
            "\n"
            "[PRIOR CONTEXT — disambiguation only; do not extract from here]\n"
            f"{prior_context_block}\n"
            "\n"
            "[ASSISTANT RESPONSE — read-only context; do not extract from here]\n"
            f"{assistant_response_block}\n"
            "\n"
            "[CURRENT USER UTTERANCE — extract from here only]\n"
            f"{current_user_utterance}"
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
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            # ② is extraction-only; the schema has no `judgments` field. All
            # new-state relation judgments live in ②b. Retry wrapper skips.
            expected_judgment_count=0,
        )

    def _build_episode_call(
        self,
        chunk: ChunkRecord,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> PendingLLMCall:
        # ③: extraction-only. Chunk states are NOT exposed here — ③b judges
        # them against the new episode separately.
        prompt = (
            "Task: Create exactly one Episode from the RECENT CONVERSATION. Summarize what happened in the conversation. Do not invent or speculate.\n"
            "\n"
            "Do not extract generalized persona traits or separate state nodes. Include user traits, preferences, or conditions only when they are part of the concrete episode being summarized.\n"
            "\n"
            "Each episode must:\n"
            "- begin with \"The user\",\n"
            "- be 1–2 sentences,\n"
            "- summarize concrete events, discussed topics, actions, and developments,\n"
            "- describe the episode itself, not generalized persona traits.\n"
            "\n"
            "Metadata:\n"
            "scope:\n"
            "  BROAD: the episode reveals or confirms information likely to affect decisions across multiple future tasks or topics.\n"
            "  NARROW: the episode is mainly tied to the current task, topic, or short-term conversation thread.\n"
            "If uncertain, choose NARROW.\n"
            "\n"
            "Follow the label rules:\n"
            f"{_LABEL_DISCIPLINE_BLOCK}\n"
            "\n"
            "Also provide:\n"
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific single-token nouns (no whitespace, no multi-word phrases)\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} single-token topical labels (no whitespace, no multi-word phrases)\n"
            "\n"
            "Assign the episode the placeholder id: new_episode.\n"
            "\n"
            "[RECENT CONVERSATION]\n"
            f"{self._format_turns(chunk.turns, anchor_conv_id, anchor_turn_id)}"
        )
        return PendingLLMCall(
            call_type="episode",
            system_prompt=SYS_EPISODE_EXTRACT,
            user_prompt=prompt,
            guided_json=EPISODE_EXTRACT_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_EPISODE,
            log_dir="call_3_episode",
            created_at=created_at,
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            # ③ is extraction-only; the schema has no `judgments` field. All
            # new-episode relation judgments live in ③b. Retry wrapper skips.
            expected_judgment_count=0,
        )

    def _build_trait_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> PendingLLMCall:
        # ④: only `[Recent conversations]` is exposed. Chunk states, chunk
        # memories, and the previous trait are NOT in the prompt — the trait
        # is inferred directly from raw conversation; relating it to existing
        # nodes is ⑤a's job.
        recent_chunks = list(self._completed_chunks)
        turns: List[TurnRecord] = []
        for chunk in recent_chunks:
            turns.extend(chunk.turns)

        conversations_block = self._format_turns(turns, anchor_conv_id, anchor_turn_id)

        prompt = (
            f"Task: Infer at most {self._cfg.TRAIT_MAX_COUNT} persona Trait(s) from the recent conversations below. "
            "If the recent conversations do not reveal a clear new persistent pattern, return an empty traits list. Do not invent or speculate.\n"
            "\n"
            "Use the \"in general\" test: a trait should still be true if you asked the user about themselves \"in general\" with no specific time, place, or context attached.\n"
            "\n"
            "A trait should:\n"
            "- begin with \"The user\",\n"
            "- be 2–3 complete sentences,\n"
            "- be inferable from the recent conversations as a likely persistent pattern, not just a one-time event or momentary feeling.\n"
            "\n"
            "Trait metadata:\n"
            "scope:\n"
            "  BROAD  : the trait applies across all domains of the user's life — it would shape the user's approach regardless of the subject being discussed\n"
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
            "[Recent conversations]\n"
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
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
        )

    def _build_5a_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> PendingLLMCall:
        recent_chunks = list(self._completed_chunks)
        state_ids: List[str] = []
        episode_ids: List[str] = []
        for chunk in recent_chunks:
            state_ids.extend(chunk.state_ids)
            if chunk.episode_id:
                episode_ids.append(chunk.episode_id)

        trait = self._graph.get_node(self._pending_trait_node_id) if self._pending_trait_node_id else None
        # id_map layout: [0]=new_trait, [1..]=states, [1+n_states..]=memories,
        # [1+n_states+n_mems]=prev_trait (if any).
        new_trait_block, trait_ids = self._format_node_list_indexed(
            [trait.node_id] if trait else [], anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        recent_states_block, s_ids = self._format_node_list_indexed(
            state_ids, anchor_conv_id, anchor_turn_id, start_idx=len(trait_ids),
        )
        recent_episodes_block, e_ids = self._format_node_list_indexed(
            episode_ids, anchor_conv_id, anchor_turn_id, start_idx=len(trait_ids) + len(s_ids),
        )
        previous_trait_block, pt_ids = self._format_node_list_indexed(
            [self._previous_trait_id] if self._previous_trait_id else [],
            anchor_conv_id, anchor_turn_id,
            start_idx=len(trait_ids) + len(s_ids) + len(e_ids),
        )
        id_map = trait_ids + s_ids + e_ids + pt_ids
        new_trait_idx = 0
        prev_trait_idx = len(trait_ids) + len(s_ids) + len(e_ids) if pt_ids else None
        expected_5a = len(s_ids) + len(e_ids) + (1 if pt_ids else 0)

        state_start = len(trait_ids)
        state_end = len(trait_ids) + len(s_ids) - 1
        episode_start = len(trait_ids) + len(s_ids)
        episode_end = len(trait_ids) + len(s_ids) + len(e_ids) - 1
        prompt = (
            "You will judge direct evidence relationships involving the newly extracted trait.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"Index 0 is the new trait. "
            f"Indices {state_start}..{state_end} are recent states. "
            f"Indices {episode_start}..{episode_end} are recent episodes. "
            + (f"Index {prev_trait_idx} is the previous trait.\n" if prev_trait_idx is not None else "\n")
            + "\n"
            "Judge each listed state↔new_trait pair, each listed episode↔new_trait pair, and the previous_trait↔new_trait pair when a previous trait exists.\n"
            "\n"
            "Direction rule:\n"
            "- source_id = listed state, episode, or previous trait index; target_id = 0.\n"
            "- SUPPORT, CONTRADICT, and IRRELEVANT are semantically symmetric, but keep this source→target direction for consistency.\n"
            "- For previous_trait↔new_trait SHIFT_TO, source_id = previous trait index and target_id = 0, following older information → newer replacement.\n"
            "- For cross-type SHIFT_TO, the source state or episode provides evidence that updates, replaces, or invalidates the new trait.\n"
            "\n"
            "Decision priority:\n"
            "- First choose IRRELEVANT if the pair has no clear evidential force.\n"
            "- If related, choose SHIFT_TO only when the source clearly updates, replaces, or invalidates the target.\n"
            "- If not replacement but clearly in tension, choose CONTRADICT.\n"
            "- If the source grounds, confirms, generalizes into, or reinforces the new trait, choose SUPPORT.\n"
            "- When uncertain, choose IRRELEVANT.\n"
            "\n"
            "Topical overlap alone is not SUPPORT. For states and episodes, the source must provide concrete evidence for the new trait. For previous_trait↔new_trait, use SHIFT_TO only when the new trait replaces the previous trait on the same underlying dimension.\n"
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
            f"Output exactly {expected_5a} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs. Use the integer indices shown above for source_id and target_id."
        )
        # Canonical pairs for IRRELEVANT-fallback:
        # state→new_trait, episode→new_trait, prev_trait→new_trait.
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
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            expected_judgment_count=expected_5a,
            id_map=id_map,
            expected_pairs=expected_pairs,
        )

    def _build_5b_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> Optional[PendingLLMCall]:
        if not self._pending_trait_node_id:
            return None
        trait = self._graph.get_node(self._pending_trait_node_id)
        if trait is None:
            return None

        # Collect recent 2-chunk state/episode IDs to exclude (already handled by ⑤a)
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

        # id_map layout: [0]=trait, [1..n_s]=candidate states, [n_s+1..]=candidate memories.
        trait_block, trait_ids = self._format_node_list_indexed(
            [trait.node_id], anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        cand_state_block, cs_ids = self._format_node_list_indexed(
            [n.node_id for n in candidate_states], anchor_conv_id, anchor_turn_id,
            start_idx=len(trait_ids),
        )
        cand_ep_block, ce_ids = self._format_node_list_indexed(
            [n.node_id for n in candidate_episodes], anchor_conv_id, anchor_turn_id,
            start_idx=len(trait_ids) + len(cs_ids),
        )
        id_map = trait_ids + cs_ids + ce_ids
        num_candidates = len(cs_ids) + len(ce_ids)
        state_start = len(trait_ids)
        state_end = len(trait_ids) + len(cs_ids) - 1
        episode_start = len(trait_ids) + len(cs_ids)
        episode_end = len(trait_ids) + len(cs_ids) + len(ce_ids) - 1

        prompt = (
            "You will judge direct evidence relationships involving the newly extracted trait and additional candidate nodes.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"Index 0 is the new trait. "
            f"Indices {state_start}..{state_end} are candidate states. "
            f"Indices {episode_start}..{episode_end} are candidate episodes.\n"
            "\n"
            "Judge each listed candidate directly against the new trait.\n"
            "\n"
            "Direction rule:\n"
            "- source_id = candidate state or episode index, target_id = 0.\n"
            "- SUPPORT, CONTRADICT, and IRRELEVANT are semantically symmetric, but keep this candidate→trait direction for consistency.\n"
            "- For cross-type SHIFT_TO, the source state or episode provides evidence that updates, replaces, or invalidates the new trait.\n"
            "\n"
            "Decision priority:\n"
            "- First choose IRRELEVANT if the pair has no clear evidential force.\n"
            "- If related, choose SHIFT_TO only when the source clearly updates, replaces, or invalidates the target.\n"
            "- If not replacement but clearly in tension, choose CONTRADICT.\n"
            "- If the source grounds, confirms, generalizes into, or reinforces the new trait, choose SUPPORT.\n"
            "- When uncertain, choose IRRELEVANT.\n"
            "\n"
            "The listed candidates were retrieved by semantic or lexical similarity, but retrieval similarity is not evidence. Topical overlap alone is not SUPPORT; the source must provide concrete evidence for the new trait.\n"
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
            f"Output exactly {num_candidates} judgments — one for each listed candidate, in the listed order. Do not skip or duplicate candidates. Use the integer indices shown above for source_id and target_id."
        )
        # Canonical pairs: candidate→new_trait for every listed candidate.
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
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
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
        anchor_conv_id: int,
        anchor_turn_id: int,
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

        # Build ordered unique id_map and pair list with integer indices.
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
            old_elapsed = self._elapsed_str(old_node.conv_id, old_node.turn_id, anchor_conv_id, anchor_turn_id)
            new_elapsed = self._elapsed_str(new_node.conv_id, new_node.turn_id, anchor_conv_id, anchor_turn_id)
            pair_lines.append(
                f"Pair {pair_num}\n"
                f"- older_state: [{old_idx}] [{old_elapsed}]: {old_node.content}\n"
                f"- newer_state: [{new_idx}] [{new_elapsed}]: {new_node.content}"
            )

        if not pair_lines:
            return None

        pair_block = "\n\n".join(pair_lines)
        num_pairs = len(pair_lines)
        prompt = (
            "You will judge direct evidence relationships for candidate State-State pairs.\n"
            "Each pair is currently unconnected in the graph.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            "The listed pairs were surfaced by semantic and lexical candidate mining; similarity alone does not imply any relation.\n"
            "\n"
            "Each pair is shown in chronological order: older_state first, newer_state second.\n"
            "\n"
            "Direction rule:\n"
            "- For every relation, source_id = older_state index and target_id = newer_state index.\n"
            "- For SHIFT_TO, this means older state → newer replacement state.\n"
            "- For SUPPORT, CONTRADICT, and IRRELEVANT, the relation is semantically symmetric, but keep the older→newer output direction for consistency.\n"
            "\n"
            "Decision priority:\n"
            "- First choose IRRELEVANT if the pair has no clear evidential force.\n"
            "- If related, choose SHIFT_TO only when the newer state clearly replaces the older state on the same underlying condition, constraint, stance, goal, or preference.\n"
            "- If not replacement but clearly in tension, choose CONTRADICT.\n"
            "- If the two states reinforce the same persona signal, choose SUPPORT.\n"
            "- When uncertain, choose IRRELEVANT.\n"
            "\n"
            "Calibration note: Topical similarity, retrieval similarity, or temporal proximity alone is not SUPPORT. A newer state that only adds detail, changes topic, or expresses a short-lived preference does not replace an older broader condition unless it explicitly invalidates it.\n"
            "\n"
            "[Candidate State-State Pairs]\n"
            f"{pair_block}\n"
            "\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs. Use the integer indices shown above for source_id and target_id."
        )
        return PendingLLMCall(
            call_type="state_state_5c",
            system_prompt=SYS_STATE_STATE_5C,
            user_prompt=prompt,
            guided_json=JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_STATE_STATE_5C,
            log_dir="call_5c_state_state_rel",
            created_at=created_at,
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            expected_judgment_count=num_pairs,
            id_map=id_map,
            expected_pairs=ss_expected_pairs,
        )

    def _build_5d_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> Optional[PendingLLMCall]:
        if not (self._pending_new_state_ids_5d or self._pending_new_episode_ids_5d):
            return None

        pending_new_s = self._pending_new_state_ids_5d
        pending_new_e = self._pending_new_episode_ids_5d
        self._pending_new_state_ids_5d = set()
        self._pending_new_episode_ids_5d = set()

        all_states = self._graph.get_nodes_by_type(NODE_S)
        all_episodes = self._graph.get_nodes_by_type(NODE_E)
        state_by_id = {n.node_id: n for n in all_states}
        episode_by_id = {n.node_id: n for n in all_episodes}

        candidate_pairs: List[Tuple[Node, Node]] = []  # (state_node, episode_node)
        seen_pair_keys: Set[Tuple[str, str]] = set()
        for sid in pending_new_s:
            s_node = state_by_id.get(sid)
            if s_node is None:
                continue
            for e_node in all_episodes:
                if self._graph.has_direct_edge(sid, e_node.node_id):
                    continue
                key = (sid, e_node.node_id)
                if key in seen_pair_keys:
                    continue
                seen_pair_keys.add(key)
                candidate_pairs.append((s_node, e_node))
        for eid in pending_new_e:
            e_node = episode_by_id.get(eid)
            if e_node is None:
                continue
            for s_node in all_states:
                if self._graph.has_direct_edge(s_node.node_id, eid):
                    continue
                key = (s_node.node_id, eid)
                if key in seen_pair_keys:
                    continue
                seen_pair_keys.add(key)
                candidate_pairs.append((s_node, e_node))

        if not candidate_pairs:
            return None

        selected_pairs = self._pair_topk_union(
            candidate_pairs, self._cfg.STATE_EPISODE_EXTRA_REL_TOPK,
        )
        if not selected_pairs:
            return None

        # ⑤d uses separate index spaces: id_map for states, id_map_b for episodes.
        id_map: List[str] = []   # state indices
        id_map_b: List[str] = []  # episode indices
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
        sm_expected_pairs: List[Tuple[str, str]] = []
        for pair_num, (s_node, e_node) in enumerate(selected_pairs):
            s_id = s_node.node_id
            e_id = e_node.node_id
            s_idx = _get_s_idx(s_id)
            e_idx = _get_e_idx(e_id)
            # Storage canonical for ⑤d is m → s.
            sm_expected_pairs.append((e_id, s_id))
            elapsed_s = self._elapsed_str(s_node.conv_id, s_node.turn_id, anchor_conv_id, anchor_turn_id)
            elapsed_e = self._elapsed_str(e_node.conv_id, e_node.turn_id, anchor_conv_id, anchor_turn_id)
            same_conv = (s_node.conv_id == e_node.conv_id)
            if same_conv:
                relation_context = "same conversation."
            else:
                gap_str = _format_conv_gap(
                    s_node.conv_id, e_node.conv_id,
                    self._cfg.TIME_PER_CONV_ID_HOURS,
                )
                relation_context = f"different conversations, {gap_str} apart."
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
            "You will judge direct evidence relationships for candidate State-Episode pairs.\n"
            "Each pair is currently unconnected in the graph.\n"
            "\n"
            "State nodes use one integer index space (state_id); episode nodes use a separate integer index space (episode_id). The listed pairs were surfaced by semantic and lexical candidate mining; similarity alone does not imply any relation.\n"
            "\n"
            "Direction rule:\n"
            "- Output one judgment per listed pair using its state_id and episode_id.\n"
            "- Interpret the relation as an episode→state evidence relation: the episode is the evidence source, and the state is the persona-state target.\n"
            "- For cross-type SHIFT_TO, this means the episode provides evidence that updates, replaces, or invalidates the listed state.\n"
            "\n"
            "Decision priority:\n"
            "- First choose IRRELEVANT if the pair has no clear evidential force.\n"
            "- If related, choose SHIFT_TO only when the episode clearly updates, replaces, or invalidates the state.\n"
            "- If not replacement but the episode clearly conflicts with the state, choose CONTRADICT.\n"
            "- If the episode concretely grounds, confirms, or reinforces the state, choose SUPPORT.\n"
            "- When uncertain, choose IRRELEVANT.\n"
            "\n"
            "Calibration note: Topical similarity, retrieval similarity, same-conversation co-extraction, or temporal adjacency alone is not SUPPORT. The episode must contain concrete evidence that would still ground or challenge the state if read in isolation.\n"
            "\n"
            "Each pair includes a \"Relation context\" line. Use it only as a caution signal: same-conversation pairs need extra scrutiny, while different-conversation pairs may provide stronger independent evidence.\n"
            "\n"
            "[Candidate State-Episode Pairs]\n"
            f"{pair_block}\n"
            "\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs. Use the integer indices shown above for state_id and episode_id."
        )
        return PendingLLMCall(
            call_type="state_episode_5d",
            system_prompt=SYS_STATE_EPISODE_5D,
            user_prompt=prompt,
            guided_json=STATE_EPISODE_JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_STATE_EPISODE_5D,
            log_dir="call_5d_state_episode_rel",
            created_at=created_at,
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            expected_judgment_count=num_pairs,
            id_map=id_map,
            id_map_b=id_map_b,
            expected_pairs=sm_expected_pairs,
        )

    def _get_recent_prev_state_ids(self, exclude_ids: Set[str], k: int) -> List[str]:
        """Return up to k most-recently created state IDs, excluding exclude_ids."""
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
        anchor_conv_id: int,
        anchor_turn_id: int,
        previous_state_ids: List[str],
        new_state_ids: List[str],
    ) -> PendingLLMCall:
        # ②b: judges every new↔new pair (when |new| ≥ 2) and every
        # (new, prev) pair (when |prev| ≥ 1). Trigger condition lives in
        # apply_call_result; this builder assumes at least one pair exists.
        self._pending_state_new_rel_new_ids = list(new_state_ids)
        prev_block, prev_ids = self._format_node_list_indexed(
            previous_state_ids, anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        new_block, new_ids = self._format_node_list_indexed(
            new_state_ids, anchor_conv_id, anchor_turn_id, start_idx=len(prev_ids),
        )
        id_map = prev_ids + new_ids
        n_prev_valid = len(prev_ids)
        n_new_valid = len(new_ids)
        n_new = n_new_valid
        n_prev = n_prev_valid
        expected = (n_new * (n_new - 1)) // 2 + n_new * n_prev
        prev_desc = _index_descriptor(0, n_prev_valid, "previous state")
        new_desc = _index_descriptor(n_prev_valid, n_new_valid, "new state")
        index_descriptor = " ".join(s for s in (prev_desc, new_desc) if s)
        prompt = (
            "You will judge direct evidence relationships involving newly extracted states.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"{index_descriptor}\n"
            "\n"
            "Judge every listed previous↔new pair. If more than one new state is listed, also judge each unordered new↔new pair exactly once. Do not output both (A,B) and (B,A).\n"
            "\n"
            "Direction rule:\n"
            "- For previous↔new pairs: source_id = previous state index, target_id = new state index.\n"
            "- For previous↔new SHIFT_TO, this means older previous state → newer replacement state.\n"
            "- For previous↔new SUPPORT, CONTRADICT, and IRRELEVANT, the relation is semantically symmetric, but still output source_id = previous state index and target_id = new state index for consistency.\n"
            "- For new↔new SHIFT_TO: source_id = older state index as described in the content, target_id = newer replacement state index.\n"
            "- For new↔new SUPPORT, CONTRADICT, and IRRELEVANT: source_id = earlier-listed new state index, target_id = later-listed new state index.\n"
            "\n"
            "Decision priority:\n"
            "- First choose IRRELEVANT if the pair has no clear evidential force.\n"
            "- If related, choose SHIFT_TO only for a clear old→new replacement of the same underlying state.\n"
            "- If not replacement but clearly in tension, choose CONTRADICT.\n"
            "- If the two states reinforce the same persona signal, choose SUPPORT.\n"
            "- When uncertain, choose IRRELEVANT.\n"
            "\n"
            "For new↔new pairs, co-extracted states usually coexist; use SHIFT_TO only when the current utterance explicitly states temporal replacement.\n"
            "\n"
            f"Output exactly {expected} judgments — one for each listed pair, in the listed order. Do not skip or duplicate pairs.\n"
            "\n"
            "[Previous States — extracted before the current turn]\n"
            f"{prev_block}\n"
            "\n"
            "[New States — extracted from the current turn]\n"
            f"{new_block}"
        )
        # Canonical pairs for IRRELEVANT-fallback: prev→new, then new(earlier)→new(later).
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
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            expected_judgment_count=expected,
            id_map=id_map,
            expected_pairs=expected_pairs,
        )

    def _build_episode_new_rel_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
        new_episode_id: str,
        previous_episode_id: Optional[str],
        chunk_state_ids: List[str],
    ) -> PendingLLMCall:
        # ③b: judges (new_episode, previous_episode) when prev exists, plus
        # one (new_episode, chunk_state_i) per chunk state in listed order.
        # Direction is fixed to source = new_episode; trigger lives in apply_call_result.
        # id_map layout: [0]=new_episode, [1]=prev_episode (if any), [2..]=chunk_states.
        self._pending_episode_new_rel = {
            "new_episode_id": new_episode_id,
            "previous_episode_id": previous_episode_id,
            "chunk_state_ids": list(chunk_state_ids),
        }

        # Build id_map: new_episode first, then prev_episode (if any), then chunk_states.
        new_block, new_ids = self._format_node_list_indexed(
            [new_episode_id], anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        prev_block, prev_ids = self._format_node_list_indexed(
            [previous_episode_id] if previous_episode_id else [],
            anchor_conv_id, anchor_turn_id, start_idx=len(new_ids),
        )
        chunk_states_block, state_ids = self._format_node_list_indexed(
            chunk_state_ids, anchor_conv_id, anchor_turn_id,
            start_idx=len(new_ids) + len(prev_ids),
        )
        id_map = new_ids + prev_ids + state_ids

        n_prev = len(prev_ids)
        n_states = len(state_ids)
        expected = n_prev + n_states

        # Compute the target index range for the prev_episode (if any) and states.
        prev_idx = len(new_ids) if n_prev else None  # index of prev_episode in id_map
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
            "Direction rule:\n"
            "- For new_episode↔previous_episode pairs with SUPPORT, CONTRADICT, or IRRELEVANT: source_id = 0, target_id = previous episode index.\n"
            "- For new_episode↔previous_episode pairs with SHIFT_TO: source_id = previous episode index, target_id = 0.\n"
            "- For new_episode↔chunk_state pairs: source_id = 0, target_id = chunk state index for every relation.\n"
            "- For cross-type SHIFT_TO in a new_episode↔chunk_state pair, this means the new episode provides evidence that updates, replaces, or invalidates the listed state.\n"
            "\n"
            "Decision priority:\n"
            "- First choose IRRELEVANT if the pair has no clear evidential force.\n"
            "- If related, choose SHIFT_TO when the source information clearly updates, replaces, or shifts the target information according to the direction rule above.\n"
            "- If not replacement but clearly in tension, choose CONTRADICT.\n"
            "- If the new episode concretely continues, confirms, grounds, or reinforces the target, choose SUPPORT.\n"
            "- When uncertain, choose IRRELEVANT.\n"
            "\n"
            "For new_episode↔chunk_state pairs, same-conversation co-extraction alone is not SUPPORT. The episode must contain concrete evidence that grounds or confirms the state.\n"
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
            f"Output exactly {expected} judgments in this order:\n"
            "  1) the (new_episode, previous_episode) judgment, if a previous episode exists;\n"
            "  2) one (new_episode, chunk_state_i) judgment for each listed chunk state, in the listed order.\n"
            "Do not invent states, episodes, or judgments."
        )
        # Canonical pairs for IRRELEVANT-fallback: new_episode → previous_episode, then new_episode → each chunk state.
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
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            expected_judgment_count=expected,
            id_map=id_map,
            expected_pairs=expected_pairs,
        )

    # ------------------------------------------------------------------
    # Result appliers
    # ------------------------------------------------------------------

    def _apply_state_result(self, call: PendingLLMCall, result: Dict) -> None:
        # ② is extraction-only. The schema has no `judgments` field; all
        # new-state relation judgments are produced by the chained ②b call
        # (see _build_state_new_rel_call / _apply_state_new_rel_result).
        raw_states = result.get("states", []) if isinstance(result, dict) else []
        new_state_ids: List[str] = []
        recent_turns = list(self._recent_turns)

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
            node = Node(
                node_id=HeterogeneousGraph.new_node_id(),
                node_type=NODE_S,
                content=content,
                keywords=keywords,
                domain_label=domain_label,
                embedding=self._embed_text(content),
                created_at=call.created_at,
                session_id=call.session_id,
                conv_id=recent_turns[-1].conv_id if recent_turns else call.anchor_conv_id,
                turn_id=recent_turns[-1].turn_id if recent_turns else call.anchor_turn_id,
                scope=_validate_scope(item.get("scope")),
                recall_priority=_validate_recall_priority(item.get("recall_priority")),
                retrieval_count=1,
            )
            self._graph.add_node(node)
            new_state_ids.append(node.node_id)

            for turn in recent_turns:
                self._graph.add_source_edge(turn.context_node_id, node.node_id)
            if self._current_chunk is not None:
                self._current_chunk.state_ids.append(node.node_id)

        self._previous_state_ids = new_state_ids
        self._state_extracted_count += len(new_state_ids)

        # Update reservoirs
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
            session_id=call.session_id,
            conv_id=chunk.conv_id,
            turn_id=chunk.last_turn_id,
            scope=_validate_scope(episode_obj.get("scope")),
            retrieval_count=1,
        )
        self._graph.add_node(episode_node)
        chunk.episode_id = episode_node.node_id
        for context_id in chunk.context_ids:
            self._graph.add_source_edge(context_id, episode_node.node_id)

        # ③ is extraction-only. All new-episode relation judgments
        # (episode↔chunk_states + episode↔previous_episode) are produced by ③b.
        self._previous_episode_id = episode_node.node_id
        self._episode_extracted_count += 1
        self._finalize_chunk_record()

        # Update sm reservoir
        self._update_sm_reservoir([], [episode_node.node_id])

    def _apply_trait_result(self, call: PendingLLMCall, result: Dict) -> Optional[str]:
        raw_traits = result.get("traits", []) if isinstance(result, dict) else []
        if not raw_traits:
            return None

        item = raw_traits[0]
        content = _validate_content(item)
        if not content:
            return None

        keywords = _validate_keywords(item.get("keywords"), self._cfg.MAX_KEYWORDS)
        domain_label = _validate_domain_labels(
            item.get("domain_label"),
            keywords,
            self._cfg.MIN_DOMAIN_LABELS,
            self._cfg.MAX_DOMAIN_LABELS,
        )
        recent_chunks = list(self._completed_chunks)
        latest_conv = recent_chunks[-1].conv_id if recent_chunks else call.anchor_conv_id
        latest_turn = recent_chunks[-1].last_turn_id if recent_chunks else call.anchor_turn_id

        trait_node = Node(
            node_id=HeterogeneousGraph.new_node_id(),
            node_type=NODE_T,
            content=content,
            keywords=keywords,
            domain_label=domain_label,
            embedding=self._embed_text(content),
            created_at=call.created_at,
            session_id=call.session_id,
            conv_id=latest_conv,
            turn_id=latest_turn,
            scope=_validate_scope(item.get("scope")),
            retrieval_count=1,
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

            rel = _validate_relation(relation)
            if rel is None:
                continue

            if src_t == NODE_T and dst_t == NODE_T:
                # trait ↔ trait: SHIFT_TO normalized to old → new by created_at.
                if rel == EVID_SHIFT_TO:
                    self._store_shift_to_by_created_at(src, dst)
                else:
                    self._store_evidence_edge(src, dst, rel)
            else:
                # state ↔ trait or episode ↔ trait: store canonically s/e → t.
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
            rel = _validate_relation(item.get("relation"))
            if src is None or dst is None or src == dst or rel is None:
                continue
            # Store canonically: candidate → trait.
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
                self._store_shift_to_by_created_at(src, dst)
            else:
                self._store_evidence_edge(src, dst, rel)
        self._5c_call_count += 1

    def _apply_5d_result(self, result: Dict, id_map: List[str], id_map_b: List[str]) -> None:
        # ⑤d uses state_id (indexes id_map) and episode_id (indexes id_map_b).
        # Storage is canonical e → s.
        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            s_idx = item.get("state_id")
            e_idx = item.get("episode_id")
            s_id = _resolve_id(id_map, s_idx)
            e_id = _resolve_id(id_map_b, e_idx)
            rel = _validate_relation(item.get("relation"))
            if s_id is None or e_id is None or rel is None:
                continue
            self._store_evidence_edge(e_id, s_id, rel)
        self._5d_call_count += 1

    def _apply_state_new_rel_result(self, result: Dict, id_map: List[str]) -> None:
        # ②b judgments span both new↔new and new↔prev. _add_state_judgment uses
        # `new_state_ids` membership to normalize SHIFT_TO direction (prev→new),
        # falling back to `created_at` when both endpoints are new.
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
        # ③b judgments span (new_episode, previous_episode) [optional, 1] +
        # (new_episode, chunk_state_i) [|chunk_state_ids|].
        # id_map layout: [0]=new_episode, [1]=prev_episode (if any), [2..]=chunk_states.
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
            rel = _validate_relation(item.get("relation"))
            if src is None or dst is None or src == dst or rel is None:
                continue

            # Determine the "other" endpoint.
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
            # else: out of pair space, skip
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
                # new(src) replaces old(dst): store as old(dst) → new(src)
                self._store_evidence_edge(dst_id, src_id, EVID_SHIFT_TO)
            elif dst_new and not src_new:
                # old(src) replaced by new(dst): store as old(src) → new(dst)
                self._store_evidence_edge(src_id, dst_id, EVID_SHIFT_TO)
            else:
                # Both new or both old: use created_at
                src_node = self._graph.get_node(src_id)
                dst_node = self._graph.get_node(dst_id)
                if src_node and dst_node and src_node.created_at > dst_node.created_at:
                    self._store_evidence_edge(dst_id, src_id, EVID_SHIFT_TO)
                else:
                    self._store_evidence_edge(src_id, dst_id, EVID_SHIFT_TO)
            return

        self._store_evidence_edge(src_id, dst_id, relation)

    def _store_evidence_edge(self, src_id: str, dst_id: str, relation: str) -> None:
        # IRRELEVANT edges are stored so that future ⑤b/⑤c/⑤d candidate selection
        # (gated by `has_direct_edge`) can skip pairs that have already been judged.
        # Sign propagation and node score counters already exclude IRRELEVANT.
        self._graph.add_evidence_edge(src_id, dst_id, relation)

    def _store_shift_to_by_created_at(self, a_id: str, b_id: str) -> None:
        """Store a SHIFT_TO edge canonically as older → newer by created_at.
        Tie-break: when a.created_at == b.created_at, a → b (preserves the
        original `<=` semantics)."""
        a = self._graph.get_node(a_id)
        b = self._graph.get_node(b_id)
        if a is None or b is None:
            return
        if a.created_at <= b.created_at:
            self._store_evidence_edge(a_id, b_id, EVID_SHIFT_TO)
        else:
            self._store_evidence_edge(b_id, a_id, EVID_SHIFT_TO)

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

    def _format_turns(self, turns: List[TurnRecord], anchor_conv_id: int, anchor_turn_id: int) -> str:
        if not turns:
            return "(none)"
        lines = []
        for turn in turns:
            elapsed = self._elapsed_str(turn.conv_id, turn.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{elapsed}] User: {turn.user_utterance}")
            lines.append(f"[{elapsed}] Assistant: {turn.gt_response}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Indexed format helpers (integer index → id_map)
    # ------------------------------------------------------------------

    def _format_node_list_indexed(
        self,
        node_ids: List[Optional[str]],
        anchor_conv_id: int,
        anchor_turn_id: int,
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
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{start_idx + i}] [{elapsed}]: {node.content}")
        return "\n".join(lines), ids

    def _format_prior_pairs(
        self,
        pairs: List[Tuple[str, str, int, int]],
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> str:
        """Render prior (user, assistant, conv_id, turn_id) pairs for the ②
        PRIOR CONTEXT block. Same surface format as _format_turns."""
        if not pairs:
            return "(none)"
        lines = []
        for user, assistant, conv_id, turn_id in pairs:
            elapsed = self._elapsed_str(conv_id, turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{elapsed}] User: {user}")
            lines.append(f"[{elapsed}] Assistant: {assistant}")
        return "\n".join(lines)

    def _elapsed_str(self, entry_conv_id: int, entry_turn_id: int, current_conv_id: int, current_turn_id: int) -> str:
        return format_elapsed_str(
            entry_conv_id, entry_turn_id,
            current_conv_id, current_turn_id,
            self._cfg.TIME_PER_CONV_ID_HOURS,
            self._cfg.TIME_PER_TURN_MINUTES,
        )

    def _embed_text(self, text: str) -> np.ndarray:
        return embed_text(self._embed_model, text)


# =============================================================================
# Helpers
# =============================================================================

def _index_descriptor(start: int, count: int, label_singular: str) -> str:
    """Format a natural-language descriptor for an index range.
    count=0 → "" ; count=1 → "Index {start} is a {label_singular}." ;
    count>=2 → "Indices {start}..{start+count-1} are {label_singular}s."
    """
    if count <= 0:
        return ""
    if count == 1:
        return f"Index {start} is a {label_singular}."
    return f"Indices {start}..{start + count - 1} are {label_singular}s."


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
    """Validate domain_label list. Drops items that match a keyword
    (case-insensitive) or contain whitespace. When fewer than min_count
    survive, appends generic 'general[_N]' fillers — keyword backfill is NOT
    used because graph_store.add_node would strip it again at storage time."""
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
                continue   # disjointness with keywords
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
    """Validate full evidence relation (SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT)."""
    if not isinstance(val, str):
        return None
    norm = val.strip().upper()
    if norm in {EVID_SUPPORT, EVID_CONTRADICT, EVID_SHIFT_TO, EVID_IRRELEVANT}:
        return norm
    return None


def _format_conv_gap(conv_id_a: int, conv_id_b: int, time_per_conv_id_hours: float) -> str:
    return format_conv_gap(conv_id_a, conv_id_b, time_per_conv_id_hours)


def _resolve_id(id_map: List[str], idx) -> Optional[str]:
    """Resolve an integer index from the LLM output to a node_id via id_map.
    Accepts int, or strings like "5", " 5 ", "[5]" as a defensive measure
    against guided_json escape paths. Returns None on out-of-range or
    unparseable input."""
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
