"""
Updater for GraphMem v5.
"""

import copy
import heapq
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from generator import _NODE_TYPE_DESC
from graph_store import (
    HeterogeneousGraph,
    Node,
    NODE_M,
    NODE_S,
    NODE_T,
    EVID_SUPPORT,
    EVID_CONTRADICT,
    EVID_SHIFT_TO,
    EVID_IRRELEVANT,
    SCOPE_BROAD,
    SCOPE_NARROW,
    IMPACT_HIGH,
    IMPACT_LOW,
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
    # Number of judgments the LLM is expected to produce for this call.
    # 0 means "judgments are not produced or are legitimately empty" — wrapper skips retry.
    expected_judgment_count: int = 0
    # Integer-index → node_id maps for relation-extraction calls.
    # id_map_b is only populated for ⑤d (separate index space for memories).
    id_map: List[str] = field(default_factory=list)
    id_map_b: List[str] = field(default_factory=list)
    # Canonical (src_node_id, dst_node_id) tuples that this call is expected
    # to judge. Populated at build time for relation-extraction calls and
    # used by the JUDGMENT_RETRY exhaustion fallback to create IRRELEVANT
    # edges so that future ⑤b/⑤c/⑤d candidate selection can skip them.
    # Direction follows each call's documented canonical:
    #   ②b: prev_state → new_state ; new_state(earlier) → new_state(later)
    #   ③b: new_memory → previous_memory ; new_memory → chunk_state
    #   ⑤a: state/memory → new_trait ; previous_trait → new_trait
    #   ⑤b: candidate → new_trait
    #   ⑤c: older_state → newer_state
    #   ⑤d: state → memory   (state-memory pair; m→s canonicalized at apply)
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
    memory_id: Optional[str] = None
    last_turn_id: int = 0


# =============================================================================
# Prompt components
# =============================================================================

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

# Change 2: keywords vs domain_label disjointness block, inserted into every
# node-emitting prompt (②, ③, ④). Apply side runs deduplicate_labels regardless.
_LABEL_DISCIPLINE_BLOCK = (
    "keywords:\n"
    "  Surface-level tokens from the source text, or close lexical variants.\n"
    "  Use concrete entities, named items, specific actions, constraints, or particular phrases.\n"
    "  Do not use broad topical categories here.\n"
    "\n"
    "domain_label:\n"
    "  Abstract topical or categorical labels at a higher level of abstraction.\n"
    "  Use broader subject areas or mid-level categories that could group related memories across different surface wording.\n"
    "\n"
    "Hard constraint:\n"
    "  Each label string MUST appear in at most one of keywords or domain_label.\n"
    "  A domain_label may overlap lexically with a keyword only when it expresses a clearly broader topical category, not a simple restatement.\n"
    "  Drop any domain_label that merely restates or narrowly rephrases a keyword."
)

SYS_STATE_EXTRACT = (
    "You are a persona state extraction assistant.\n"
    "Extract user states from the current user turn.\n"
    "Do not classify relationships in this call.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC
)

SYS_MEMORY_EXTRACT = (
    "You are an episodic memory extraction assistant.\n"
    "Summarize the recent conversation into one episodic memory.\n"
    "Do not classify relationships in this call.\n"
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

SYS_STATE_MEMORY_5D = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships for candidate state-memory pairs.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_REDUCED
)

SYS_STATE_NEW_REL = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships involving newly extracted states: pairs among the new states themselves, and pairs between each new state and each previously extracted state.\n"
    "Respond in strict JSON.\n\n"
    + _NODE_TYPE_DESC + "\n\n" + _EVID_DESC_FULL
)

SYS_MEMORY_NEW_REL = (
    "You are an evidence classification assistant.\n"
    "Classify direct relationships involving a newly extracted memory: the pair (new_memory, previous_memory) when a previous memory exists, and one pair (new_memory, chunk_state_i) for each chunk state.\n"
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

_STATE_MEMORY_JUDGMENT_OBJECT = {
    "type": "object",
    "properties": {
        "state_id": {"type": "integer"},
        "memory_id": {"type": "integer"},
        "relation": {"type": "string"},
    },
    "required": ["state_id", "memory_id", "relation"],
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
                    "current_decision_impact": {"type": "string"},
                },
                "required": [
                    "id", "content", "keywords", "domain_label",
                    "scope", "current_decision_impact",
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

MEMORY_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
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
    # ③ is extraction-only — no `judgments` field. All new-memory relation
    # judgments (memory↔chunk_states + memory↔previous_memory) are produced
    # by ③b.
    "required": ["memory"],
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

STATE_MEMORY_JUDGMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": _STATE_MEMORY_JUDGMENT_OBJECT,
        },
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


# =============================================================================
# Updater
# =============================================================================

class GraphUpdater:
    """Updater for GraphMem v5 internal calls ②③④⑤."""

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
        # for the PRIOR CONTEXT block (Change 7).
        self._context_cache = context_cache

        self._current_chunk: Optional[ChunkRecord] = None
        self._completed_chunks: deque = deque(maxlen=config.TRAIT_EXTRACTION_CHUNKS)
        self._recent_turns: deque = deque(maxlen=config.STATE_EXTRACTION_H)
        self._previous_state_ids: List[str] = []
        self._previous_memory_id: Optional[str] = None
        self._previous_trait_id: Optional[str] = None

        self._user_turn_count = 0
        self._chunk_count = 0

        self._call_type_tokens: Dict[str, Dict[str, int]] = {}

        self._state_extracted_count = 0
        self._memory_extracted_count = 0
        self._trait_extracted_count = 0
        self._5a_call_count = 0
        self._5b_call_count = 0
        self._5c_call_count = 0
        self._5d_call_count = 0
        self._state_new_rel_call_count = 0
        self._memory_new_rel_call_count = 0

        # Pending trait ID carried across ⑤a → ⑤b chain
        self._pending_trait_node_id: Optional[str] = None

        # Pending data carried across state/memory → prev_rel chain
        self._pending_state_new_rel_new_ids: List[str] = []
        # ③b pending data: holds new_memory_id, previous_memory_id (or None),
        # and chunk_state_ids of the just-finalized chunk.
        self._pending_memory_new_rel: Optional[Dict[str, Any]] = None

        # Global reservoirs: list of (-score, pair_tuple) for min-heap eviction
        # ss_reservoir: pairs of (state_id_a, state_id_b)
        # sm_reservoir: pairs of (state_id, memory_id)
        self._ss_reservoir: List[Tuple[float, Tuple[str, str]]] = []
        self._sm_reservoir: List[Tuple[float, Tuple[str, str]]] = []

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
        return self._build_memory_call(
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
        return self._build_memory_call(
            chunk=self._current_chunk,
            session_id=session_id,
            created_at=global_turn,
            anchor_conv_id=self._current_chunk.conv_id,
            anchor_turn_id=self._current_chunk.last_turn_id,
        )

    def apply_irrelevant_fallback(self, call: PendingLLMCall) -> None:
        """Store IRRELEVANT edges for every expected pair on this call.
        Called when JUDGMENT_RETRY is exhausted with empty judgments so that
        future ⑤b/⑤c/⑤d candidate selection (gated by has_direct_edge) skips
        the pair instead of re-judging it. Direction follows each call's
        canonical (already encoded in expected_pairs at build time)."""
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

        if call.call_type == "memory":
            old_prev_memory_id = self._previous_memory_id
            self._apply_memory_result(call, result)
            new_memory_id = self._previous_memory_id
            # Trigger ③b iff there is at least one judgeable pair:
            #   |chunk_state_ids| ≥ 1  OR  previous_memory exists.
            # The just-finalized chunk lives at _completed_chunks[-1].
            if (
                new_memory_id
                and new_memory_id != old_prev_memory_id
                and self._completed_chunks
            ):
                chunk_state_ids = list(self._completed_chunks[-1].state_ids)
                prev_id_for_call = (
                    old_prev_memory_id if old_prev_memory_id else None
                )
                if chunk_state_ids or prev_id_for_call:
                    return self._build_memory_new_rel_call(
                        session_id=call.session_id,
                        created_at=call.created_at,
                        anchor_conv_id=call.anchor_conv_id,
                        anchor_turn_id=call.anchor_turn_id,
                        new_memory_id=new_memory_id,
                        previous_memory_id=prev_id_for_call,
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

        if call.call_type == "memory_new_rel":
            self._apply_memory_new_rel_result(result, call.id_map)
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
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._ss_reservoir:
                return self._build_5c_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._sm_reservoir:
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
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._ss_reservoir:
                return self._build_5c_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            if self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION and self._sm_reservoir:
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
            if self._ss_reservoir:
                return self._build_5c_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            if self._sm_reservoir:
                return self._build_5d_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "state_state_5c":
            self._apply_5c_result(result, call.id_map)
            if self._sm_reservoir:
                return self._build_5d_call(
                    session_id=call.session_id,
                    created_at=call.created_at,
                    anchor_conv_id=call.anchor_conv_id,
                    anchor_turn_id=call.anchor_turn_id,
                )
            return None

        if call.call_type == "state_memory_5d":
            self._apply_5d_result(result, call.id_map, call.id_map_b)
            return None

        return None

    # ------------------------------------------------------------------
    # Sequential wrapper
    # ------------------------------------------------------------------

    def execute_call(self, call: PendingLLMCall) -> Dict:
        """Execute the LLM call, with empty-judgment retry (Change 1).

        If `call.expected_judgment_count > 0` and the parsed `judgments` is
        empty, retry up to JUDGMENT_RETRY times. Each judgment-retry attempt
        internally still allows up to JSON_RETRY JSON-parse retries via the
        underlying generator. After exhaustion, return whatever the last
        attempt produced (the apply layer treats an empty judgments array as
        all-IRRELEVANT — no edges, no hard failure). `IRRELEVANT` judgments
        are valid; a non-empty array is NOT retried.
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
            "num_memory_nodes_extracted": self._memory_extracted_count,
            "num_trait_nodes_extracted": self._trait_extracted_count,
            "num_5a_trait_evidence_calls": self._5a_call_count,
            "num_5b_trait_extra_rel_calls": self._5b_call_count,
            "num_5c_state_state_rel_calls": self._5c_call_count,
            "num_5d_state_memory_rel_calls": self._5d_call_count,
            "num_state_new_rel_calls": self._state_new_rel_call_count,
            "num_memory_new_rel_calls": self._memory_new_rel_call_count,
        }
        self._state_extracted_count = 0
        self._memory_extracted_count = 0
        self._trait_extracted_count = 0
        self._5a_call_count = 0
        self._5b_call_count = 0
        self._5c_call_count = 0
        self._5d_call_count = 0
        self._state_new_rel_call_count = 0
        self._memory_new_rel_call_count = 0
        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._current_chunk = None
        self._completed_chunks.clear()
        self._recent_turns.clear()
        self._previous_state_ids = []
        self._previous_memory_id = None
        self._previous_trait_id = None
        self._pending_trait_node_id = None
        self._user_turn_count = 0
        self._chunk_count = 0
        self._call_type_tokens = {}
        self._state_extracted_count = 0
        self._memory_extracted_count = 0
        self._trait_extracted_count = 0
        self._5a_call_count = 0
        self._5b_call_count = 0
        self._5c_call_count = 0
        self._5d_call_count = 0
        self._state_new_rel_call_count = 0
        self._memory_new_rel_call_count = 0
        self._pending_state_new_rel_new_ids = []
        self._pending_memory_new_rel = None
        self._ss_reservoir = []
        self._sm_reservoir = []

    def set_graph(self, graph: HeterogeneousGraph) -> None:
        self._graph = graph

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Reservoir maintenance
    # ------------------------------------------------------------------

    def _pair_score(self, node_a: Node, node_b: Node) -> float:
        sem = float(np.dot(node_a.embedding, node_b.embedding))
        overlap_a = set(node_a.keywords) | {l.lower() for l in node_a.domain_label}
        overlap_b = set(node_b.keywords) | {l.lower() for l in node_b.domain_label}
        lex = len(overlap_a & overlap_b)
        return self._cfg.W_PAIR_SEM * sem + self._cfg.W_PAIR_LEX * math.log1p(lex)

    def _reservoir_insert(
        self,
        reservoir: List,
        cap: int,
        score: float,
        pair: Tuple[str, str],
    ) -> None:
        """Insert (score, pair) into a bounded min-heap reservoir."""
        heapq.heappush(reservoir, (score, pair))
        if len(reservoir) > cap:
            heapq.heappop(reservoir)  # evict lowest-scoring

    def _update_ss_reservoir(self, new_state_ids: List[str]) -> None:
        """Update state-state reservoir when new states are added."""
        if not self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION:
            return
        all_states = self._graph.get_nodes_by_type(NODE_S)
        cap = self._cfg.STATE_STATE_EXTRA_REL_TOPK
        for new_id in new_state_ids:
            new_node = self._graph.get_node(new_id)
            if new_node is None:
                continue
            for existing in all_states:
                if existing.node_id == new_id:
                    continue
                if self._graph.has_direct_edge(new_id, existing.node_id):
                    continue
                score = self._pair_score(new_node, existing)
                self._reservoir_insert(self._ss_reservoir, cap, score, (new_id, existing.node_id))

    def _update_sm_reservoir(self, new_state_ids: List[str], new_memory_ids: List[str]) -> None:
        """Update state-memory reservoir when new states or memories are added."""
        if not self._cfg.ENABLE_EXTRA_RELATION_EXTRACTION:
            return
        cap = self._cfg.STATE_MEMORY_EXTRA_REL_TOPK
        all_states = self._graph.get_nodes_by_type(NODE_S)
        all_memories = self._graph.get_nodes_by_type(NODE_M)

        for new_s_id in new_state_ids:
            new_s = self._graph.get_node(new_s_id)
            if new_s is None:
                continue
            for m in all_memories:
                if self._graph.has_direct_edge(new_s_id, m.node_id):
                    continue
                score = self._pair_score(new_s, m)
                self._reservoir_insert(self._sm_reservoir, cap, score, (new_s_id, m.node_id))

        for new_m_id in new_memory_ids:
            new_m = self._graph.get_node(new_m_id)
            if new_m is None:
                continue
            for s in all_states:
                if self._graph.has_direct_edge(s.node_id, new_m_id):
                    continue
                score = self._pair_score(s, new_m)
                self._reservoir_insert(self._sm_reservoir, cap, score, (s.node_id, new_m_id))

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
        # gmem5 ②: per-turn extraction (Change 5), with definition + good/NOT
        # examples + skip rule (Change 6) and a CURRENT TURN / PRIOR CONTEXT
        # split (Change 7). The current turn is the most recent registered
        # TurnRecord; PRIOR CONTEXT comes from the context_cache (which already
        # includes the current turn — we filter it out by (conv_id, turn_id)).
        recent_turns = list(self._recent_turns)
        if not recent_turns:
            current_turn_block = "None"
            current_conv_id = anchor_conv_id
            current_turn_id = anchor_turn_id
        else:
            current_turn = recent_turns[-1]
            current_turn_block = self._format_turns([current_turn], anchor_conv_id, anchor_turn_id)
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
            "A state does not have to be permanently stable. If the current turn reveals\n"
            "a useful persona-relevant signal but there is not yet enough evidence to call\n"
            "it a long-term trait, extract it as a state. Later stages may generalize\n"
            "repeated or stable states into traits.\n"
            "\n"
            "Do not extract if the information is only:\n"
            "  - a greeting, thanks, or conversational behavior,\n"
            "  - a description of the current query rather than the user,\n"
            "  - assistant-side information,\n"
            "  - a transient emotion with no effect on an active task, decision, constraint, or safety issue.\n"
            "\n"
            "Task:\n"
            f"Extract up to {self._cfg.STATE_MAX_COUNT} persona state(s) revealed in the CURRENT TURN.\n"
            "If the current turn contains no new persona-relevant signal, return an empty states list.\n"
            "Do not invent or speculate.\n"
            "\n"
            "Use the assistant response only as read-only context for resolving references in the user's utterance.\n"
            "Do NOT extract a state if the condition is stated only by the assistant and not expressed or clearly implied by the user.\n"
            "\n"
            "Each state must:\n"
            "- begin with \"The user\",\n"
            "- be a single concise sentence,\n"
            "- avoid raw episodic narration unless it directly functions as a current condition or useful persona signal.\n"
            "\n"
            "State metadata:\n"
            "scope:\n"
            "  BROAD  : a currently valid persona signal, condition, value-driven stance, health-related limitation, lifestyle constraint, role, or preference that may affect decisions across unrelated topics.\n"
            "  NARROW : a currently valid preference, goal, condition, or constraint tied mainly to the current task, topic, or short-term situation.\n"
            "\n"
            "If uncertain between BROAD and NARROW, choose NARROW.\n"
            "\n"
            "current_decision_impact:\n"
            "  HIGH : the assistant must actively remember this right now.\n"
            "         Use HIGH only when BOTH are true:\n"
            "         (1) the user would reasonably expect this to be remembered without re-stating it,\n"
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
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific nouns or noun phrases\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} short topical labels\n"
            "\n"
            "[CURRENT TURN — extract state(s) from this turn only]\n"
            f"{current_turn_block}\n"
            "\n"
            "[PRIOR CONTEXT — for disambiguation only.\n"
            " These turns may be from earlier today or earlier sessions.\n"
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
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            # ② is extraction-only; the schema has no `judgments` field. All
            # new-state relation judgments live in ②b. Retry wrapper skips.
            expected_judgment_count=0,
        )

    def _build_memory_call(
        self,
        chunk: ChunkRecord,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> PendingLLMCall:
        # gmem5 ③: extraction-only. The chunk-states block has been removed —
        # ③ summarizes the conversation only; chunk states are passed to ③b
        # for relation judgment. Rules first, data block at bottom (option B).
        prompt = (
            "Create exactly one episodic memory node summarizing what happened in the recent conversation.\n"
            "Each memory must:\n"
            "- begin with \"The user\",\n"
            "- be 1–2 sentences,\n"
            "- summarize concrete events, discussed topics, actions, and developments,\n"
            "- describe the episode itself, not generalized persona traits.\n"
            "\n"
            "Do NOT include chunk-level state extractions or restate persona traits unless\n"
            "they are part of the episode itself. The state nodes are extracted by a\n"
            "separate pipeline; this call summarizes only the conversation.\n"
            "\n"
            "Memory metadata:\n"
            "scope:\n"
            "  BROAD  : the episode reveals or confirms a cross-topic user characteristic (e.g., a health event, a major life decision, a value-revealing exchange, or a standing constraint the user reaffirmed).\n"
            "  NARROW : the episode is self-contained within the current topic or task — its implications do not extend beyond the current conversation thread.\n"
            "\n"
            "If uncertain between BROAD and NARROW, choose NARROW.\n"
            "\n"
            f"{_LABEL_DISCIPLINE_BLOCK}\n"
            "\n"
            "Also provide:\n"
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific nouns or noun phrases\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} short topical labels\n"
            "\n"
            "Assign the memory the placeholder id: new_memory.\n"
            "\n"
            "[Recent Conversation]\n"
            f"{self._format_turns(chunk.turns, anchor_conv_id, anchor_turn_id)}"
        )
        return PendingLLMCall(
            call_type="memory",
            system_prompt=SYS_MEMORY_EXTRACT,
            user_prompt=prompt,
            guided_json=MEMORY_EXTRACT_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_MEMORY,
            log_dir="call_3_memory",
            created_at=created_at,
            anchor_conv_id=anchor_conv_id,
            anchor_turn_id=anchor_turn_id,
            session_id=session_id,
            # ③ is extraction-only; the schema has no `judgments` field. All
            # new-memory relation judgments live in ③b. Retry wrapper skips.
            expected_judgment_count=0,
        )

    def _build_trait_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> PendingLLMCall:
        # gmem5 ④: only `[Recent Two Conversations]` is exposed to the LLM.
        # Chunk states, chunk memories, and the previously extracted trait are
        # NOT in the prompt — the trait is inferred directly from the raw
        # conversation as a likely persistent pattern. Concrete examples and
        # state-vs-trait comparison pairs are also removed to avoid
        # canonical-bias mimicry.
        recent_chunks = list(self._completed_chunks)
        turns: List[TurnRecord] = []
        for chunk in recent_chunks:
            turns.extend(chunk.turns)

        conversations_block = self._format_turns(turns, anchor_conv_id, anchor_turn_id)

        prompt = (
            "Use the \"in general\" test: a trait should still be true if you asked the user about themselves \"in general\" with no specific time, place, or context attached.\n"
            "\n"
            "Extract 0 or 1 trait from the recent two conversations below.\n"
            "A trait should:\n"
            "- begin with \"The user\",\n"
            "- be 2–3 complete sentences,\n"
            "- be inferable from the recent conversations as a likely persistent pattern, not just a one-time event or momentary feeling.\n"
            "\n"
            "If the recent conversations do not reveal a clear new persistent pattern, output 0 traits.\n"
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
            f"- keywords: up to {self._cfg.MAX_KEYWORDS} specific nouns or noun phrases\n"
            f"- domain_label: {self._cfg.MIN_DOMAIN_LABELS} to {self._cfg.MAX_DOMAIN_LABELS} short topical labels\n"
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
        memory_ids: List[str] = []
        for chunk in recent_chunks:
            state_ids.extend(chunk.state_ids)
            if chunk.memory_id:
                memory_ids.append(chunk.memory_id)

        trait = self._graph.get_node(self._pending_trait_node_id) if self._pending_trait_node_id else None
        # id_map layout: [0]=new_trait, [1..]=states, [1+n_states..]=memories,
        # [1+n_states+n_mems]=prev_trait (if any).
        new_trait_block, trait_ids = self._format_trait_list_indexed(
            [trait.node_id] if trait else [], anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        recent_states_block, s_ids = self._format_state_list_indexed(
            state_ids, anchor_conv_id, anchor_turn_id, start_idx=len(trait_ids),
        )
        recent_memories_block, m_ids = self._format_memory_list_indexed(
            memory_ids, anchor_conv_id, anchor_turn_id, start_idx=len(trait_ids) + len(s_ids),
        )
        previous_trait_block, pt_ids = self._format_trait_list_indexed(
            [self._previous_trait_id] if self._previous_trait_id else [],
            anchor_conv_id, anchor_turn_id,
            start_idx=len(trait_ids) + len(s_ids) + len(m_ids),
        )
        id_map = trait_ids + s_ids + m_ids + pt_ids
        new_trait_idx = 0
        prev_trait_idx = len(trait_ids) + len(s_ids) + len(m_ids) if pt_ids else None
        expected_5a = len(s_ids) + len(m_ids) + (1 if pt_ids else 0)

        prompt = (
            "You will judge direct evidence relationships between a newly extracted trait and nearby existing nodes.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"Index {new_trait_idx} is the new trait. "
            f"Indices {len(trait_ids)}..{len(trait_ids)+len(s_ids)-1} are recent states. "
            f"Indices {len(trait_ids)+len(s_ids)}..{len(trait_ids)+len(s_ids)+len(m_ids)-1} are recent memories. "
            + (f"Index {prev_trait_idx} is the previous trait.\n" if prev_trait_idx is not None else "\n")
            + "\n"
            "Judge:\n"
            "- each (recent state, new_trait) pair, in the listed order;\n"
            "- each (recent memory, new_trait) pair, in the listed order;\n"
            "- the (previous_trait, new_trait) pair, when a previous trait exists.\n"
            "\n"
            "Direction rule:\n"
            f"- For (recent_state, new_trait) and (recent_memory, new_trait): source_id = state/memory index, target_id = {new_trait_idx}.\n"
            + (f"- For (previous_trait, new_trait) SHIFT_TO: source_id MUST be {prev_trait_idx} (previous trait) and target_id MUST be {new_trait_idx} (new trait).\n"
               if prev_trait_idx is not None else "")
            + "\n"
            "Relation rules:\n"
            "- For state↔trait and memory↔trait: use only SUPPORT | CONTRADICT | IRRELEVANT.\n"
            "- For previous_trait↔new_trait: use SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT. SHIFT_TO is allowed only for a true old_trait → new_trait replacement.\n"
            "\n"
            "SUPPORT calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- For (state, new_trait): use SUPPORT only when the state describes a current/recent condition that the new trait would naturally generalize from or predict. Sharing a domain or keyword is not enough.\n"
            "- For (memory, new_trait): use SUPPORT only when the memory captures concrete user behavior that the trait would predict; the memory must add evidential force beyond mere topic overlap.\n"
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
            "[Recent Memories]\n"
            f"{recent_memories_block}\n"
            "\n"
            "[Previous Trait]\n"
            f"{previous_trait_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {expected_5a} judgments — one for each listed pair, in the listed order.\n"
            "Use the integer indices shown above for source_id and target_id.\n"
            "Return strict JSON only."
        )
        # Canonical pairs for IRRELEVANT-fallback:
        # state→new_trait, memory→new_trait, prev_trait→new_trait.
        new_trait_id = trait_ids[0] if trait_ids else None
        expected_pairs: List[Tuple[str, str]] = []
        if new_trait_id is not None:
            for sid in s_ids:
                expected_pairs.append((sid, new_trait_id))
            for mid in m_ids:
                expected_pairs.append((mid, new_trait_id))
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

        # Collect recent 2-chunk state/memory IDs to exclude (already handled by ⑤a)
        recent_ids: Set[str] = set()
        for chunk in self._completed_chunks:
            recent_ids.update(chunk.state_ids)
            if chunk.memory_id:
                recent_ids.add(chunk.memory_id)

        k_state = self._cfg.TRAIT_EXTRA_REL_TOPK_STATE
        k_mem = self._cfg.TRAIT_EXTRA_REL_TOPK_MEMORY

        candidate_states = self._5b_candidates(trait, NODE_S, recent_ids, k_state)
        candidate_memories = self._5b_candidates(trait, NODE_M, recent_ids, k_mem)

        if not candidate_states and not candidate_memories:
            return None

        # id_map layout: [0]=trait, [1..n_s]=candidate states, [n_s+1..]=candidate memories.
        trait_block, trait_ids = self._format_trait_list_indexed(
            [trait.node_id], anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        cand_state_block, cs_ids = self._format_state_list_indexed(
            [n.node_id for n in candidate_states], anchor_conv_id, anchor_turn_id,
            start_idx=len(trait_ids),
        )
        cand_mem_block, cm_ids = self._format_memory_list_indexed(
            [n.node_id for n in candidate_memories], anchor_conv_id, anchor_turn_id,
            start_idx=len(trait_ids) + len(cs_ids),
        )
        id_map = trait_ids + cs_ids + cm_ids
        trait_idx = 0
        num_candidates = len(cs_ids) + len(cm_ids)

        prompt = (
            "You will judge direct evidence relationships between a newly extracted trait and additional candidate nodes that are currently unconnected to it in the graph.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            f"Index {trait_idx} is the new trait. "
            f"Indices {len(trait_ids)}..{len(trait_ids)+len(cs_ids)-1} are candidate states. "
            f"Indices {len(trait_ids)+len(cs_ids)}..{len(trait_ids)+len(cs_ids)+len(cm_ids)-1} are candidate memories.\n"
            "\n"
            "Judge each listed candidate directly against the new trait.\n"
            "\n"
            "Direction rule:\n"
            f"- source_id = candidate index, target_id = {trait_idx} (new trait) for every judgment.\n"
            "\n"
            "Relation rules:\n"
            "- state ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT\n"
            "- memory ↔ trait: SUPPORT | CONTRADICT | IRRELEVANT\n"
            "- SHIFT_TO is NOT allowed in this call. Cross-type pairs (state↔trait, memory↔trait) cannot be temporal replacements.\n"
            "\n"
            "The listed candidates are surfaced by global similarity mining over unconnected state↔trait and memory↔trait pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each candidate's content directly.\n"
            "\n"
            "SUPPORT calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- For (state, new_trait): use SUPPORT only when the state describes a current/recent condition that would be expected GIVEN the new trait, or that the trait would naturally generalize from. Sharing a domain or keyword is not enough.\n"
            "- For (memory, new_trait): use SUPPORT only when the memory captures concrete user behavior that the trait would predict; the memory must add evidential force beyond mere topic overlap.\n"
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
            "[Additional Candidate Memories]\n"
            f"{cand_mem_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {num_candidates} judgments — one for each listed candidate, in the listed order.\n"
            "Use the integer indices shown above for source_id and target_id.\n"
            "Return strict JSON only."
        )
        # Canonical pairs: candidate→new_trait for every listed candidate.
        new_trait_id = trait_ids[0]
        expected_pairs: List[Tuple[str, str]] = [
            (cid, new_trait_id) for cid in (cs_ids + cm_ids)
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
        """Retrieve top-k extra candidates for ⑤b via semantic+lexical union."""
        all_nodes = [
            n for n in self._graph.get_nodes_by_type(node_type)
            if n.node_id not in exclude_ids
            and not self._graph.has_direct_edge(trait.node_id, n.node_id)
        ]
        if not all_nodes:
            return []

        # Semantic top-k
        sem_scored = sorted(
            all_nodes,
            key=lambda n: float(np.dot(trait.embedding, n.embedding)),
            reverse=True,
        )[:top_k]

        # Lexical top-k
        trait_lex = set(trait.keywords) | {l.lower() for l in trait.domain_label}
        lex_scored = sorted(
            all_nodes,
            key=lambda n: len(trait_lex & (set(n.keywords) | {l.lower() for l in n.domain_label})),
            reverse=True,
        )[:top_k]

        # Union + dedup + rerank by pair_score + cap
        seen: Set[str] = set()
        union: List[Node] = []
        for n in sem_scored + lex_scored:
            if n.node_id not in seen:
                seen.add(n.node_id)
                union.append(n)

        union.sort(key=lambda n: self._pair_score(trait, n), reverse=True)
        return union[:top_k]

    def _build_5c_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> Optional[PendingLLMCall]:
        if not self._ss_reservoir:
            return None

        # Consume all current reservoir pairs; apply stale-pair check at consumption time.
        raw_pairs = [pair for _, pair in self._ss_reservoir]
        self._ss_reservoir.clear()

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
        for pair_num, (id_a, id_b) in enumerate(raw_pairs):
            # Stale check: skip if an edge has since been created.
            if self._graph.has_direct_edge(id_a, id_b):
                continue
            node_a = self._graph.get_node(id_a)
            node_b = self._graph.get_node(id_b)
            if node_a is None or node_b is None:
                continue
            if node_a.created_at <= node_b.created_at:
                old_node, new_node = node_a, node_b
                old_id, new_id = id_a, id_b
            else:
                old_node, new_node = node_b, node_a
                old_id, new_id = id_b, id_a
            old_idx = _5c_get_idx(old_id)
            new_idx = _5c_get_idx(new_id)
            ss_expected_pairs.append((old_id, new_id))
            old_elapsed = self._elapsed_str(old_node.conv_id, old_node.turn_id, anchor_conv_id, anchor_turn_id)
            new_elapsed = self._elapsed_str(new_node.conv_id, new_node.turn_id, anchor_conv_id, anchor_turn_id)
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
        if not self._sm_reservoir:
            return None

        raw_pairs = [pair for _, pair in self._sm_reservoir]
        self._sm_reservoir.clear()

        # ⑤d uses separate index spaces: id_map for states, id_map_b for memories.
        id_map: List[str] = []   # state indices
        id_map_b: List[str] = []  # memory indices
        s_to_idx: Dict[str, int] = {}
        m_to_idx: Dict[str, int] = {}

        def _get_s_idx(nid: str) -> int:
            if nid not in s_to_idx:
                s_to_idx[nid] = len(id_map)
                id_map.append(nid)
            return s_to_idx[nid]

        def _get_m_idx(nid: str) -> int:
            if nid not in m_to_idx:
                m_to_idx[nid] = len(id_map_b)
                id_map_b.append(nid)
            return m_to_idx[nid]

        pair_lines = []
        sm_expected_pairs: List[Tuple[str, str]] = []
        for pair_num, (s_id, m_id) in enumerate(raw_pairs):
            # Stale check: skip if an edge has since been created.
            if self._graph.has_direct_edge(s_id, m_id):
                continue
            s_node = self._graph.get_node(s_id)
            m_node = self._graph.get_node(m_id)
            if s_node is None or m_node is None:
                continue
            s_idx = _get_s_idx(s_id)
            m_idx = _get_m_idx(m_id)
            # Storage canonical for ⑤d is m → s.
            sm_expected_pairs.append((m_id, s_id))
            elapsed_s = self._elapsed_str(s_node.conv_id, s_node.turn_id, anchor_conv_id, anchor_turn_id)
            elapsed_m = self._elapsed_str(m_node.conv_id, m_node.turn_id, anchor_conv_id, anchor_turn_id)
            same_conv = (s_node.conv_id == m_node.conv_id)
            if same_conv:
                relation_context = "same conversation."
            else:
                gap_str = _format_conv_gap(
                    s_node.conv_id, m_node.conv_id,
                    self._cfg.TIME_PER_CONV_ID_HOURS,
                )
                relation_context = f"different conversations, {gap_str} apart."
            pair_lines.append(
                f"Pair {pair_num}\n"
                f"- state:  [{s_idx}] [{elapsed_s}]: {s_node.content}\n"
                f"- memory: [{m_idx}] [{elapsed_m}]: {m_node.content}\n"
                f"  Relation context: {relation_context}"
            )

        if not pair_lines:
            return None

        pair_block = "\n\n".join(pair_lines)
        num_pairs = len(pair_lines)
        prompt = (
            "You will judge direct evidence relationships for candidate state-memory pairs.\n"
            "Each pair is currently unconnected in the graph.\n"
            "\n"
            "State nodes use one integer index space (state_id); memory nodes use a separate integer index space (memory_id).\n"
            "\n"
            "The listed pairs are surfaced by global similarity mining over unconnected state-memory pairs in the graph. Their semantic similarity does NOT, by itself, imply any relation. Judge each pair's content directly.\n"
            "\n"
            "Direction rule:\n"
            "- This call uses keyed identifiers `state_id` and `memory_id` (not `source_id`/`target_id`). Output one judgment per listed pair, naming each pair by its state and memory integer indices.\n"
            "\n"
            "Relation rules:\n"
            "- SUPPORT: the memory provides concrete episodic evidence that independently grounds or confirms the state.\n"
            "- CONTRADICT: the memory provides concrete episodic evidence that conflicts with the state.\n"
            "- IRRELEVANT: merely topically related without evidential force, or unrelated.\n"
            "\n"
            "SUPPORT calibration:\n"
            "- Topical similarity alone is NOT SUPPORT.\n"
            "- Same-conversation co-extraction or temporal adjacency alone is NOT SUPPORT.\n"
            "- Use SUPPORT only when the memory provides concrete episodic evidence that independently grounds or confirms the state. The memory must add an episodic fact that would still ground the state if read in isolation.\n"
            "- When uncertain between SUPPORT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "CONTRADICT calibration:\n"
            "- Use CONTRADICT only when the memory contains an episodic fact that directly invalidates or conflicts with the state's claim.\n"
            "- Do not use CONTRADICT for unrelated topics or for facts that can naturally coexist with the state.\n"
            "- When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            "Relation context interpretation:\n"
            "Each pair carries a \"Relation context\" line indicating whether the state and memory come from the same conversation.\n"
            "- \"same conversation\" pairs deserve extra scrutiny: same-conversation co-extraction alone is NOT SUPPORT. Look for an episodic fact in the memory that would still ground the state outside that conversation.\n"
            "- \"different conversations\" pairs come from temporally distinct episodes; SUPPORT here is most justified when the memory's events directly evidence the state.\n"
            "\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order.\n"
            "Do not skip or duplicate pairs.\n"
            "\n"
            "[Candidate State-Memory Pairs]\n"
            f"{pair_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {num_pairs} judgments — one for each listed pair, in the listed order.\n"
            "Use the integer indices shown above for state_id and memory_id.\n"
            "Return strict JSON only."
        )
        return PendingLLMCall(
            call_type="state_memory_5d",
            system_prompt=SYS_STATE_MEMORY_5D,
            user_prompt=prompt,
            guided_json=STATE_MEMORY_JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_STATE_MEMORY_5D,
            log_dir="call_5d_state_memory_rel",
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
        # State blocks expose `scope` (but not impact) per gmem5_prompt.md §3.2.
        self._pending_state_new_rel_new_ids = list(new_state_ids)
        prev_block, prev_ids = self._format_state_list_with_scope_indexed(
            previous_state_ids, anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        new_block, new_ids = self._format_state_list_with_scope_indexed(
            new_state_ids, anchor_conv_id, anchor_turn_id, start_idx=len(prev_ids),
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

    def _build_memory_new_rel_call(
        self,
        session_id: int,
        created_at: int,
        anchor_conv_id: int,
        anchor_turn_id: int,
        new_memory_id: str,
        previous_memory_id: Optional[str],
        chunk_state_ids: List[str],
    ) -> PendingLLMCall:
        # ③b: judges (new_memory, previous_memory) when prev exists, plus
        # one (new_memory, chunk_state_i) per chunk state in listed order.
        # Direction is fixed to source = new_memory; trigger lives in apply_call_result.
        # id_map layout: [0]=new_memory, [1]=prev_memory (if any), [2..]=chunk_states.
        self._pending_memory_new_rel = {
            "new_memory_id": new_memory_id,
            "previous_memory_id": previous_memory_id,
            "chunk_state_ids": list(chunk_state_ids),
        }

        # Build id_map: new_memory first, then prev_memory (if any), then chunk_states.
        new_block, new_ids = self._format_memory_list_indexed(
            [new_memory_id], anchor_conv_id, anchor_turn_id, start_idx=0,
        )
        prev_block, prev_ids = self._format_memory_list_indexed(
            [previous_memory_id] if previous_memory_id else [],
            anchor_conv_id, anchor_turn_id, start_idx=len(new_ids),
        )
        chunk_states_block, state_ids = self._format_state_list_indexed(
            chunk_state_ids, anchor_conv_id, anchor_turn_id,
            start_idx=len(new_ids) + len(prev_ids),
        )
        id_map = new_ids + prev_ids + state_ids

        n_prev = len(prev_ids)
        n_states = len(state_ids)
        expected = n_prev + n_states

        # Compute the target index range for the prev_memory (if any) and states.
        prev_idx = len(new_ids) if n_prev else None  # index of prev_memory in id_map
        state_start = len(new_ids) + n_prev

        prompt = (
            "You will judge direct evidence relationships involving the newly extracted memory.\n"
            "\n"
            "Each node is identified by an integer index shown in brackets: [index].\n"
            "Index 0 is the new memory. "
            + (f"Index {prev_idx} is the previous memory. " if prev_idx is not None else "")
            + (f"Indices {state_start}..{state_start + n_states - 1} are chunk states.\n" if n_states else "\n")
            + "\n"
            "Judge:\n"
            "- the (new_memory, previous_memory) pair, when a previous memory exists; and\n"
            "- one (new_memory, chunk_state_i) pair for each listed chunk state, in the listed order.\n"
            "\n"
            "Direction rules (FIXED):\n"
            "- source_id MUST be 0 (new memory index) for every judgment.\n"
            "- target_id is the previous_memory index or chunk_state index.\n"
            "\n"
            "Relation rules:\n"
            "\n"
            "For (new_memory, previous_memory):\n"
            "  Use SUPPORT | CONTRADICT | IRRELEVANT.\n"
            "  Temporal adjacency alone is NOT SUPPORT.\n"
            "  Use SUPPORT only when the new memory continues, confirms, or concretely reinforces the previous memory.\n"
            "\n"
            "For (new_memory, chunk_state_i):\n"
            "  Use SUPPORT | CONTRADICT | IRRELEVANT.\n"
            "  Use SUPPORT only when the memory provides concrete episodic evidence that independently grounds or confirms the state.\n"
            "  Same-conversation co-extraction alone is NOT sufficient evidential force.\n"
            "  The memory must add an episodic fact that would still ground the state if read in isolation.\n"
            "\n"
            "When uncertain between CONTRADICT and IRRELEVANT, choose IRRELEVANT.\n"
            "\n"
            f"Output exactly {expected} judgments in this order:\n"
            "  1) the (new_memory, previous_memory) judgment, if a previous memory exists;\n"
            "  2) one (new_memory, chunk_state_i) judgment for each listed chunk state, in the listed order.\n"
            "Do not invent states, memories, or judgments.\n"
            "\n"
            "[New Memory]\n"
            f"{new_block}\n"
            "\n"
            "[Previous Memory]\n"
            f"{prev_block}\n"
            "\n"
            "[Chunk States — listed in extraction order]\n"
            f"{chunk_states_block}\n"
            "\n"
            "Final instruction:\n"
            f"Output exactly {expected} judgments in the order specified above (previous_memory first if present, then chunk states in listed order).\n"
            "Use the integer indices shown above for source_id and target_id.\n"
            "Return strict JSON only."
        )
        # Canonical pairs for IRRELEVANT-fallback: new_memory → previous_memory, then new_memory → each chunk state.
        expected_pairs: List[Tuple[str, str]] = []
        if prev_ids:
            expected_pairs.append((new_ids[0], prev_ids[0]))
        for s in state_ids:
            expected_pairs.append((new_ids[0], s))
        return PendingLLMCall(
            call_type="memory_new_rel",
            system_prompt=SYS_MEMORY_NEW_REL,
            user_prompt=prompt,
            guided_json=JUDGMENTS_SCHEMA,
            max_tokens=self._cfg.MAX_TOKENS_MEMORY_NEW_REL,
            log_dir="call_3b_memory_new_rel",
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
                current_decision_impact=_validate_impact(item.get("current_decision_impact")),
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

    def _apply_memory_result(self, call: PendingLLMCall, result: Dict) -> None:
        memory_obj = result.get("memory", {}) if isinstance(result, dict) else {}
        content = _validate_content(memory_obj)
        if not content:
            self._finalize_chunk_record()
            return

        chunk = self._current_chunk
        if chunk is None:
            return

        keywords = _validate_keywords(memory_obj.get("keywords"), self._cfg.MAX_KEYWORDS)
        domain_label = _validate_domain_labels(
            memory_obj.get("domain_label"),
            keywords,
            self._cfg.MIN_DOMAIN_LABELS,
            self._cfg.MAX_DOMAIN_LABELS,
        )
        memory_node = Node(
            node_id=HeterogeneousGraph.new_node_id(),
            node_type=NODE_M,
            content=content,
            keywords=keywords,
            domain_label=domain_label,
            embedding=self._embed_text(content),
            created_at=call.created_at,
            session_id=call.session_id,
            conv_id=chunk.conv_id,
            turn_id=chunk.last_turn_id,
            scope=_validate_scope(memory_obj.get("scope")),
            retrieval_count=1,
        )
        self._graph.add_node(memory_node)
        chunk.memory_id = memory_node.node_id
        for context_id in chunk.context_ids:
            self._graph.add_source_edge(context_id, memory_node.node_id)

        # ③ is extraction-only. All new-memory relation judgments
        # (memory↔chunk_states + memory↔previous_memory) are produced by ③b.
        self._previous_memory_id = memory_node.node_id
        self._memory_extracted_count += 1
        self._finalize_chunk_record()

        # Update sm reservoir
        self._update_sm_reservoir([], [memory_node.node_id])

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
            if chunk.memory_id:
                self._graph.add_source_edge(chunk.memory_id, trait_node.node_id)

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
                # trait ↔ trait: SHIFT_TO allowed
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
                # state → trait or memory → trait: reduced (no SHIFT_TO)
                rel = _validate_relation_reduced(relation)
                if rel is None:
                    continue
                # Store canonically: s → t or m → t
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
                # Enforce old → new direction by created_at.
                if src_node.created_at <= dst_node.created_at:
                    self._store_evidence_edge(src, dst, EVID_SHIFT_TO)
                else:
                    self._store_evidence_edge(dst, src, EVID_SHIFT_TO)
            else:
                self._store_evidence_edge(src, dst, rel)
        self._5c_call_count += 1

    def _apply_5d_result(self, result: Dict, id_map: List[str], id_map_b: List[str]) -> None:
        # ⑤d uses state_id (indexes id_map) and memory_id (indexes id_map_b).
        # Storage is canonical m → s.
        judgments = result.get("judgments", []) if isinstance(result, dict) else []
        for item in judgments:
            s_idx = item.get("state_id")
            m_idx = item.get("memory_id")
            s_id = _resolve_id(id_map, s_idx)
            m_id = _resolve_id(id_map_b, m_idx)
            rel = _validate_relation_reduced(item.get("relation"))
            if s_id is None or m_id is None or rel is None:
                continue
            self._store_evidence_edge(m_id, s_id, rel)
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

    def _apply_memory_new_rel_result(self, result: Dict, id_map: List[str]) -> None:
        # ③b judgments span (new_memory, previous_memory) [optional, 1] +
        # (new_memory, chunk_state_i) [|chunk_state_ids|].
        # id_map layout: [0]=new_memory, [1]=prev_memory (if any), [2..]=chunk_states.
        pending = self._pending_memory_new_rel
        self._pending_memory_new_rel = None
        if pending is None:
            return
        new_id = pending["new_memory_id"]
        prev_id = pending.get("previous_memory_id")
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
        self._memory_new_rel_call_count += 1

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

    def _format_state_list(self, node_ids: List[Optional[str]], anchor_conv_id: int, anchor_turn_id: int) -> str:
        ids = [nid for nid in node_ids if nid]
        if not ids:
            return "(none)"
        lines = []
        for nid in ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{nid}] [{elapsed}]: {node.content}")
        return "\n".join(lines) if lines else "(none)"

    def _format_state_list_with_scope(
        self,
        node_ids: List[Optional[str]],
        anchor_conv_id: int,
        anchor_turn_id: int,
    ) -> str:
        ids = [nid for nid in node_ids if nid]
        if not ids:
            return "(none)"
        lines = []
        for nid in ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{nid}] [{elapsed}] (scope={node.scope}): {node.content}")
        return "\n".join(lines) if lines else "(none)"

    def _format_memory_list(self, node_ids: List[Optional[str]], anchor_conv_id: int, anchor_turn_id: int) -> str:
        ids = [nid for nid in node_ids if nid]
        if not ids:
            return "(none)"
        lines = []
        for nid in ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{nid}] [{elapsed}]: {node.content}")
        return "\n".join(lines) if lines else "(none)"

    def _format_trait_list(self, node_ids: List[Optional[str]], anchor_conv_id: int, anchor_turn_id: int) -> str:
        ids = [nid for nid in node_ids if nid]
        if not ids:
            return "(none)"
        lines = []
        for nid in ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{nid}] [{elapsed}]: {node.content}")
        return "\n".join(lines) if lines else "(none)"

    # ------------------------------------------------------------------
    # Indexed format helpers (integer index → id_map)
    # ------------------------------------------------------------------

    def _format_state_list_indexed(
        self,
        node_ids: List[Optional[str]],
        anchor_conv_id: int,
        anchor_turn_id: int,
        start_idx: int = 0,
    ) -> Tuple[str, List[str]]:
        """Returns (formatted_block, id_map) using integer indices starting at start_idx."""
        ids = [nid for nid in node_ids if nid and self._graph.get_node(nid)]
        if not ids:
            return "(none)", []
        lines = []
        for i, nid in enumerate(ids):
            node = self._graph.get_node(nid)
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{start_idx + i}] [{elapsed}]: {node.content}")
        return "\n".join(lines), ids

    def _format_state_list_with_scope_indexed(
        self,
        node_ids: List[Optional[str]],
        anchor_conv_id: int,
        anchor_turn_id: int,
        start_idx: int = 0,
    ) -> Tuple[str, List[str]]:
        ids = [nid for nid in node_ids if nid and self._graph.get_node(nid)]
        if not ids:
            return "(none)", []
        lines = []
        for i, nid in enumerate(ids):
            node = self._graph.get_node(nid)
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{start_idx + i}] [{elapsed}] (scope={node.scope}): {node.content}")
        return "\n".join(lines), ids

    def _format_memory_list_indexed(
        self,
        node_ids: List[Optional[str]],
        anchor_conv_id: int,
        anchor_turn_id: int,
        start_idx: int = 0,
    ) -> Tuple[str, List[str]]:
        ids = [nid for nid in node_ids if nid and self._graph.get_node(nid)]
        if not ids:
            return "(none)", []
        lines = []
        for i, nid in enumerate(ids):
            node = self._graph.get_node(nid)
            elapsed = self._elapsed_str(node.conv_id, node.turn_id, anchor_conv_id, anchor_turn_id)
            lines.append(f"[{start_idx + i}] [{elapsed}]: {node.content}")
        return "\n".join(lines), ids

    def _format_trait_list_indexed(
        self,
        node_ids: List[Optional[str]],
        anchor_conv_id: int,
        anchor_turn_id: int,
        start_idx: int = 0,
    ) -> Tuple[str, List[str]]:
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
        """Render prior (user, assistant, conv_id, turn_id) pairs for ② PRIOR CONTEXT
        block (Change 7). Identical surface format to _format_turns."""
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
        seen.add(norm)
        result.append(norm)
        if len(result) >= max_kw:
            break
    return result


def _validate_domain_labels(val, keywords: List[str], min_count: int, max_count: int) -> List[str]:
    """Validate domain_label list. Enforces keyword/domain_label disjointness
    (Change 2): tokens that match a keyword (case-insensitive) are dropped.
    When fewer than min_count survive, generic 'general[_N]' fillers are
    appended — keyword backfill is NOT used because that would re-introduce
    overlap that graph_store.add_node would then strip again.
    """
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


def _validate_impact(val) -> str:
    if isinstance(val, str) and val.strip().upper() in {IMPACT_HIGH, IMPACT_LOW}:
        return val.strip().upper()
    return IMPACT_LOW


def _validate_relation(val) -> Optional[str]:
    """Validate full evidence relation (SUPPORT | CONTRADICT | SHIFT_TO | IRRELEVANT)."""
    if not isinstance(val, str):
        return None
    norm = val.strip().upper()
    if norm in {EVID_SUPPORT, EVID_CONTRADICT, EVID_SHIFT_TO, EVID_IRRELEVANT}:
        return norm
    return None


def _validate_relation_reduced(val) -> Optional[str]:
    """Validate reduced evidence relation (SUPPORT | CONTRADICT | IRRELEVANT)."""
    if not isinstance(val, str):
        return None
    norm = val.strip().upper()
    if norm in {EVID_SUPPORT, EVID_CONTRADICT, EVID_IRRELEVANT}:
        return norm
    return None


def _resolve_id(item: Dict, key: str, placeholder_map: Dict[str, str]) -> Optional[str]:
    value = item.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return placeholder_map.get(value, value) if value else None


def _format_conv_gap(conv_id_a: int, conv_id_b: int, time_per_conv_id_hours: float) -> str:
    """Thin alias around graph_store.format_conv_gap (Change 3.3 ⑤d)."""
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
