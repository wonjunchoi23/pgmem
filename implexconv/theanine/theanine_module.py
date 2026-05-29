"""
Theanine Module — ImplexConv

Unified wrapper integrating MemoryGraph, TimelineRetriever, and Generator
for the Theanine experiment interface used by run_experiment.py.

Corresponds to the original Theanine class (src/theanine.py) split into
three dedicated components:
  MemoryGraph       → memory node construction, embedding, relation linking
  TimelineRetriever → path building and LLM-based timeline refinement
  Generator         → QA answering (Phase 2 only)

(Response generation removed — Phase 1 stores memory only, no LLM response.)

Phase 1 interface:
  finalize_conv(conv_id, session_id, dialogue)     → mem_token_info          [at conv boundary]

Phase 2 interface (per QA, memory frozen):
  retrieve(question)                               → (timelines, log_items)
  answer_qa(question, timelines, subset)           → (answer, qa_tokens, refine_tokens, prompt)

Lifecycle:
  clear()
  save_memory_snapshot(directory)
  get_memory_count()
  get_memory_stats()

Token categories (for run_experiment.py aggregation):
  call_3_summarization — finalize_conv summarization
  call_4_relation      — finalize_conv relation extraction
  call_2_refinement    — refine_all (timeline refinement) from answer_qa
  call_5_qa            — answer_qa LLM call
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config as cfg
from memory_graph import MemoryGraph
from timeline import TimelineRetriever
from generator import Generator

logger = logging.getLogger(__name__)


class LLMCallLogger:
    """
    Logs all LLM calls (input + output) to call-type-specific folders as JSONL files.

    Folder structure:
        {base_dir}/call_1_response/
        {base_dir}/call_2_refinement/
        {base_dir}/call_3_summarization/
        {base_dir}/call_4_relation/
        {base_dir}/call_5_qa/

    Each folder contains a single 'calls.jsonl' file with one JSON object per line.
    """

    CALL_DIRS = [
        "call_2_refinement",
        "call_3_summarization",
        "call_4_relation",
        "call_5_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        """
        Append one log entry to {base_dir}/{call_type}/calls.jsonl.

        Args:
            call_type:     One of the CALL_DIRS strings.
            system_prompt: System prompt string (empty string if not used).
            user_prompt:   Full prompt string sent to the LLM.
            output:        Raw LLM response (dict or str). _usage key is excluded.
        """
        entry = {
            "timestamp":     datetime.datetime.now().isoformat(),
            "call_type":     call_type,
            "system_prompt": system_prompt,
            "user_prompt":   user_prompt,
            "output":        output if not isinstance(output, dict) else {
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class TheanineModule:
    """
    Unified Theanine experiment interface.

    Holds one MemoryGraph (persists and grows across conv_ids within a session,
    cleared between sessions) and stateless TimelineRetriever + Generator.

    Token tracking:
    - Memory tokens: accumulated in memory_graph, collected via get_and_reset_token_counts_by_type()
      (called by run_experiment.py after Phase 1).
    - Refine tokens: accumulated in timeline_retriever, collected via get_and_reset_token_counts()
      (called by run_experiment.py after Phase 2).
    - QA tokens: returned directly from answer_qa().
    """

    def __init__(self, llm_client, model_path: str = "", embedding_model=None):
        self.llm_client  = llm_client
        self.model_path  = model_path
        self.memory_graph = MemoryGraph(llm_client, cfg, embedding_model=embedding_model)
        self.timeline     = TimelineRetriever(llm_client, model_path)
        self.generator    = Generator(llm_client, model_path)

    def set_llm_logger(self, llm_logger: Optional[LLMCallLogger]) -> None:
        """
        Inject a per-session LLMCallLogger into all sub-components.
        Call once per session before processing begins.
        Pass None to disable logging.
        """
        self.memory_graph.set_llm_logger(llm_logger)
        self.timeline.set_llm_logger(llm_logger)
        self.generator.set_llm_logger(llm_logger)

    # =========================================================================
    # Retrieval (used in Phase 2 QA)
    # =========================================================================

    def retrieve(self, query: str) -> Tuple[Dict, List[Dict]]:
        """
        Retrieve timeline paths for a given query.

        Calls TimelineRetriever.retrieve_timeline() which internally calls
        MemoryGraph.retrieve() (cosine similarity) then get_all_path() for each
        retrieved node.  Returns both the full timelines dict (passed unchanged
        to answer_qa) and log_items for retrieval logging.

        If the memory graph is empty, returns empty timelines gracefully.

        Args:
            query: Query string used for Phase 2 QA retrieval.

        Returns:
            timelines:  Dict from TimelineRetriever.retrieve_timeline():
                        {"retrieved_nodes": [(node_id, score), ...],
                         "use_timeline":    [path_tuple, ...],
                         "timeline":        [...]}
            log_items:  List of {"memory_id", "content_preview", "score",
                        "source_turn"} for retrieval log.
        """
        timelines = self.timeline.retrieve_timeline(query, self.memory_graph)

        nodes = self.memory_graph.nodes
        log_items = [
            {
                "memory_id":      node_id,
                "content_preview": nodes[node_id].summary[:120]
                                   if node_id in nodes else node_id,
                "score":          float(score),
                "source_turn": {
                    "session_id":           nodes[node_id].session_id           if node_id in nodes else -1,
                    "conv_id":              nodes[node_id].conv_id               if node_id in nodes else -1,
                    "turn_id_start":        nodes[node_id].turn_id_start        if node_id in nodes else -1,
                    "turn_id_end":          nodes[node_id].turn_id_end          if node_id in nodes else -1,
                    "global_turn_id_start": nodes[node_id].global_turn_id_start if node_id in nodes else -1,
                    "global_turn_id_end":   nodes[node_id].global_turn_id_end   if node_id in nodes else -1,
                },
            }
            for node_id, score in timelines["retrieved_nodes"]
        ]

        return timelines, log_items

    def finalize_conv(
        self,
        conv_id: int,
        session_id: int,
        full_conv_dialogue: str,
        turn_id_start: int = -1,
        turn_id_end: int = -1,
        global_turn_id_start: int = -1,
        global_turn_id_end: int = -1,
    ) -> Dict:
        """
        Summarize a completed conv and add nodes to the memory graph.
        Called at conv_id boundaries and after the last turn in a session.

        Delegates to MemoryGraph.finalize_conv():
          summarize → create nodes → embed → link to past nodes → register.

        Tokens from summarization + relation extraction are accumulated inside
        MemoryGraph and can be collected via get_and_reset_memory_tokens()
        (called by run_experiment.py after this method).

        Args:
            conv_id:              Completed conversation ID.
            session_id:           Current session ID.
            full_conv_dialogue:   Full formatted dialogue of the conv (GT turns).
                                  Format: "User: ...\nAssistant: ...\nUser: ..."
            turn_id_start:        turn_id of the first turn in this conv.
            turn_id_end:          turn_id of the last turn in this conv.
            global_turn_id_start: global_turn_id of the first turn in this conv.
            global_turn_id_end:   global_turn_id of the last turn in this conv.

        Returns:
            {"input": ..., "output": ..., "llm_calls": ...}  — token counts.
        """
        logger.info(
            f"finalize_conv: conv_id={conv_id}, session_id={session_id}, "
            f"nodes_before={self.memory_graph.get_node_count()}"
        )
        return self.memory_graph.finalize_conv(
            conv_id, session_id, full_conv_dialogue,
            turn_id_start, turn_id_end, global_turn_id_start, global_turn_id_end,
        )

    # =========================================================================
    # Phase 2 — QA Answering (memory frozen)
    # =========================================================================

    def answer_qa(
        self,
        question: str,
        timelines: Dict,
        subset: str,
        current_dialogue: str = "",
    ) -> Tuple[str, Dict, Dict, str]:
        """
        Refine timeline paths and generate QA answer.

        Uses the full timeline pipeline:
        TimelineRetriever.refine_all() → Generator.generate_qa_answer().
        This ensures QA answers are grounded in the same relation-aware,
        LLM-refined timeline context as responses.

        retrieve() should be called first to obtain timelines;
        the question string is used as the retrieval query.

        QA questions are independent of each other — each question is answered
        using the same frozen memory graph and the same final_dialogue.
        No new memory is written during QA.

        Args:
            question:         QA question string.
            timelines:        Dict from retrieve() (or retrieve_timeline()).
            subset:           "opposed" (free-form) or "supportive" (yes/no/unknown).
            current_dialogue: Full session dialogue only. The QA question is
                              passed separately to refinement and generation.

        Returns:
            (answer, qa_tokens, refine_tokens, prompt_snapshot)
            answer:        Generated answer string.
            qa_tokens:     {"input": ..., "output": ..., "model": ...}
                           → counted as qa_tokens in token_statistics.
            refine_tokens: {"input": ..., "output": ..., "llm_calls": ...}
                           → counted as call_2_refinement in token_statistics.
            prompt_snapshot: Full prompt sent to Generator.
        """
        use_timelines = timelines.get("use_timeline", [])

        # Timeline refinement
        refined_texts, refine_tokens = self.timeline.refine_all(
            use_timelines, current_dialogue, self.memory_graph, question
        )

        # QA generation (qa tokens)
        answer, qa_tokens, prompt = self.generator.generate_qa_answer(
            question, refined_texts, subset, current_dialogue
        )

        return answer, qa_tokens, refine_tokens, prompt

    # =========================================================================
    # Token tracking helpers
    # =========================================================================

    def get_and_reset_memory_tokens(self) -> Dict:
        """
        Collect and reset memory-layer token counts (summarization + relation
        extraction) per call type.  Called by run_experiment.py after Phase 1.

        Returns:
            {
                "call_3_summarization": {"input": int, "output": int, "llm_calls": int,
                                         "parse_fallback_count": int},
                "call_4_relation":      {"input": int, "output": int, "llm_calls": int},
            }
        """
        return self.memory_graph.get_and_reset_token_counts_by_type()

    def get_memory_stats(self) -> Dict:
        """
        Return current memory node count and estimated total content tokens.
        Called by run_experiment.py after Phase 1 ends, before Phase 2 begins.

        Returns:
            {"num_memories": int, "total_content_tokens": int}
        """
        return self.memory_graph.get_memory_stats()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def clear(self):
        """Reset all memory state. Called between sessions."""
        self.memory_graph.clear()
        # Reset timeline refinement counters too
        self.timeline.get_and_reset_token_counts()

    def get_memory_count(self) -> int:
        """Return total number of memory nodes in the graph."""
        return self.memory_graph.get_node_count()

    def save_memory_snapshot(self, directory: Path):
        """Save memory graph snapshot for this session."""
        self.memory_graph.save_snapshot(directory)
