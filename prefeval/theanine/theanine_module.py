"""
Theanine Module — PrefEval port.

Trimmed from exp_implexconv_no_response/theanine/theanine_module.py:
- Removed `subset` parameter from answer_qa() and Generator calls.
- Added load_memory_snapshot() helper.

Otherwise unchanged: wraps MemoryGraph + TimelineRetriever + Generator,
same token tracking and per-call logging.
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
    """Unified Theanine experiment interface (single-chain variant)."""

    def __init__(self, llm_client, model_path: str = "", embedding_model=None):
        self.llm_client   = llm_client
        self.model_path   = model_path
        self.memory_graph = MemoryGraph(llm_client, cfg, embedding_model=embedding_model)
        self.timeline     = TimelineRetriever(llm_client, model_path)
        self.generator    = Generator(llm_client, model_path)

    def set_llm_logger(self, llm_logger: Optional[LLMCallLogger]) -> None:
        self.memory_graph.set_llm_logger(llm_logger)
        self.timeline.set_llm_logger(llm_logger)
        self.generator.set_llm_logger(llm_logger)

    # =========================================================================
    # Retrieval (Phase 2 QA)
    # =========================================================================

    def retrieve(self, query: str) -> Tuple[Dict, List[Dict]]:
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
        current_dialogue: str = "",
    ) -> Tuple[str, Dict, Dict, str]:
        use_timelines = timelines.get("use_timeline", [])

        refined_texts, refine_tokens = self.timeline.refine_all(
            use_timelines, current_dialogue, self.memory_graph, question,
        )

        answer, qa_tokens, prompt = self.generator.generate_qa_answer(
            question, refined_texts, current_dialogue,
        )

        return answer, qa_tokens, refine_tokens, prompt

    # =========================================================================
    # Token tracking helpers
    # =========================================================================

    def get_and_reset_memory_tokens(self) -> Dict:
        return self.memory_graph.get_and_reset_token_counts_by_type()

    def get_memory_stats(self) -> Dict:
        return self.memory_graph.get_memory_stats()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def clear(self):
        self.memory_graph.clear()
        self.timeline.get_and_reset_token_counts()

    def get_memory_count(self) -> int:
        return self.memory_graph.get_node_count()

    def save_memory_snapshot(self, directory: Path):
        self.memory_graph.save_snapshot(directory)

    def load_memory_snapshot(self, directory: Path):
        self.memory_graph.load_snapshot(directory)
