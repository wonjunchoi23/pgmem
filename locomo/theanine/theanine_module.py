"""
Theanine Module — LoComo batch-enabled variant
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config as cfg
from generator import Generator
from memory_graph import MemoryGraph
from timeline import TimelineRetriever

logger = logging.getLogger(__name__)


class LLMCallLogger:
    CALL_DIRS = [
        "call_1_refinement",
        "call_2_summarization",
        "call_3_relation",
        "call_4_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "call_type": call_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output if not isinstance(output, dict) else {
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class TheanineModule:
    def __init__(self, llm_client, model_path: str = "", embedding_model=None):
        self.llm_client = llm_client
        self.model_path = model_path
        self.memory_graph = MemoryGraph(llm_client, cfg, embedding_model=embedding_model)
        self.timeline = TimelineRetriever(llm_client, model_path)
        self.generator = Generator(llm_client, model_path)

    def set_llm_logger(self, llm_logger: Optional[LLMCallLogger]) -> None:
        self.memory_graph.set_llm_logger(llm_logger)
        self.timeline.set_llm_logger(llm_logger)
        self.generator.set_llm_logger(llm_logger)

    def retrieve_for_response(self, query: str) -> Tuple[Dict, List[Dict]]:
        timelines = self.timeline.retrieve_timeline(query, self.memory_graph)
        nodes = self.memory_graph.nodes
        log_items = [
            {
                "memory_id": node_id,
                "content_preview": nodes[node_id].summary[:120] if node_id in nodes else node_id,
                "score": float(score),
                "source": {
                    "finalize_idx": nodes[node_id].finalize_idx if node_id in nodes else -1,
                    "sample_id": nodes[node_id].sample_id if node_id in nodes else "",
                    "session_ids": nodes[node_id].session_ids if node_id in nodes else [],
                    "dia_ids": nodes[node_id].dia_ids if node_id in nodes else [],
                    "date": nodes[node_id].date if node_id in nodes else "",
                },
            }
            for node_id, score in timelines["retrieved_nodes"]
        ]
        return timelines, log_items

    def finalize_conv(
        self,
        finalize_idx: int,
        sample_id: str,
        full_dialogue: str,
        dia_ids: List[str],
        session_ids: List[int],
        date: str,
    ) -> Dict:
        logger.info(
            f"finalize_conv: finalize_idx={finalize_idx}, sample_id={sample_id}, "
            f"nodes_before={self.memory_graph.get_node_count()}"
        )
        return self.memory_graph.finalize_conv(
            finalize_idx, sample_id, full_dialogue, dia_ids, session_ids, date
        )

    def answer_qa(
        self,
        question: str,
        timelines: Dict,
        category: int,
        current_dialogue: str = "",
        adversarial_answer: str = "",
        choice_order_seed: Optional[str] = None,
    ) -> Tuple[str, Dict, Dict, str]:
        use_timelines = timelines.get("use_timeline", [])
        refined_texts, refine_tokens = self.timeline.refine_all(
            use_timelines, current_dialogue, self.memory_graph, question=question
        )
        answer, qa_tokens, prompt = self.generator.generate_qa_answer(
            question,
            refined_texts,
            category,
            current_dialogue=current_dialogue,
            adversarial_answer=adversarial_answer,
            choice_order_seed=choice_order_seed,
        )
        return answer, qa_tokens, refine_tokens, prompt

    def get_memory_stats(self) -> Dict:
        return self.memory_graph.get_memory_stats()

    def clear(self):
        self.memory_graph.clear()
        self.timeline.get_and_reset_token_counts()

    def get_memory_count(self) -> int:
        return self.memory_graph.get_node_count()

    def save_memory_snapshot(self, directory: Path):
        self.memory_graph.save_snapshot(directory)
