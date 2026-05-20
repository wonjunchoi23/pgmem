"""
Timeline Retriever for Theanine — LoComo batch-enabled variant
"""

import json
import logging
import random
from collections import deque
from typing import Dict, List, Tuple

import config as cfg
from memory_graph import MemoryGraph, MemoryNode, _extract_token_info

REFINEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "refined_text": {"type": "string"},
    },
    "required": ["refined_text"],
}

logger = logging.getLogger(__name__)


TIMELINE_REFINEMENT_PROMPT = """\
Given Timelines, which are structured in this format: [Event A] - (relation) - [Event B] ..., your job is to naturally transform each timeline into useful information that can help a language model answer the Question about the Current Dialogue.

These are the explanation of each relation type:
1. Changed: when events in [Event A] changed to events in [Event B]
2. Cause: when events in [Event A] caused events in [Event B]
3. Reason: when events in [Event A] are due to events in [Event B]
4. HinderedBy: when events in [Event B] can be hindered by events in [Sentence A], and vice versa
5. React: when, as a result of events in [Event A], the subject feels as mentioned in [Event B]
6. Want: when, as a result of events in [Event A], the subject wants events in [Event B] to happen
7. SameTopic: when the specific topic addressed in [Event A] is also discussed in [Event B]

If a given relation is not proper, naturally connect them without using that relation.

Current Dialogue:
{current_dialogue}

Question:
{question}

Timelines:
{input_path}

Respond with a JSON object: {{"refined_text": "the transformed timeline as natural language"}}"""


class TimelineRetriever:
    def __init__(self, llm_client, model_path: str = ""):
        self.llm_client = llm_client
        self.model_path = model_path
        self._llm_logger = None
        self._refine_tokens = {"input": 0, "output": 0, "calls": 0}

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Path building
    # ------------------------------------------------------------------

    def get_all_path(self, search_node_id: str, memory_graph: MemoryGraph) -> List[tuple]:
        nodes = memory_graph.nodes
        if search_node_id not in nodes:
            logger.warning(f"get_all_path: {search_node_id} not in memory graph")
            return [(search_node_id,)]

        memory_past = {nid: {} for nid in nodes}
        memory_future = {nid: {} for nid in nodes}

        for nid, node in nodes.items():
            for sub_nid, relation in node.links.items():
                if sub_nid not in nodes:
                    continue
                sub_node = nodes[sub_nid]
                if node.finalize_idx > sub_node.finalize_idx:
                    memory_past[nid][sub_nid] = relation
                else:
                    memory_future[nid][sub_nid] = relation

        past_paths = []
        past_search = deque(
            (head, memory_past[search_node_id][head], search_node_id)
            for head in memory_past[search_node_id]
        )
        while past_search:
            search_path = past_search.popleft()
            first_head = search_path[0]
            next_heads = list(memory_past[first_head].keys())
            if not next_heads:
                past_paths.append(search_path)
            else:
                for next_head in next_heads:
                    past_search.append((next_head, memory_past[first_head][next_head]) + search_path)

        future_paths = []
        future_search = deque(
            (search_node_id, memory_future[search_node_id][tail], tail)
            for tail in memory_future[search_node_id]
        )
        while future_search:
            search_path = future_search.popleft()
            last_tail = search_path[-1]
            next_tails = list(memory_future[last_tail].keys())
            if not next_tails:
                future_paths.append(search_path)
            else:
                for next_tail in next_tails:
                    future_search.append(search_path + (memory_future[last_tail][next_tail], next_tail))

        all_path = []
        if not future_paths and not past_paths:
            all_path.append((search_node_id,))
        elif not future_paths:
            all_path += past_paths
        elif not past_paths:
            all_path += future_paths
        else:
            for past_path in past_paths:
                for future_path in future_paths:
                    all_path.append(past_path[:-1] + future_path)
        return all_path

    def get_path_text(self, path: tuple, memory_graph: MemoryGraph) -> str:
        nodes = memory_graph.nodes
        text = ""
        for i, element in enumerate(path):
            if i % 2 == 0:
                node = nodes.get(element)
                summary = node.summary if node else element
                text += f"[{summary}] - "
            else:
                text += f"({element}) - "
        return text[:-2]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_timeline(self, query: str, memory_graph: MemoryGraph) -> Dict:
        retrieved = memory_graph.retrieve(query, k=cfg.RETRIEVE_TOP_K)
        retrieved.sort(key=lambda x: x[0].finalize_idx, reverse=True)

        timeline = []
        use_timeline = []
        for node, score in retrieved:
            all_paths = self.get_all_path(node.node_id, memory_graph)
            timeline.append({
                "retrieved_node": node.node_id,
                "all_timeline": all_paths,
            })

            all_paths_ = all_paths.copy()
            nothing_to_add = False
            while not nothing_to_add:
                chosen_path = random.sample(all_paths_, k=1)
                all_paths_.remove(chosen_path[0])
                if chosen_path[0] in use_timeline:
                    nothing_to_add = False
                else:
                    use_timeline.append(chosen_path[0])
                    nothing_to_add = True
                if len(all_paths_) == 0:
                    nothing_to_add = True

        return {
            "retrieved_nodes": [(node.node_id, score) for node, score in retrieved],
            "use_timeline": use_timeline,
            "timeline": timeline,
        }

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def build_refine_prompt(self, path_text: str, current_dialogue: str, question: str = "") -> str:
        return TIMELINE_REFINEMENT_PROMPT.format(
            current_dialogue=current_dialogue,
            question=question,
            input_path=path_text,
        )

    def refine_timeline(self, path_text: str, current_dialogue: str, question: str = "") -> Tuple[str, Dict]:
        prompt = self.build_refine_prompt(path_text, current_dialogue, question=question)
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_TOKENS,
                guided_json=REFINEMENT_SCHEMA,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
            token_info = _extract_token_info(result, self.model_path)
            text = result.get("refined_text", "") if isinstance(result, dict) else ""
            if self._llm_logger is not None:
                self._llm_logger.log("call_1_refinement", "", prompt, result)
        except json.JSONDecodeError:
            logger.warning("refine_timeline: JSON parse failed after all retries, retrying without guided_json")
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    temperature=cfg.TEMPERATURE,
                    max_tokens=cfg.MAX_TOKENS,
                    return_usage=True,
                )
                token_info = _extract_token_info(result, self.model_path)
                text = result.get("content", path_text) if isinstance(result, dict) else path_text
                if self._llm_logger is not None:
                    self._llm_logger.log("call_1_refinement", "", prompt, result)
            except Exception:
                logger.warning("refine_timeline: fallback also failed, using raw path_text")
                token_info = {"input": 0, "output": 0}
                text = path_text

        self._refine_tokens["input"] += token_info["input"]
        self._refine_tokens["output"] += token_info["output"]
        self._refine_tokens["calls"] += 1
        return text, token_info

    def refine_all(
        self,
        use_timelines: List[tuple],
        current_dialogue: str,
        memory_graph: MemoryGraph,
        question: str = "",
    ) -> Tuple[List[str], Dict]:
        refined_texts = []
        total_input = total_output = 0
        for path in use_timelines:
            path_text = self.get_path_text(path, memory_graph)
            refined, token_info = self.refine_timeline(path_text, current_dialogue, question=question)
            refined_texts.append(refined)
            total_input += token_info["input"]
            total_output += token_info["output"]
        return refined_texts, {
            "input": total_input,
            "output": total_output,
            "llm_calls": len(use_timelines),
        }

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_and_reset_token_counts(self) -> Dict:
        counts = {
            "input": self._refine_tokens["input"],
            "output": self._refine_tokens["output"],
            "llm_calls": self._refine_tokens["calls"],
        }
        self._refine_tokens = {"input": 0, "output": 0, "calls": 0}
        return counts
