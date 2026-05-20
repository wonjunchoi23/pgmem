"""
Timeline Retriever for Theanine-New — ImplexConv

Based on original Theanine (NAACL 2025) src/theanine.py:
  - Theanine.get_all_path()       → TimelineRetriever.get_all_path()
  - Theanine.get_path_text()      → TimelineRetriever.get_path_text()
  - Theanine.retrieve_timeline()  → TimelineRetriever.retrieve_timeline()
  - Theanine.link_refinement()    → TimelineRetriever.refine_timeline()
  (+ refine_all() convenience wrapper used in theanine_module.py)

Adaptations for exp_implexconv:
  - Uses MemoryNode objects and conv_id instead of string session keys + int(node[1])
  - memory_graph.retrieve() returns [(MemoryNode, score)] instead of [(node_id, score)]
  - llm_client.generate() instead of LangChain LLMChain
  - TIMELINE_REFINEMENT_PROMPT as inline string (identical to timeline-refinement.txt)
  - Token tracking via get_and_reset_token_counts()

Node key format: "c{conv_id}-m{idx}"  (original: "s{session_num}-m{idx}")
Mapping:
  original "session_num"  ≈  conv_id in this experiment
  original int(node[1])   ≈  node.conv_id
"""

import json
import random
import logging
from typing import Dict, List, Tuple

import config as cfg
from memory_graph import MemoryGraph, MemoryNode


# =============================================================================
# JSON SCHEMA  (used with guided_json to prevent <think> tokens in memory)
# =============================================================================

REFINEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "refined_text": {"type": "string"},
    },
    "required": ["refined_text"],
}
_REFINEMENT_GUIDED_JSON = REFINEMENT_SCHEMA

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT  (inline; adapted from original timeline-refinement.txt)
# =============================================================================

# From timeline-refinement.txt — adapted for separate QA question input
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


# =============================================================================
# HELPERS
# =============================================================================

def _extract_token_info(result, model_path: str = "") -> Dict:
    """Extract token usage from llm_client.generate() result."""
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(result, dict) and "_usage" in result:
        usage = result["_usage"]
        token_info["input"]  = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def _extract_text(result) -> str:
    """
    Extract text content from llm_client.generate() result for non-JSON calls.

    vLLM with return_usage=True returns {'content': text, '_usage': {...}}.
    Together / OpenAI return the text string directly.
    """
    if isinstance(result, dict):
        return result.get("content", result.get("text", ""))
    return str(result) if result else ""


# =============================================================================
# TIMELINE RETRIEVER
# =============================================================================

class TimelineRetriever:
    """
    Timeline path builder and refinement for Theanine.

    Adapts the path traversal (get_all_path, get_path_text),
    retrieval (retrieve_timeline), and LLM-based refinement
    (refine_timeline, refine_all) from original Theanine.

    Based on Theanine.get_all_path(), get_path_text(),
    retrieve_timeline(), link_refinement() in src/theanine.py.
    """

    def __init__(self, llm_client, model_path: str = ""):
        self.llm_client  = llm_client
        self.model_path  = model_path
        self._llm_logger = None  # set via set_llm_logger()

        # Internal token counters for refinement calls
        self._refine_input  = 0
        self._refine_output = 0
        self._refine_calls  = 0

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    def build_refine_prompt(self, path_text: str, current_dialogue: str, question: str = "") -> str:
        """Build the refinement prompt without making an LLM call."""
        return TIMELINE_REFINEMENT_PROMPT.format(
            current_dialogue=current_dialogue,
            question=question,
            input_path=path_text,
        )

    def parse_refine_result(self, result, path_text: str) -> str:
        """Extract refined text from a raw LLM result."""
        if isinstance(result, dict):
            return result.get("refined_text", "") or path_text
        text = _extract_text(result)
        return text or path_text

    def accumulate_usage(self, input_tokens: int, output_tokens: int, api_calls: int = 1):
        """Accumulate token counts from an external batch refinement call."""
        self._refine_input += input_tokens
        self._refine_output += output_tokens
        self._refine_calls += api_calls

    # =========================================================================
    # Path building  (Theanine.get_all_path + get_path_text)
    # =========================================================================

    def get_all_path(self, search_node_id: str, memory_graph: MemoryGraph) -> List[tuple]:
        """
        Get all timeline paths through search_node_id in the memory graph.
        Adaptation of Theanine.get_all_path().

        Builds memory_past / memory_future adjacency dicts from the bidirectional
        links stored on each MemoryNode, then runs BFS in both the past (older
        conv_ids) and future (newer conv_ids) directions from search_node_id.
        Combines past and future paths at search_node_id.

        Uses node.conv_id for ordering instead of int(node[1]) in the original.

        Args:
            search_node_id: node_id of the retrieved node to build paths from.
            memory_graph:   MemoryGraph containing all registered nodes.

        Returns:
            List of tuples, each an alternating sequence:
              (node_id, relation, node_id, relation, ..., node_id)
            At minimum [(search_node_id,)] if search_node_id has no links.
        """
        nodes = memory_graph.nodes

        if search_node_id not in nodes:
            logger.warning(f"get_all_path: {search_node_id} not in memory graph")
            return [(search_node_id,)]

        # Build directional adjacency from bidirectional links
        # memory_past[node]   → {sub_node: relation}  where sub_node.conv_id < node.conv_id
        # memory_future[node] → {sub_node: relation}  where sub_node.conv_id > node.conv_id
        # (Same as original: if int(node[1]) > int(sub_node[1]) → memory_past)
        memory_past   = {nid: {} for nid in nodes}
        memory_future = {nid: {} for nid in nodes}

        for nid, node in nodes.items():
            for sub_nid, relation in node.links.items():
                if sub_nid not in nodes:
                    continue
                sub_node = nodes[sub_nid]
                if node.conv_id > sub_node.conv_id:
                    # node is newer → sub_node is in node's past
                    memory_past[nid][sub_nid] = relation
                else:
                    # node is older → sub_node is in node's future
                    memory_future[nid][sub_nid] = relation

        # --- BFS backwards (towards older conv_ids) ---
        # Each element of past_search is a tuple representing a path:
        # (oldest_node_id, relation, ..., relation, search_node_id)
        past_paths  = []
        past_search = []
        for head in list(memory_past[search_node_id].keys()):
            past_search.append((head, memory_past[search_node_id][head], search_node_id))

        while past_search:
            search_path = past_search[0]
            first_head  = search_path[0]
            next_heads  = list(memory_past[first_head].keys())
            if not next_heads:
                past_paths.append(search_path)
                past_search = past_search[1:]
            else:
                for next_head in next_heads:
                    new_path = (next_head, memory_past[first_head][next_head],) + search_path
                    past_search.append(new_path)
                past_search = past_search[1:]

        # --- BFS forwards (towards newer conv_ids) ---
        # Each element of future_search is a tuple representing a path:
        # (search_node_id, relation, ..., relation, newest_node_id)
        future_paths  = []
        future_search = []
        for tail in list(memory_future[search_node_id].keys()):
            future_search.append((search_node_id, memory_future[search_node_id][tail], tail))

        while future_search:
            search_path = future_search[0]
            last_tail   = search_path[-1]
            next_tails  = list(memory_future[last_tail].keys())
            if not next_tails:
                future_paths.append(search_path)
                future_search = future_search[1:]
            else:
                for next_tail in next_tails:
                    new_path = search_path + (memory_future[last_tail][next_tail], next_tail,)
                    future_search.append(new_path)
                future_search = future_search[1:]

        # --- Combine ---
        # past_path ends with search_node_id; future_path starts with search_node_id.
        # Combine: past_path[:-1] + future_path removes the duplicate search_node_id.
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
        """
        Convert a path tuple to human-readable text.
        Adaptation of Theanine.get_path_text().

        Even indices → "[node.summary] - "
        Odd indices  → "(relation) - "
        Returns text[:-2] to strip trailing " -" (same as original).

        Args:
            path:         Alternating (node_id, relation, node_id, ...) tuple.
            memory_graph: MemoryGraph for node lookup.

        Returns:
            Formatted path string, e.g. "[fact1] - (Cause) - [fact2]"
        """
        nodes = memory_graph.nodes
        text  = ""
        for i, element in enumerate(path):
            if i % 2 == 0:
                node  = nodes.get(element)
                summary = node.summary if node else element
                text += f"[{summary}] - "
            else:
                text += f"({element}) - "
        return text[:-2]

    # =========================================================================
    # Timeline retrieval  (Theanine.retrieve_timeline)
    # =========================================================================

    def retrieve_timeline(self, query: str, memory_graph: MemoryGraph) -> Dict:
        """
        Retrieve top-k nodes and build timeline paths.
        Adaptation of Theanine.retrieve_timeline().

        Retrieves top-k nodes by cosine similarity, sorts by conv_id descending
        (most recent first), then samples up to TIMELINE_SAMPLE_N unique paths
        per retrieved node via random.sample. When TIMELINE_SAMPLE_N == 1, this
        matches the original one-path sampling loop.

        Args:
            query:        Query string (current accumulated dialogue).
            memory_graph: MemoryGraph to retrieve from.

        Returns:
            {
              "retrieved_nodes": [(node_id, score), ...],
              "use_timeline":    [path_tuple, ...],   # sampled unique paths
              "timeline":        [{"retrieved_node": node_id,
                                   "all_timeline":   [path, ...]}, ...]
            }
        """
        retrieved = memory_graph.retrieve(query, k=cfg.RETRIEVE_TOP_K)

        # Sort by conv_id descending — same as original:
        # retrieved_nodes.sort(key=lambda x: int(x[0][1]), reverse=True)
        retrieved.sort(key=lambda x: x[0].conv_id, reverse=True)

        timeline     = []
        use_timeline = []
        all_timeline = {}

        for node, score in retrieved:
            all_paths = self.get_all_path(node.node_id, memory_graph)
            timeline.append({
                "retrieved_node": node.node_id,
                "all_timeline":   all_paths,
            })

            # Sample up to TIMELINE_SAMPLE_N unique paths not already in
            # use_timeline. With TIMELINE_SAMPLE_N == 1 this matches the
            # original while loop (nothing_to_add pattern).
            all_paths_ = all_paths.copy()
            sampled_count = 0
            while all_paths_ and sampled_count < cfg.TIMELINE_SAMPLE_N:
                chosen_path = random.sample(all_paths_, k=1)[0]
                all_paths_.remove(chosen_path)
                if chosen_path in use_timeline:
                    continue
                use_timeline.append(chosen_path)
                sampled_count += 1

        all_timeline["retrieved_nodes"] = [(node.node_id, score) for node, score in retrieved]
        all_timeline["use_timeline"]    = use_timeline
        all_timeline["timeline"]        = timeline
        return all_timeline

    # =========================================================================
    # Refinement  (Theanine.link_refinement)
    # =========================================================================

    def refine_timeline(
        self,
        path_text: str,
        current_dialogue: str,
        question: str = "",
    ) -> Tuple[str, Dict]:
        """
        Refine a single timeline path text into natural language.
        Adaptation of Theanine.link_refinement(). 1 LLM call.

        Args:
            path_text:        Formatted path string from get_path_text().
            current_dialogue: Accumulated dialogue string for context.
            question:         QA question kept separate from dialogue context.

        Returns:
            (refined_text, token_info)
        """
        prompt = self.build_refine_prompt(path_text, current_dialogue, question)
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
            text = self.parse_refine_result(result, path_text)
            if self._llm_logger is not None:
                self._llm_logger.log("call_2_refinement", "", prompt, result)
        except json.JSONDecodeError:
            logger.warning("_refine_one: JSON parse failed after all retries, retrying without guided_json")
            try:
                result = self.llm_client.generate(
                    prompt=prompt,
                    temperature=cfg.TEMPERATURE,
                    max_tokens=cfg.MAX_TOKENS,
                    return_usage=True,
                )
                token_info = _extract_token_info(result, self.model_path)
                text = self.parse_refine_result(result, path_text)
                if self._llm_logger is not None:
                    self._llm_logger.log("call_2_refinement", "", prompt, result)
            except Exception:
                logger.warning("_refine_one: fallback also failed, using raw path_text")
                token_info = {"input": 0, "output": 0}
                text = path_text

        self._refine_input  += token_info["input"]
        self._refine_output += token_info["output"]
        self._refine_calls  += 1

        return text, token_info

    def refine_all(
        self,
        use_timelines: List[tuple],
        current_dialogue: str,
        memory_graph: MemoryGraph,
        question: str = "",
    ) -> Tuple[List[str], Dict]:
        """
        Convert all sampled path tuples to texts and refine each one.

        Convenience wrapper for the loop in original theanine_all():
          for timeline in timelines["use_timeline"]:
              input_path = self.get_path_text(timeline)
              result, cost = self.link_refinement(current_dialogue, input_path)
              input_memory.append(result)

        Args:
            use_timelines:    List of path tuples from retrieve_timeline().
            current_dialogue: Accumulated dialogue string.
            memory_graph:     MemoryGraph for node lookup during get_path_text().
            question:         QA question kept separate from dialogue context.

        Returns:
            (refined_texts, total_token_info)
            refined_texts:    One refined string per path (input to Generator).
            total_token_info: {"input": ..., "output": ..., "llm_calls": ...}
        """
        refined_texts = []
        total_input = total_output = 0

        for path in use_timelines:
            path_text = self.get_path_text(path, memory_graph)
            refined, token_info = self.refine_timeline(path_text, current_dialogue, question)
            refined_texts.append(refined)
            total_input  += token_info["input"]
            total_output += token_info["output"]

        return refined_texts, {
            "input":     total_input,
            "output":    total_output,
            "llm_calls": len(use_timelines),
        }

    # =========================================================================
    # Token tracking
    # =========================================================================

    def get_and_reset_token_counts(self) -> Dict:
        """
        Return accumulated refinement token counts and reset all counters.

        Returns:
            {"input": ..., "output": ..., "llm_calls": ...}
        """
        counts = {
            "input":     self._refine_input,
            "output":    self._refine_output,
            "llm_calls": self._refine_calls,
        }
        self._refine_input  = 0
        self._refine_output = 0
        self._refine_calls  = 0
        return counts
