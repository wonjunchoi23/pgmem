"""
Memory Graph for Theanine — PersonaMem variant

Internal field names (`session_id`, `conv_id`, `turn_id`, node key
`c{conv_id}-m{idx}`) are retained from the ImplexConv variant. The runner
injects PersonaMem identifiers into those slots:
  context_index → session_id    block_idx → conv_id    local_msg_idx → turn_id
Result metadata is remapped in run_experiment.py.

Based on original Theanine (NAACL 2025) src/ modules:
  - src/summarize.py        → _process_text(), _summarize_conv()
  - src/retriever.py        → cos_sim(), _embed_nodes(), retrieve(), _find_associative()
  - src/memory_constructor.py → _extract_relation(), _find_links(), finalize_conv()
"""

import re
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

import config as cfg

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_TIKTOKEN_ENC.encode(text))
except Exception:
    def _count_tokens(text: str) -> int:
        return len(text) // 4

def _truncate_oldest_first(text: str, max_tokens: int) -> str:
    """Drop leading newline-separated lines until token count ≤ max_tokens."""
    lines = text.split("\n")
    original = len(lines)
    while len(lines) > 1 and _count_tokens("\n".join(lines)) > max_tokens:
        lines.pop(0)
    if len(lines) < original:
        logger.warning(
            "dialogue truncated from %d to %d lines to fit %d-token budget",
            original, len(lines), max_tokens,
        )
    return "\n".join(lines)


# =============================================================================
# PROMPTS  (inline; identical to original Theanine resources/prompts/)
# =============================================================================

# From dialogue-summarization.txt — unchanged (output format changed to JSON)
SUMMARIZATION_PROMPT = """\
Summarize the given dialogue by extracting useful information about the speakers.
Only write down the facts about the speakers that can be deduced from what they say.

<Example>
[Dialogue]
Speaker A: Hello, how are you?
Speaker B: Great I cant wait for easter weekend!
Speaker A: Same here! I get to see all eight of my grandchildren.
Speaker B: I hope my new blue hair matches my new easter dress.
Speaker A: I'm sure it will! At my old age, I don't think I could pull it off.
Speaker B: I turn 29 next week, I cant wait to leave for spring break.
Speaker A: Where are you going? My spring breaks used to be quiet and not too crazy.
Speaker B: My mom just gave me a new car, so ill just be driving.
Speaker A: Very nice! A librarian was my mother's profession.
Speaker B: Are you a librarian too?
Speaker A: I do like the quiet nature of it, but I am retired.
Speaker B: What do you enjoy doing in nature.

[Summary]
{{"sentences": [
  "Speaker A is older and retired, mentioning grandchildren.",
  "Speaker B is 28 years old, turning 29 next week.",
  "Speaker B is excited about Easter weekend and spring break.",
  "Speaker B recently got a new blue hair color and a new Easter dress.",
  "Speaker A used to have quiet spring breaks, contrasting with Speaker B's plans for a more adventurous one.",
  "Speaker B received a new car from their mom and plans to drive during spring break.",
  "Speaker A's mother and Speaker A were both a librarian but Speaker A is retired despite enjoying the quiet nature of it."
]}}

<Your turn>
[Dialogue]
{dialogue}

Respond with a JSON object: {{"sentences": ["fact1", "fact2", ...]}}"""

# Plain-text fallback prompt — used when JSON summarization fails after all retries.
# Outputs a numbered list parseable by _process_text().
SUMMARIZATION_PROMPT_PLAINTEXT = """\
Summarize the given dialogue by extracting useful information about the speakers.
Only write down the facts about the speakers that can be deduced from what they say.
List the sentences with numbers.

<Example>
[Dialogue]
Speaker A: Hello, how are you?
Speaker B: Great I cant wait for easter weekend!
Speaker A: Same here! I get to see all eight of my grandchildren.
Speaker B: I hope my new blue hair matches my new easter dress.
Speaker A: I'm sure it will! At my old age, I don't think I could pull it off.
Speaker B: I turn 29 next week, I cant wait to leave for spring break.
Speaker A: Where are you going? My spring breaks used to be quiet and not too crazy.
Speaker B: My mom just gave me a new car, so ill just be driving.
Speaker A: Very nice! A librarian was my mother's profession.
Speaker B: Are you a librarian too?
Speaker A: I do like the quiet nature of it, but I am retired.
Speaker B: What do you enjoy doing in nature.

[Summary]
1. Speaker A is older and retired, mentioning grandchildren.
2. Speaker B is 28 years old, turning 29 next week.
3. Speaker B is excited about Easter weekend and spring break.
4. Speaker B recently got a new blue hair color and a new Easter dress.
5. Speaker A used to have quiet spring breaks, contrasting with Speaker B's plans for a more adventurous one.
6. Speaker B received a new car from their mom and plans to drive during spring break.
7. Speaker A's mother and Speaker A were both a librarian but Speaker A is retired despite enjoying the quiet nature of it.

<Your turn>
[Dialogue]
{dialogue}

[Summary]"""

# From relation-extraction.txt — unchanged (output format changed to JSON)
RELATION_EXTRACTION_PROMPT = """\
Your task is to find the relation between [Sentence A] and [Sentence B].
Keep in mind that [Sentence A] happened before [Sentence B].
The dialogues where each of the sentence is originated from are provided to help your reasoning.

First, identify if the relation holds among the following six relations:
1. Changed: when events in [Sentence A] changed to events in [Sentence B]
2. Cause: when events in [Sentence A] caused events in [Sentence B]
3. Reason: when events in [Sentence A] are due to events in [Sentence B]
4. HinderedBy: when events in [Sentence B] can be hindered by events in [Sentence A], and vice versa
5. React: when, as a result of events in [Sentence A], the subject feels as mentioned in [Sentence B]
6. Want: when, as a result of events in [Sentence A], the subject wants events in [Sentence B] to happen

Then, if the relation does not belong to any of the relations from 1 to 6, choose between the following two options:
7. SameTopic: when the specific topic addressed in [Sentence A] is also discussed in [Sentence B]
8. None: when [Sentence A] and [Sentence B] are irrelevant

- For relations from 1 to 7, choose them only if there is clear evidence that matches the description of the relation. Otherwise, just choose "None" without making excessive inferences beyond the given sentence.
- Pay attention to who the subject of each sentence is.
- Do not confuse the roles of [Sentence A] and [Sentence B] when determining the relationship.

Now, read the two dialogues and find the relation between [Sentence A] and [Sentence B].

[Dialogue for Sentence A]:
{dialogue1}
[Dialogue for Sentence B]:
{dialogue2}

[Sentence A]: {sentence1}
[Sentence B]: {sentence2}

Respond with a JSON object: {{"explanation": "your reasoning", "relation": "one of Changed|Cause|Reason|HinderedBy|React|Want|SameTopic|None"}}"""


# =============================================================================
# JSON SCHEMAS  (used with guided_json to prevent <think> tokens in memory)
# =============================================================================

SUMMARIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 15,   # xgrammar enforces ] after 15 items → prevents truncation
        }
    },
    "required": ["sentences"],
}
_SUMMARIZATION_GUIDED_JSON = SUMMARIZATION_SCHEMA

RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "relation": {
            "type": "string",
            "enum": ["Changed", "Cause", "Reason", "HinderedBy",
                     "React", "Want", "SameTopic", "None"],
        },
    },
    "required": ["explanation", "relation"],
}
_RELATION_GUIDED_JSON = RELATION_SCHEMA


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
# DATA CLASS
# =============================================================================

@dataclass
class MemoryNode:
    """
    A single memory node in the Theanine relation-aware memory graph.

    Corresponds to one fact sentence extracted from a conv_id dialogue summary.
    Node key format: "c{conv_id}-m{idx}" (1-indexed).
    (Original: "s{session_num}-m{idx}")
    """
    node_id: str                        # e.g., "c0-m1"
    summary: str                        # fact sentence from summarization
    conv_id: int                        # source conv_id
    session_id: int                     # source session_id
    source_conv_dialogue: str           # full dialogue of source conv
                                        # (used as context in relation extraction)
    turn_id_start: int = -1             # turn_id of first turn in source conv
    turn_id_end: int = -1               # turn_id of last turn in source conv
    global_turn_id_start: int = -1      # global_turn_id of first turn in source conv
    global_turn_id_end: int = -1        # global_turn_id of last turn in source conv
    embedding: Optional[np.ndarray] = None
    links: Dict[str, str] = field(default_factory=dict)  # {node_id: relation_type}


# =============================================================================
# MEMORY GRAPH
# =============================================================================

class MemoryGraph:
    """
    Relation-aware memory graph for Theanine.

    Manages memory nodes created from conv_id-level dialogue summarization,
    embeds them with sentence-transformers, and builds a directed graph via
    relation-aware linking (ATOMIC 2020 relation types).

    Memory accumulates across conv_ids within a session; cleared between sessions.

    Based on:
      src/summarize.py        → _process_text(), _summarize_conv()
      src/retriever.py        → cos_sim(), _embed_nodes(), retrieve(),
                                 _find_associative()
      src/memory_constructor.py → _extract_relation(), _find_links(),
                                   finalize_conv()
    """

    def __init__(self, llm_client, config, embedding_model: Optional[SentenceTransformer] = None):
        self.nodes: Dict[str, MemoryNode] = {}   # node_id → MemoryNode
        self.llm_client  = llm_client
        self.encoder     = embedding_model or SentenceTransformer(config.EMBEDDING_MODEL)
        self._llm_logger = None  # set via set_llm_logger()

        # Internal token counters (reset via get_and_reset_token_counts())
        self._summarize_input         = 0
        self._summarize_output        = 0
        self._summarize_calls         = 0
        self._summarize_fallback_count = 0
        self._relation_input          = 0
        self._relation_output         = 0
        self._relation_calls          = 0

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    # =========================================================================
    # Summarization  (src/summarize.py)
    # =========================================================================

    def _process_text(self, result: str) -> List[str]:
        """
        Strip numbered list formatting → list of fact sentences.
        Same as Summarizer.process_text().
        """
        pattern = re.compile(r"[0-9]+\.\s")
        sentences = []
        for line in result.strip().split("\n"):
            refined = pattern.sub("", line).strip()
            if refined:
                sentences.append(refined)
        return sentences

    def build_summarize_prompt(self, dialogue: str, plaintext_fallback: bool = False) -> str:
        """Build the summarization prompt without making an LLM call."""
        budget = int(cfg.FINALIZE_INPUT_CONTEXT_LIMIT * cfg.FINALIZE_CONTEXT_UTILIZATION) - cfg.SUMMARIZE_MAX_TOKENS
        dialogue = _truncate_oldest_first(dialogue, budget)
        template = SUMMARIZATION_PROMPT_PLAINTEXT if plaintext_fallback else SUMMARIZATION_PROMPT
        return template.format(dialogue=dialogue)

    def parse_summarize_result(self, result, plaintext_fallback: bool = False) -> List[str]:
        """Parse a summarization result into fact sentences."""
        if plaintext_fallback:
            text = result.get("content", "") if isinstance(result, dict) else str(result or "")
            return self._process_text(text)
        if isinstance(result, dict):
            sentences = result.get("sentences", [])
            return [s for s in sentences if isinstance(s, str) and s.strip()]
        return []

    def summarize_plaintext_fallback(self, dialogue: str) -> Tuple[List[str], Dict, str, Any]:
        """Run the sequential plain-text fallback used when JSON summarization fails."""
        prompt = self.build_summarize_prompt(dialogue, plaintext_fallback=True)
        result = self.llm_client.generate(
            prompt=prompt,
            temperature=cfg.TEMPERATURE,
            max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
            return_usage=True,
        )
        token_info = _extract_token_info(result)
        sentences = self.parse_summarize_result(result, plaintext_fallback=True)
        return sentences, token_info, prompt, result

    def _summarize_conv(self, dialogue: str) -> Tuple[List[str], Dict]:
        """
        Summarize a conv_id dialogue → list of fact sentences.  1 LLM call.
        Same as Summarizer.summarize() / generate_gpt_response().

        Args:
            dialogue: Full formatted dialogue string of the conv.
                      Format: "User: ...\nAssistant: ...\nUser: ..."

        Returns:
            (sentences, token_info)
        """
        # --- Primary: JSON mode with constrained decoding ---
        prompt = self.build_summarize_prompt(dialogue)
        sentences = []
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                guided_json=_SUMMARIZATION_GUIDED_JSON,
                return_usage=True,
            )
            token_info = _extract_token_info(result)
            sentences = self.parse_summarize_result(result)
            if self._llm_logger is not None:
                self._llm_logger.log("call_3_summarization", "", prompt, result)

        except Exception as json_err:
            # --- Fallback: plain-text numbered list (all JSON retries exhausted) ---
            logger.warning(
                f"_summarize_conv: JSON mode failed after all retries "
                f"({json_err}). Falling back to plain-text summarization."
            )
            self._summarize_fallback_count += 1
            try:
                sentences, token_info, fallback_prompt, fb_result = self.summarize_plaintext_fallback(dialogue)
                if self._llm_logger is not None:
                    self._llm_logger.log("call_3_summarization", "", fallback_prompt, fb_result)
                logger.info(
                    f"_summarize_conv: plain-text fallback recovered "
                    f"{len(sentences)} sentence(s)."
                )
            except Exception as fb_err:
                logger.error(
                    f"_summarize_conv: plain-text fallback also failed ({fb_err}). "
                    f"Returning empty sentence list."
                )
                token_info = {"input": 0, "output": 0}

        self._summarize_input  += token_info["input"]
        self._summarize_output += token_info["output"]
        self._summarize_calls  += 1
        return sentences, token_info

    def _create_nodes_from_summary(
        self,
        sentences: List[str],
        conv_id: int,
        session_id: int,
        source_conv_dialogue: str,
        turn_id_start: int = -1,
        turn_id_end: int = -1,
        global_turn_id_start: int = -1,
        global_turn_id_end: int = -1,
    ) -> List[MemoryNode]:
        """
        Create MemoryNode objects from fact sentences.
        Same as Summarizer.create_node().

        Node IDs: "c{conv_id}-m1", "c{conv_id}-m2", ... (1-indexed).
        """
        nodes = []
        for idx, sentence in enumerate(sentences):
            node_id = f"c{conv_id}-m{idx + 1}"
            nodes.append(MemoryNode(
                node_id=node_id,
                summary=sentence,
                conv_id=conv_id,
                session_id=session_id,
                source_conv_dialogue=source_conv_dialogue,
                turn_id_start=turn_id_start,
                turn_id_end=turn_id_end,
                global_turn_id_start=global_turn_id_start,
                global_turn_id_end=global_turn_id_end,
            ))
        return nodes

    # =========================================================================
    # Embedding & Retrieval  (src/retriever.py)
    # =========================================================================

    def cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity. Same as Retriever.cos_sim()."""
        return float(dot(a, b) / (norm(a) * norm(b)))

    def _embed_nodes(self, nodes: List[MemoryNode]):
        """
        Embed node summaries in batch and assign embeddings to each node.
        Same as Retriever.memory_to_embedding() (batch variant).
        """
        if not nodes:
            return
        texts      = [node.summary for node in nodes]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        for node, emb in zip(nodes, embeddings):
            node.embedding = emb

    def prepare_new_nodes(
        self,
        sentences: List[str],
        conv_id: int,
        session_id: int,
        source_conv_dialogue: str,
        turn_id_start: int = -1,
        turn_id_end: int = -1,
        global_turn_id_start: int = -1,
        global_turn_id_end: int = -1,
    ) -> List[MemoryNode]:
        """Create and embed nodes from summarization sentences."""
        new_nodes = self._create_nodes_from_summary(
            sentences, conv_id, session_id, source_conv_dialogue,
            turn_id_start, turn_id_end, global_turn_id_start, global_turn_id_end,
        )
        self._embed_nodes(new_nodes)
        return new_nodes

    def retrieve(self, query: str, k: int) -> List[Tuple[MemoryNode, float]]:
        """
        Retrieve top-k nodes by cosine similarity to query.
        Same as Retriever.retrieve_nodes().

        Returns:
            List of (MemoryNode, score) sorted by score descending.
            Empty list if no nodes in the graph.
        """
        if not self.nodes:
            return []
        query_emb = self.encoder.encode([query], convert_to_numpy=True)[0]
        scores = []
        for node in self.nodes.values():
            if node.embedding is not None:
                score = self.cos_sim(query_emb, node.embedding)
                scores.append((node, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _find_associative(self, target_node: MemoryNode, j: int) -> List[MemoryNode]:
        """
        Find top-j similar nodes from all past conv_ids (conv_id < target_node.conv_id).
        Same as Retriever.retrieve_for_linking().

        Only considers nodes already committed to the graph (self.nodes),
        so nodes from the same conv_id batch are never linked to each other.
        """
        past_nodes = [
            n for n in self.nodes.values()
            if n.conv_id < target_node.conv_id and n.embedding is not None
        ]
        if not past_nodes or target_node.embedding is None:
            return []

        scores = [
            (node, self.cos_sim(target_node.embedding, node.embedding))
            for node in past_nodes
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scores[:j]]

    # =========================================================================
    # Relation Extraction & Linking  (src/memory_constructor.py)
    # =========================================================================

    def _extract_relation(
        self,
        sentence1: str,
        sentence2: str,
        dialogue1: str,
        dialogue2: str,
    ) -> Tuple[str, Dict]:
        """
        Extract relation between sentence1 (past) and sentence2 (new).
        1 LLM call.  Same as MemoryConstructor.extract_relations().

        sentence1 happened before sentence2 (sentence1 is from an earlier conv).
        dialogue1 / dialogue2 are the full source conv dialogues (context).

        Returns:
            (relation_str, token_info)
            relation_str: one of Changed | Cause | Reason | HinderedBy |
                          React | Want | SameTopic | None
        """
        prompt = self.build_relation_prompt(
            sentence1=sentence1,
            sentence2=sentence2,
            dialogue1=dialogue1,
            dialogue2=dialogue2,
        )
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_TOKENS,
                guided_json=_RELATION_GUIDED_JSON,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
            token_info = _extract_token_info(result)
            relation = self.parse_relation_result(result)
            if self._llm_logger is not None:
                self._llm_logger.log("call_4_relation", "", prompt, result)
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.warning(f"_extract_relation: generation failed ({type(e).__name__}: {e}), returning 'None'")
            token_info = {"input": 0, "output": 0}
            relation = "None"

        self._relation_input  += token_info["input"]
        self._relation_output += token_info["output"]
        self._relation_calls  += 1

        return relation, token_info

    def build_relation_prompt(
        self,
        sentence1: str,
        sentence2: str,
        dialogue1: str,
        dialogue2: str,
    ) -> str:
        """Build the relation-extraction prompt without making an LLM call."""
        budget = int(cfg.FINALIZE_INPUT_CONTEXT_LIMIT * cfg.FINALIZE_CONTEXT_UTILIZATION) - cfg.MAX_TOKENS
        half = budget // 2
        dialogue1 = _truncate_oldest_first(dialogue1, half)
        dialogue2 = _truncate_oldest_first(dialogue2, half)
        return RELATION_EXTRACTION_PROMPT.format(
            dialogue1=dialogue1,
            dialogue2=dialogue2,
            sentence1=sentence1,
            sentence2=sentence2,
        )

    def parse_relation_result(self, result) -> str:
        """Extract the relation label from a raw LLM result."""
        if isinstance(result, dict):
            return result.get("relation", "None")
        return "None"
    
    def _node_recency_key(self, node: MemoryNode) -> Tuple[int, int]:
        """
        More recent = larger (conv_id, memory_index).
        node_id format: c{conv_id}-m{idx}
        """
        try:
            mem_idx = int(node.node_id.split("-m")[1])
        except Exception:
            mem_idx = 0
        return (node.conv_id, mem_idx)


    def _get_connected_component(self, start_node_id: str) -> List[str]:
        """
        Return the connected component containing start_node_id.
        Connectivity is treated as undirected over existing bidirectional links.
        Only traverses nodes already registered in self.nodes (i.e., G_t).
        """
        if start_node_id not in self.nodes:
            return []

        visited = set()
        stack = [start_node_id]

        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)

            node = self.nodes[nid]
            for neigh_id in node.links.keys():
                if neigh_id in self.nodes and neigh_id not in visited:
                    stack.append(neigh_id)

        return list(visited)


    def _select_representatives_by_component(
        self,
        candidate_nodes: List[MemoryNode],
    ) -> List[MemoryNode]:
        """
        Given M_a* (candidates with non-None relation), group them by connected
        component in current graph G_t and keep only the most recent node from each
        component, following the paper's Eq. (7)-(8).
        """
        if not candidate_nodes:
            return []

        component_to_best: Dict[Tuple[str, ...], MemoryNode] = {}

        for node in candidate_nodes:
            comp_nodes = self._get_connected_component(node.node_id)
            comp_key = tuple(sorted(comp_nodes)) if comp_nodes else (node.node_id,)

            if comp_key not in component_to_best:
                component_to_best[comp_key] = node
            else:
                prev = component_to_best[comp_key]
                if self._node_recency_key(node) > self._node_recency_key(prev):
                    component_to_best[comp_key] = node

        # Most recent representative per component
        reps = list(component_to_best.values())
        reps.sort(key=self._node_recency_key, reverse=True)
        return reps

    def _find_links(
        self,
        new_node: MemoryNode,
        associative_nodes: List[MemoryNode],
    ) -> Dict:
        """
        Paper-faithful linking:
        1) run relation extraction for every associative node in M_a
        2) keep only M_a* = {m in M_a | relation != None}
        3) find connected components in current graph G_t that contain M_a*
        4) link new_node only to the most recent memory in each component

        Returns:
            {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "relation_calls": number_of_relation_extractions,
                "linked_edges": number_of_final_edges_added
            }
        """
        total_input = total_output = 0
        relation_calls = 0

        # Step A: relation extraction on every associative candidate
        sorted_nodes = sorted(
            associative_nodes,
            key=lambda n: self._node_recency_key(n),
            reverse=True
        )

        related_candidates: List[Tuple[MemoryNode, str]] = []

        for past_node in sorted_nodes:
            logger.debug(f"linking {new_node.node_id} with {past_node.node_id} ...")
            relation, token_info = self._extract_relation(
                sentence1=past_node.summary,
                sentence2=new_node.summary,
                dialogue1=past_node.source_conv_dialogue,
                dialogue2=new_node.source_conv_dialogue,
            )
            total_input  += token_info["input"]
            total_output += token_info["output"]
            relation_calls += 1

            if "None" not in relation:
                related_candidates.append((past_node, relation))

        # Step B: build M_a* and select one representative per connected component
        candidate_nodes = [node for node, _ in related_candidates]
        representatives = self._select_representatives_by_component(candidate_nodes)

        # Map node_id -> extracted relation
        relation_map = {node.node_id: relation for node, relation in related_candidates}

        # Step C: final linking only to component representatives
        linked_edges = 0
        for past_node in representatives:
            relation = relation_map[past_node.node_id]
            new_node.links[past_node.node_id] = relation
            past_node.links[new_node.node_id] = relation
            linked_edges += 1

        return {
            "input": total_input,
            "output": total_output,
            "relation_calls": relation_calls,
            "linked_edges": linked_edges,
        }

    def build_relation_jobs(self, new_nodes: List[MemoryNode]) -> List[Dict[str, Any]]:
        """Prepare relation-extraction jobs for a list of new nodes."""
        jobs: List[Dict[str, Any]] = []
        for new_node in new_nodes:
            associative_nodes = sorted(
                self._find_associative(new_node, cfg.LINKING_TOP_J),
                key=lambda n: self._node_recency_key(n),
                reverse=True,
            )
            for past_node in associative_nodes:
                jobs.append({
                    "new_node_id": new_node.node_id,
                    "past_node_id": past_node.node_id,
                    "prompt": self.build_relation_prompt(
                        sentence1=past_node.summary,
                        sentence2=new_node.summary,
                        dialogue1=past_node.source_conv_dialogue,
                        dialogue2=new_node.source_conv_dialogue,
                    ),
                })
        return jobs

    def apply_relation_results(
        self,
        new_nodes: List[MemoryNode],
        relation_jobs: List[Dict[str, Any]],
        relation_results: List[str],
    ) -> Dict[str, int]:
        """Apply relation results, finalize links, and commit the new nodes."""
        new_node_map = {node.node_id: node for node in new_nodes}
        related_by_new_node: Dict[str, List[Tuple[MemoryNode, str]]] = {
            node.node_id: [] for node in new_nodes
        }

        for job, relation in zip(relation_jobs, relation_results):
            if relation == "None":
                continue
            past_node = self.nodes.get(job["past_node_id"])
            new_node = new_node_map.get(job["new_node_id"])
            if past_node is None or new_node is None:
                continue
            related_by_new_node[new_node.node_id].append((past_node, relation))

        linked_edges = 0
        for new_node in new_nodes:
            related_candidates = related_by_new_node.get(new_node.node_id, [])
            candidate_nodes = [node for node, _ in related_candidates]
            representatives = self._select_representatives_by_component(candidate_nodes)
            relation_map = {node.node_id: relation for node, relation in related_candidates}
            for past_node in representatives:
                relation = relation_map[past_node.node_id]
                new_node.links[past_node.node_id] = relation
                past_node.links[new_node.node_id] = relation
                linked_edges += 1

        for node in new_nodes:
            self.nodes[node.node_id] = node

        return {
            "relation_calls": len(relation_jobs),
            "linked_edges": linked_edges,
        }

    # =========================================================================
    # Main Interface
    # =========================================================================

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
        Called after a conv_id completes.
        Summarize → create nodes → embed → link to past nodes → register.

        Corresponds to the original pipeline:
          Summarizer.summarize()          → fact sentences / nodes
          MemoryConstructor.find_all_links() → link new nodes to past nodes

        New nodes are added to self.nodes AFTER linking, so nodes from the same
        conv_id are never linked to each other (matches original cross-session
        linking only behaviour).

        Args:
            conv_id:              The completed conversation ID.
            session_id:           The current session ID.
            full_conv_dialogue:   Formatted dialogue string of the conv.
                                  "User: ...\nAssistant: ...\nUser: ..."
                                  (built from GT turns by run_experiment.py)
            turn_id_start:        turn_id of the first turn in this conv.
            turn_id_end:          turn_id of the last turn in this conv.
            global_turn_id_start: global_turn_id of the first turn in this conv.
            global_turn_id_end:   global_turn_id of the last turn in this conv.

        Returns:
            {"input": ..., "output": ..., "llm_calls": ...}
            Tokens from summarization + all relation extractions for this conv.
        """
        # Step 1: Summarize → fact sentences
        sentences, summ_tokens = self._summarize_conv(full_conv_dialogue)
        if not sentences:
            logger.warning(
                f"finalize_conv: empty summary for "
                f"conv_id={conv_id}, session_id={session_id}"
            )
            return {"input": summ_tokens["input"], "output": summ_tokens["output"], "llm_calls": 1}

        # Step 2: Create node objects
        new_nodes = self._create_nodes_from_summary(
            sentences, conv_id, session_id, full_conv_dialogue,
            turn_id_start, turn_id_end, global_turn_id_start, global_turn_id_end,
        )

        # Step 3: Embed new nodes (batch)
        self._embed_nodes(new_nodes)

        # Step 4: Link each new node to top-j similar past nodes
        # (self.nodes contains only nodes from earlier conv_ids at this point)
        total_relation_input  = 0
        total_relation_output = 0
        relation_calls        = 0
        linked_edges          = 0

        for new_node in new_nodes:
            associative = self._find_associative(new_node, cfg.LINKING_TOP_J)
            if associative:
                rel_info = self._find_links(new_node, associative)
                total_relation_input  += rel_info["input"]
                total_relation_output += rel_info["output"]
                relation_calls        += rel_info["relation_calls"]
                linked_edges          += rel_info["linked_edges"]

        # Step 5: Register new nodes in the graph
        for node in new_nodes:
            self.nodes[node.node_id] = node

        total_llm_calls = 1 + relation_calls  # summarize + relation extractions
        logger.info(
            f"finalize_conv: conv_id={conv_id}, session_id={session_id}, "
            f"nodes_created={len(new_nodes)}, llm_calls={total_llm_calls}, "
            f"linked_edges={linked_edges}"
        )
        return {
            "input":     summ_tokens["input"]  + total_relation_input,
            "output":    summ_tokens["output"] + total_relation_output,
            "llm_calls": total_llm_calls,
        }

    def get_node_count(self) -> int:
        """Return total number of memory nodes currently in the graph."""
        return len(self.nodes)

    def accumulate_summarize_usage(self, input_tokens: int, output_tokens: int, api_calls: int = 1):
        """Accumulate token counts from an external batch summarization call."""
        self._summarize_input += input_tokens
        self._summarize_output += output_tokens
        self._summarize_calls += api_calls

    def accumulate_relation_usage(self, input_tokens: int, output_tokens: int, api_calls: int = 1):
        """Accumulate token counts from an external batch relation call."""
        self._relation_input += input_tokens
        self._relation_output += output_tokens
        self._relation_calls += api_calls

    def accumulate_fallback_count(self, n: int = 1):
        """Increment the summarization fallback counter (called from batched path)."""
        self._summarize_fallback_count += n

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def clear(self):
        """
        Reset all memory state.  Called between sessions.
        Clears all nodes and resets token counters.
        """
        self.nodes                     = {}
        self._summarize_input          = 0
        self._summarize_output         = 0
        self._summarize_calls          = 0
        self._summarize_fallback_count = 0
        self._relation_input           = 0
        self._relation_output          = 0
        self._relation_calls           = 0

    def save_snapshot(self, directory: Path):
        """
        Save memory graph state to directory.

        Files written:
          nodes.json        — node metadata + link structure
          embeddings.npy    — embedding matrix (one row per node)
          embedding_ids.json — node_id order aligned with embeddings.npy
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Node metadata and links (JSON-serializable)
        nodes_data = {
            node_id: {
                "node_id":              node.node_id,
                "summary":              node.summary,
                "conv_id":              node.conv_id,
                "session_id":           node.session_id,
                "source_conv_dialogue": node.source_conv_dialogue,
                "turn_id_start":        node.turn_id_start,
                "turn_id_end":          node.turn_id_end,
                "global_turn_id_start": node.global_turn_id_start,
                "global_turn_id_end":   node.global_turn_id_end,
                "links":                node.links,
            }
            for node_id, node in self.nodes.items()
        }
        with open(directory / "nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, indent=2, ensure_ascii=False)

        # Embeddings — numpy array aligned with embedding_ids.json
        nodes_with_emb = [
            (nid, node) for nid, node in self.nodes.items()
            if node.embedding is not None
        ]
        if nodes_with_emb:
            emb_ids    = [nid for nid, _ in nodes_with_emb]
            emb_matrix = np.stack([node.embedding for _, node in nodes_with_emb])
            np.save(directory / "embeddings.npy", emb_matrix)
            with open(directory / "embedding_ids.json", "w") as f:
                json.dump(emb_ids, f)

    def load_snapshot(self, directory: Path):
        """Restore memory graph state from directory."""
        directory  = Path(directory)
        nodes_file = directory / "nodes.json"
        emb_file   = directory / "embeddings.npy"
        ids_file   = directory / "embedding_ids.json"

        if not nodes_file.exists():
            logger.warning(f"load_snapshot: nodes.json not found in {directory}")
            return

        with open(nodes_file, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)

        emb_map: Dict[str, np.ndarray] = {}
        if emb_file.exists() and ids_file.exists():
            embeddings = np.load(emb_file)
            with open(ids_file, "r") as f:
                emb_ids = json.load(f)
            for i, nid in enumerate(emb_ids):
                emb_map[nid] = embeddings[i]

        self.nodes = {}
        for node_id, data in nodes_data.items():
            node = MemoryNode(
                node_id=data["node_id"],
                summary=data["summary"],
                conv_id=data["conv_id"],
                session_id=data["session_id"],
                source_conv_dialogue=data["source_conv_dialogue"],
                links=data["links"],
                embedding=emb_map.get(node_id),
            )
            self.nodes[node_id] = node

    def get_memory_stats(self) -> Dict:
        """Return current memory node count and estimated total content tokens."""
        num_memories = len(self.nodes)
        total_chars = sum(len(node.summary) for node in self.nodes.values())
        total_content_tokens = total_chars // 4
        return {
            "num_memories":         num_memories,
            "total_content_tokens": total_content_tokens,
        }

    def get_and_reset_token_counts_by_type(self) -> Dict:
        """
        Return per-call-type token counts and reset all counters to zero.

        Returns:
            {
                "call_3_summarization": {"input": int, "output": int, "llm_calls": int,
                                         "parse_fallback_count": int},
                "call_4_relation":      {"input": int, "output": int, "llm_calls": int},
            }
        """
        counts = {
            "call_3_summarization": {
                "input":               self._summarize_input,
                "output":              self._summarize_output,
                "llm_calls":           self._summarize_calls,
                "parse_fallback_count": self._summarize_fallback_count,
            },
            "call_4_relation": {
                "input":     self._relation_input,
                "output":    self._relation_output,
                "llm_calls": self._relation_calls,
            },
        }
        self._summarize_input          = 0
        self._summarize_output         = 0
        self._summarize_calls          = 0
        self._summarize_fallback_count = 0
        self._relation_input           = 0
        self._relation_output          = 0
        self._relation_calls           = 0
        return counts

    def get_and_reset_token_counts(self) -> Dict:
        """
        Return accumulated internal token counts (summarization + relation
        extraction) aggregated, and reset all counters to zero.

        Backward-compatible flat version. Prefer get_and_reset_token_counts_by_type()
        for structured per-call-type statistics.

        Returns:
            {"input": ..., "output": ..., "llm_calls": ...}
        """
        counts = {
            "input":     self._summarize_input  + self._relation_input,
            "output":    self._summarize_output + self._relation_output,
            "llm_calls": self._summarize_calls  + self._relation_calls,
        }
        self._summarize_input          = 0
        self._summarize_output         = 0
        self._summarize_calls          = 0
        self._summarize_fallback_count = 0
        self._relation_input           = 0
        self._relation_output          = 0
        self._relation_calls           = 0
        return counts
