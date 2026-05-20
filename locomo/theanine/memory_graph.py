"""
Memory Graph for Theanine — LoComo batch-enabled variant
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

import config as cfg

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

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
# JSON SCHEMAS
# =============================================================================

SUMMARIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 15,
        }
    },
    "required": ["sentences"],
}

RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "relation": {
            "type": "string",
            "enum": ["Changed", "Cause", "Reason", "HinderedBy", "React", "Want", "SameTopic", "None"],
        },
    },
    "required": ["explanation", "relation"],
}

SUMMARIZATION_GUIDED_JSON = SUMMARIZATION_SCHEMA
RELATION_GUIDED_JSON = RELATION_SCHEMA


# =============================================================================
# HELPERS
# =============================================================================

def _extract_token_info(result, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(result, dict) and "_usage" in result:
        usage = result["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def _extract_text(result) -> str:
    if isinstance(result, dict):
        return result.get("content", result.get("text", ""))
    return str(result) if result else ""


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class MemoryNode:
    node_id: str
    summary: str
    finalize_idx: int
    sample_id: str
    session_ids: List[int]
    dia_ids: List[str]
    date: str
    source_session_dialogue: str
    embedding: Optional[np.ndarray] = None
    links: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# MEMORY GRAPH
# =============================================================================

class MemoryGraph:
    def __init__(self, llm_client, config, embedding_model: Optional[Union[str, SentenceTransformer]] = None):
        self.nodes: Dict[str, MemoryNode] = {}
        self.llm_client = llm_client
        self.encoder = (
            SentenceTransformer(config.EMBEDDING_MODEL)
            if embedding_model is None or isinstance(embedding_model, str)
            else embedding_model
        )
        self._llm_logger = None
        self._summ_tokens = {"input": 0, "output": 0, "calls": 0, "fallback": 0}
        self._rel_tokens = {"input": 0, "output": 0, "calls": 0}

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _process_text(self, result: str) -> List[str]:
        pattern = re.compile(r"[0-9]+\.\s")
        sentences = []
        for line in result.strip().split("\n"):
            refined = pattern.sub("", line).strip()
            if refined:
                sentences.append(refined)
        return sentences

    def build_summarize_prompt(self, dialogue: str, plain_text: bool = False) -> str:
        if plain_text:
            return SUMMARIZATION_PROMPT_PLAINTEXT.format(dialogue=dialogue)
        return SUMMARIZATION_PROMPT.format(dialogue=dialogue)

    def _summarize_conv(self, dialogue: str) -> Tuple[List[str], Dict]:
        prompt = self.build_summarize_prompt(dialogue)
        sentences = []
        token_info = {"input": 0, "output": 0}
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                guided_json=SUMMARIZATION_SCHEMA,
                return_usage=True,
            )
            token_info = _extract_token_info(result)
            sentences = result.get("sentences", []) if isinstance(result, dict) else []
            if self._llm_logger is not None:
                self._llm_logger.log("call_2_summarization", "", prompt, result)
        except Exception as json_err:
            logger.warning(f"_summarize_conv: JSON mode failed after all retries ({json_err}). Falling back to plain-text summarization.")
            self._summ_tokens["fallback"] += 1
            fallback_prompt = self.build_summarize_prompt(dialogue, plain_text=True)
            try:
                fb_result = self.llm_client.generate(
                    prompt=fallback_prompt,
                    temperature=cfg.TEMPERATURE,
                    max_tokens=cfg.SUMMARIZE_MAX_TOKENS,
                    return_usage=True,
                )
                token_info = _extract_token_info(fb_result)
                text = _extract_text(fb_result)
                sentences = self._process_text(text)
                if self._llm_logger is not None:
                    self._llm_logger.log("call_2_summarization", "", fallback_prompt, fb_result)
            except Exception as fb_err:
                logger.error(f"_summarize_conv: plain-text fallback also failed ({fb_err}). Returning empty sentence list.")
                token_info = {"input": 0, "output": 0}

        self._summ_tokens["input"] += token_info["input"]
        self._summ_tokens["output"] += token_info["output"]
        self._summ_tokens["calls"] += 1
        return sentences, token_info

    def apply_summarize_result(
        self,
        summarize_result,
        finalize_idx: int,
        sample_id: str,
        session_ids: List[int],
        dia_ids: List[str],
        date: str,
        source_session_dialogue: str,
    ) -> List["MemoryNode"]:
        if isinstance(summarize_result, dict):
            sentences = summarize_result.get("sentences", [])
        elif isinstance(summarize_result, list):
            sentences = summarize_result
        else:
            sentences = []
        return self._create_nodes_from_summary(
            sentences,
            finalize_idx,
            sample_id,
            session_ids,
            dia_ids,
            date,
            source_session_dialogue,
        )

    def _create_nodes_from_summary(
        self,
        sentences: List[str],
        finalize_idx: int,
        sample_id: str,
        session_ids: List[int],
        dia_ids: List[str],
        date: str,
        source_session_dialogue: str,
    ) -> List["MemoryNode"]:
        nodes = []
        for idx, sentence in enumerate(sentences):
            node_id = f"f{finalize_idx}-m{idx + 1}"
            nodes.append(MemoryNode(
                node_id=node_id,
                summary=sentence,
                finalize_idx=finalize_idx,
                sample_id=sample_id,
                session_ids=list(session_ids),
                dia_ids=list(dia_ids),
                date=date,
                source_session_dialogue=source_session_dialogue,
            ))
        return nodes

    # ------------------------------------------------------------------
    # Embedding & Retrieval
    # ------------------------------------------------------------------

    def cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(dot(a, b) / (norm(a) * norm(b)))

    def embed_nodes(self, nodes: List[MemoryNode]):
        if not nodes:
            return
        texts = [node.summary for node in nodes]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        for node, emb in zip(nodes, embeddings):
            node.embedding = emb

    def retrieve(self, query: str, k: int) -> List[Tuple[MemoryNode, float]]:
        if not self.nodes:
            return []
        query_emb = self.encoder.encode([query], convert_to_numpy=True)[0]
        scores = []
        for node in self.nodes.values():
            if node.embedding is not None:
                scores.append((node, self.cos_sim(query_emb, node.embedding)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _find_associative(self, target_node: MemoryNode, j: int) -> List[MemoryNode]:
        past_nodes = [
            n for n in self.nodes.values()
            if n.finalize_idx < target_node.finalize_idx and n.embedding is not None
        ]
        if not past_nodes or target_node.embedding is None:
            return []

        scores = [(node, self.cos_sim(target_node.embedding, node.embedding)) for node in past_nodes]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scores[:j]]

    # ------------------------------------------------------------------
    # Relation Extraction & Linking
    # ------------------------------------------------------------------

    def build_relation_prompt(
        self,
        sentence1: str,
        sentence2: str,
        dialogue1: str,
        dialogue2: str,
    ) -> str:
        return RELATION_EXTRACTION_PROMPT.format(
            dialogue1=dialogue1,
            dialogue2=dialogue2,
            sentence1=sentence1,
            sentence2=sentence2,
        )

    def _extract_relation(
        self,
        sentence1: str,
        sentence2: str,
        dialogue1: str,
        dialogue2: str,
    ) -> Tuple[str, Dict]:
        prompt = self.build_relation_prompt(sentence1, sentence2, dialogue1, dialogue2)
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_TOKENS,
                guided_json=RELATION_SCHEMA,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
            token_info = _extract_token_info(result)
            relation = result.get("relation", "None") if isinstance(result, dict) else "None"
            if self._llm_logger is not None:
                self._llm_logger.log("call_3_relation", "", prompt, result)
        except Exception as e:
            logger.warning(f"_extract_relation: generation failed ({type(e).__name__}: {e}), returning 'None'")
            token_info = {"input": 0, "output": 0}
            relation = "None"

        self._rel_tokens["input"] += token_info["input"]
        self._rel_tokens["output"] += token_info["output"]
        self._rel_tokens["calls"] += 1
        return relation, token_info

    def build_relation_jobs(self, new_nodes: List[MemoryNode]) -> List[Dict]:
        jobs = []
        for new_node in new_nodes:
            associative = self._find_associative(new_node, cfg.LINKING_TOP_J)
            for past_node in sorted(associative, key=lambda n: self._node_recency_key(n), reverse=True):
                jobs.append({
                    "new_node_id": new_node.node_id,
                    "past_node_id": past_node.node_id,
                    "prompt": self.build_relation_prompt(
                        sentence1=past_node.summary,
                        sentence2=new_node.summary,
                        dialogue1=past_node.source_session_dialogue,
                        dialogue2=new_node.source_session_dialogue,
                    ),
                    "past_node": past_node,
                    "new_node": new_node,
                })
        return jobs

    def apply_relation_results(self, new_nodes: List[MemoryNode], relation_results: List[Dict]) -> Dict:
        result_by_new_node: Dict[str, List[Tuple[MemoryNode, str]]] = {}
        for item in relation_results:
            relation = item.get("relation", "None")
            if "None" in relation:
                continue
            result_by_new_node.setdefault(item["new_node_id"], []).append((item["past_node"], relation))

        linked_edges = 0
        for new_node in new_nodes:
            related_candidates = result_by_new_node.get(new_node.node_id, [])
            if not related_candidates:
                continue
            candidate_nodes = [node for node, _ in related_candidates]
            representatives = self._select_representatives_by_component(candidate_nodes)
            relation_map = {node.node_id: relation for node, relation in related_candidates}
            for past_node in representatives:
                relation = relation_map[past_node.node_id]
                new_node.links[past_node.node_id] = relation
                past_node.links[new_node.node_id] = relation
                linked_edges += 1
        return {"linked_edges": linked_edges}

    def register_new_nodes(self, new_nodes: List[MemoryNode]) -> None:
        for node in new_nodes:
            self.nodes[node.node_id] = node

    def _node_recency_key(self, node: MemoryNode) -> Tuple[int, int]:
        try:
            mem_idx = int(node.node_id.split("-m")[1])
        except Exception:
            mem_idx = 0
        return (node.finalize_idx, mem_idx)

    def _get_connected_component(self, start_node_id: str) -> List[str]:
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

    def _select_representatives_by_component(self, candidate_nodes: List[MemoryNode]) -> List[MemoryNode]:
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
        reps = list(component_to_best.values())
        reps.sort(key=self._node_recency_key, reverse=True)
        return reps

    # ------------------------------------------------------------------
    # Main Interface
    # ------------------------------------------------------------------

    def finalize_conv(
        self,
        finalize_idx: int,
        sample_id: str,
        full_dialogue: str,
        dia_ids: List[str],
        session_ids: List[int],
        date: str,
    ) -> Dict:
        sentences, summ_tokens = self._summarize_conv(full_dialogue)
        if not sentences:
            logger.warning(f"finalize_conv: empty summary for finalize_idx={finalize_idx}, sample_id={sample_id}")
            return {"input": summ_tokens["input"], "output": summ_tokens["output"], "llm_calls": 1}

        new_nodes = self._create_nodes_from_summary(
            sentences, finalize_idx, sample_id, session_ids, dia_ids, date, full_dialogue
        )
        self.embed_nodes(new_nodes)

        total_relation_input = 0
        total_relation_output = 0
        relation_calls = 0
        linked_edges = 0
        for new_node in new_nodes:
            associative = self._find_associative(new_node, cfg.LINKING_TOP_J)
            for past_node in associative:
                relation, token_info = self._extract_relation(
                    sentence1=past_node.summary,
                    sentence2=new_node.summary,
                    dialogue1=past_node.source_session_dialogue,
                    dialogue2=new_node.source_session_dialogue,
                )
                total_relation_input += token_info["input"]
                total_relation_output += token_info["output"]
                relation_calls += 1
                if "None" not in relation:
                    new_node.links[past_node.node_id] = relation
                    past_node.links[new_node.node_id] = relation
                    linked_edges += 1

        self.register_new_nodes(new_nodes)
        total_llm_calls = 1 + relation_calls
        logger.info(
            f"finalize_conv: finalize_idx={finalize_idx}, sample_id={sample_id}, "
            f"sessions={session_ids}, nodes_created={len(new_nodes)}, "
            f"llm_calls={total_llm_calls}, linked_edges={linked_edges}"
        )
        return {
            "input": summ_tokens["input"] + total_relation_input,
            "output": summ_tokens["output"] + total_relation_output,
            "llm_calls": total_llm_calls,
        }

    def get_node_count(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self):
        self.nodes = {}
        self._summ_tokens = {"input": 0, "output": 0, "calls": 0, "fallback": 0}
        self._rel_tokens = {"input": 0, "output": 0, "calls": 0}

    def save_snapshot(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        nodes_data = {
            node_id: {
                "node_id": node.node_id,
                "summary": node.summary,
                "finalize_idx": node.finalize_idx,
                "sample_id": node.sample_id,
                "session_ids": node.session_ids,
                "dia_ids": node.dia_ids,
                "date": node.date,
                "source_session_dialogue": node.source_session_dialogue,
                "links": node.links,
            }
            for node_id, node in self.nodes.items()
        }
        with open(directory / "nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, indent=2, ensure_ascii=False)

        nodes_with_emb = [(nid, node) for nid, node in self.nodes.items() if node.embedding is not None]
        if nodes_with_emb:
            emb_ids = [nid for nid, _ in nodes_with_emb]
            emb_matrix = np.stack([node.embedding for _, node in nodes_with_emb])
            np.save(directory / "embeddings.npy", emb_matrix)
            with open(directory / "embedding_ids.json", "w") as f:
                json.dump(emb_ids, f)

    def load_snapshot(self, directory: Path):
        directory = Path(directory)
        nodes_file = directory / "nodes.json"
        emb_file = directory / "embeddings.npy"
        ids_file = directory / "embedding_ids.json"

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
                finalize_idx=data["finalize_idx"],
                sample_id=data["sample_id"],
                session_ids=data.get("session_ids", []),
                dia_ids=data.get("dia_ids", []),
                date=data.get("date", ""),
                source_session_dialogue=data.get("source_session_dialogue", ""),
                links=data.get("links", {}),
                embedding=emb_map.get(node_id),
            )
            self.nodes[node_id] = node

    def get_and_reset_token_counts_by_type(self) -> Dict:
        counts = {
            "call_2_summarization": {
                "input": self._summ_tokens["input"],
                "output": self._summ_tokens["output"],
                "llm_calls": self._summ_tokens["calls"],
                "parse_fallback_count": self._summ_tokens["fallback"],
            },
            "call_3_relation": {
                "input": self._rel_tokens["input"],
                "output": self._rel_tokens["output"],
                "llm_calls": self._rel_tokens["calls"],
            },
        }
        self._summ_tokens = {"input": 0, "output": 0, "calls": 0, "fallback": 0}
        self._rel_tokens = {"input": 0, "output": 0, "calls": 0}
        return counts

    def get_memory_stats(self) -> Dict:
        num_memories = len(self.nodes)
        total_chars = sum(len(node.summary) for node in self.nodes.values())
        return {
            "num_memories": num_memories,
            "total_content_tokens": total_chars // 4,
        }

    def get_and_reset_summarize_fallback_count(self) -> int:
        count = self._summ_tokens["fallback"]
        self._summ_tokens["fallback"] = 0
        return count
