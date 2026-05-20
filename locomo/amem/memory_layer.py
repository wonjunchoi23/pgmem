"""
Agentic Memory System (A-MEM) for LLM Agents — LoComo batch-enabled variant
"""

import json
import uuid
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    _heuristic_context,
    _heuristic_keywords,
    parse_analyze_content,
    validate_analysis_result,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EVOLUTION PROMPT & JSON SCHEMA
# =============================================================================

EVOLUTION_SYSTEM_PROMPT = (
    "You are an AI memory evolution agent responsible for managing and evolving "
    "a knowledge base. Analyze the new memory note according to keywords and context, "
    "also with their several nearest neighbors memory. Make decisions about its evolution.\n\n"
    "The new memory context:\n"
    "{context}\n"
    "content: {content}\n"
    "keywords: {keywords}\n\n"
    "The nearest neighbors memories:\n"
    "{nearest_neighbors_memories}\n\n"
    "Based on this information, determine:\n"
    "1. Should this memory be evolved? Consider its relationships with other memories.\n"
    "2. What specific actions should be taken (strengthen, update_neighbor)?\n"
    "   2.1 If choose to strengthen the connection, which memory should it be connected to? "
    "Can you give the updated tags of this memory?\n"
    "   2.2 If choose to update_neighbor, you can update the context and tags of these memories "
    "based on the understanding of these memories. If the context and the tags are not updated, "
    "the new context and tags should be the same as the original ones. Generate the new context "
    "and tags in the sequential order of the input neighbors.\n"
    "Tags should be determined by the content of these characteristic of these memories, "
    "which can be used to retrieve them later and categorize them.\n"
    "Note that the length of new_tags_neighborhood must equal the number of input neighbors, "
    "and the length of new_context_neighborhood must equal the number of input neighbors.\n"
    "The number of neighbors is {neighbor_number}.\n"
    "Return your decision in JSON format with the following structure:\n"
    "{{\n"
    '    "should_evolve": true or false,\n'
    '    "actions": ["strengthen", "update_neighbor"],\n'
    '    "suggested_connections": [neighbor_memory_indices],\n'
    '    "tags_to_update": ["tag_1", ..., "tag_n"],\n'
    '    "new_context_neighborhood": ["new context", ..., "new context"],\n'
    '    "new_tags_neighborhood": [["tag_1", ..., "tag_n"], ..., ["tag_1", ..., "tag_n"]]\n'
    "}}"
)

_EVOLUTION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "should_evolve": {"type": "boolean"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "suggested_connections": {"type": "array", "items": {"type": "integer"}},
                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                "new_tags_neighborhood": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "required": [
                "should_evolve",
                "actions",
                "suggested_connections",
                "tags_to_update",
                "new_context_neighborhood",
                "new_tags_neighborhood",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_EVOLUTION_GUIDED_JSON = _EVOLUTION_JSON_SCHEMA["json_schema"]["schema"]


# =============================================================================
# TIMESTAMP HELPERS
# =============================================================================

def parse_display_time(timestamp: str) -> str:
    if "|" in timestamp:
        return timestamp.split("|", 1)[1]
    return timestamp


def parse_dia_id(timestamp: str) -> str:
    if "|" in timestamp:
        return timestamp.split("|", 1)[0]
    return timestamp


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryNote:
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context: str = "General"
    links: List[int] = field(default_factory=list)
    importance_score: float = 1.0
    retrieval_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    evolution_history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryNote":
        return cls(**data)


# =============================================================================
# RETRIEVER
# =============================================================================

class SimpleEmbeddingRetriever:
    """
    Simple retrieval system using sentence embeddings and cosine similarity.
    """

    def __init__(self, model_name_or_instance: Union[str, SentenceTransformer] = "all-MiniLM-L6-v2"):
        if isinstance(model_name_or_instance, str):
            self.model = SentenceTransformer(model_name_or_instance)
            self.model_name = model_name_or_instance
        else:
            self.model = model_name_or_instance
            self.model_name = "shared"
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[str]):
        if not documents:
            return
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            return
        self.corpus.extend(documents)
        new_embeddings = self.model.encode(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, k: int = 5) -> List[int]:
        if not self.corpus or self.embeddings is None:
            return []
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()

    def search_with_scores(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        if not self.corpus or self.embeddings is None:
            return [], []
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_indices]
        return top_k_indices.tolist(), top_k_scores.tolist()

    def save(self, cache_file: str, embeddings_file: str):
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)
        state = {
            "model_name": self.model_name,
            "corpus": self.corpus,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    def load(self, cache_file: str, embeddings_file: str) -> "SimpleEmbeddingRetriever":
        embeddings_path = Path(embeddings_file)
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_file)
        cache_path = Path(cache_file)
        if cache_path.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.corpus = state["corpus"]
            self.model_name = state.get("model_name", "all-MiniLM-L6-v2")
        return self

    def clear(self):
        self.corpus = []
        self.embeddings = None


# =============================================================================
# LLM WRAPPER
# =============================================================================

class LLMWrapper:
    """
    Wrapper for LLM client with JSON retry logic and internal token tracking.
    """

    PLAIN_TEXT_SYSTEM = (
        "Follow the format specified in the prompt exactly. "
        "Do not add extra commentary."
    )
    JSON_SYSTEM = "You must respond with a valid JSON object."

    def __init__(self, llm_client, json_retry: int = 3):
        self.client = llm_client
        self.json_retry = json_retry
        # Per-call-type token counters
        self._note_input_tokens = 0
        self._note_output_tokens = 0
        self._note_llm_calls = 0
        self._evo_input_tokens = 0
        self._evo_output_tokens = 0
        self._evo_llm_calls = 0

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        guided_json = None
        if response_format and "json_schema" in response_format:
            guided_json = response_format["json_schema"].get("schema")

        try:
            result = self.client.generate(
                prompt=prompt,
                system_prompt=self.JSON_SYSTEM,
                guided_json=guided_json,
                temperature=temperature,
                max_tokens=max_tokens,
                json_retry=self.json_retry,
                validate_json=True,
                return_usage=True,
            )

            if isinstance(result, dict) and "_usage" in result:
                usage = result["_usage"]
                self._evo_input_tokens += usage.get("prompt_tokens", 0)
                self._evo_output_tokens += usage.get("completion_tokens", 0)
            self._evo_llm_calls += 1

            if isinstance(result, dict):
                result_clean = {k: v for k, v in result.items() if k != "_usage"}
                return json.dumps(result_clean)
            return result

        except Exception as e:
            logger.error(f"LLM JSON completion error: {e}")
            if guided_json:
                return json.dumps(self._generate_empty_response(guided_json))
            raise

    def get_plain_text_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        try:
            result = self.client.generate(
                prompt=prompt,
                system_prompt=self.PLAIN_TEXT_SYSTEM,
                guided_json=None,
                temperature=temperature,
                max_tokens=max_tokens,
                json_retry=0,
                validate_json=False,
                return_usage=True,
            )

            if isinstance(result, dict) and "_usage" in result:
                usage = result["_usage"]
                self._note_input_tokens += usage.get("prompt_tokens", 0)
                self._note_output_tokens += usage.get("completion_tokens", 0)
            self._note_llm_calls += 1

            if isinstance(result, dict):
                return result.get("content", result.get("text", ""))
            return str(result) if result is not None else ""

        except Exception as e:
            logger.error(f"LLM plain-text completion error: {e}")
            return ""

    def accumulate_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_calls: int = 1,
        call_type: str = "call_1_note_construction",
    ):
        """Accumulate token counts from an external (batch) LLM call."""
        if call_type == "call_2_evolution":
            self._evo_input_tokens += input_tokens
            self._evo_output_tokens += output_tokens
            self._evo_llm_calls += llm_calls
        else:
            self._note_input_tokens += input_tokens
            self._note_output_tokens += output_tokens
            self._note_llm_calls += llm_calls

    def get_and_reset_token_counts(self) -> Dict[str, int]:
        """Aggregate totals across all call types (backward compat)."""
        counts = {
            "input": self._note_input_tokens + self._evo_input_tokens,
            "output": self._note_output_tokens + self._evo_output_tokens,
            "llm_calls": self._note_llm_calls + self._evo_llm_calls,
        }
        self._note_input_tokens = 0
        self._note_output_tokens = 0
        self._note_llm_calls = 0
        self._evo_input_tokens = 0
        self._evo_output_tokens = 0
        self._evo_llm_calls = 0
        return counts

    def get_and_reset_token_counts_by_type(self) -> Dict[str, Dict]:
        """Return per-call-type token counts and reset all counters."""
        counts = {
            "call_1_note_construction": {
                "input": self._note_input_tokens,
                "output": self._note_output_tokens,
                "llm_calls": self._note_llm_calls,
            },
            "call_2_evolution": {
                "input": self._evo_input_tokens,
                "output": self._evo_output_tokens,
                "llm_calls": self._evo_llm_calls,
            },
        }
        self._note_input_tokens = 0
        self._note_output_tokens = 0
        self._note_llm_calls = 0
        self._evo_input_tokens = 0
        self._evo_output_tokens = 0
        self._evo_llm_calls = 0
        return counts

    def _generate_empty_response(self, schema: dict) -> dict:
        result = {}
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            if prop_type == "array":
                result[prop_name] = []
            elif prop_type == "string":
                result[prop_name] = ""
            elif prop_type == "boolean":
                result[prop_name] = False
            elif prop_type in ["number", "integer"]:
                result[prop_name] = 0
            elif prop_type == "object":
                result[prop_name] = {}
            else:
                result[prop_name] = None
        return result


# =============================================================================
# AGENTIC MEMORY SYSTEM
# =============================================================================

class AgenticMemorySystem:
    """
    Main memory management system with embedding-based retrieval.
    """

    def __init__(
        self,
        llm_client,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_model: Optional[SentenceTransformer] = None,
        evo_threshold: int = 100,
        json_retry: int = 3,
    ):
        self.memories: Dict[str, MemoryNote] = {}
        self.retriever = (
            SimpleEmbeddingRetriever(embedding_model)
            if embedding_model is not None
            else SimpleEmbeddingRetriever(model_name)
        )
        self.llm = LLMWrapper(llm_client, json_retry)
        self.model_name = model_name
        self.evo_threshold = evo_threshold
        self.evo_cnt = 0
        self.llm_logger = None  # type: Optional[Any]
        # internal stats counters (reset by get_and_reset_internal_stats)
        self._evo_triggered = 0
        self._evo_actions: Dict[str, int] = {"strengthen": 0, "update_neighbor": 0}
        self._note_parse_fallback_count = 0

    # -------------------------------------------------------------------------
    # Public sequential API
    # -------------------------------------------------------------------------

    def add_note(self, content: str, time: Optional[str] = None) -> str:
        analyze_prompt = self.build_analyze_prompt(content)
        analysis_text = self.llm.get_plain_text_completion(analyze_prompt)
        if self.llm_logger is not None:
            self.llm_logger.log("call_2_note_construction", "", analyze_prompt, analysis_text)

        note, evolve_prompt, evolve_ctx = self.apply_analyze_result(content, analysis_text, time)
        if evolve_prompt is None:
            self.store_note(note)
            return note.id

        try:
            evolve_response = self.llm.get_completion(
                evolve_prompt,
                response_format=_EVOLUTION_JSON_SCHEMA,
            )
            if self.llm_logger is not None:
                self.llm_logger.log("call_3_evolution", "", evolve_prompt, evolve_response)
            response_json = (
                json.loads(evolve_response) if isinstance(evolve_response, str) else evolve_response
            )
        except json.JSONDecodeError as e:
            logger.error(f"Evolution JSON parse error (storing without evolution): {e}")
            self.store_note(note)
            return note.id
        except Exception as e:
            logger.error(f"Evolution failed (storing without evolution): {e}")
            self.store_note(note)
            return note.id

        self.apply_evolve_result(note, response_json, evolve_ctx)
        return note.id

    # -------------------------------------------------------------------------
    # Step-wise API
    # -------------------------------------------------------------------------

    def build_analyze_prompt(self, content: str) -> str:
        return ANALYZE_CONTENT_PROMPT.format(content=content)

    def apply_analyze_result(
        self,
        content: str,
        analysis_text: str,
        time: Optional[str] = None,
    ) -> Tuple["MemoryNote", Optional[str], Optional[tuple]]:
        try:
            analysis = parse_analyze_content(analysis_text, content)
            analysis = validate_analysis_result(analysis, content)
        except Exception as e:
            logger.error(f"Error parsing analysis result: {e}")
            self._note_parse_fallback_count += 1
            analysis = {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }

        note = MemoryNote(
            content=content,
            timestamp=time or datetime.now().isoformat(),
            keywords=analysis.get("keywords", []),
            tags=analysis.get("tags", []),
            context=analysis.get("context", "General"),
        )
        if isinstance(note.context, list):
            note.context = " ".join(note.context)

        if not self.memories:
            return note, None, None

        neighbor_memory, indices = self._find_neighbors(note.content, k=5)
        if not indices:
            return note, None, None

        evolve_prompt = EVOLUTION_SYSTEM_PROMPT.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices),
        )
        evolve_ctx = (indices, list(self.memories.values()), list(self.memories.keys()))
        return note, evolve_prompt, evolve_ctx

    def apply_evolve_result(
        self,
        note: "MemoryNote",
        response_json: dict,
        evolve_ctx: tuple,
    ) -> bool:
        indices, all_notes, note_ids = evolve_ctx

        evolved = False
        if response_json.get("should_evolve"):
            evolved = True
            self._evo_triggered += 1
            for action in response_json.get("actions", []):
                if action == "strengthen":
                    self._evo_actions["strengthen"] += 1
                    note.links.extend(response_json.get("suggested_connections", []))
                    new_tags = response_json.get("tags_to_update", [])
                    if new_tags:
                        note.tags = new_tags
                elif action == "update_neighbor":
                    self._evo_actions["update_neighbor"] += 1
                    new_contexts = response_json.get("new_context_neighborhood", [])
                    new_tags_nbr = response_json.get("new_tags_neighborhood", [])
                    for i in range(min(len(indices), len(new_tags_nbr))):
                        memory_idx = indices[i]
                        if memory_idx >= len(all_notes):
                            continue
                        note_tmp = all_notes[memory_idx]
                        note_tmp.tags = new_tags_nbr[i]
                        if i < len(new_contexts):
                            note_tmp.context = new_contexts[i]
                        self.memories[note_ids[memory_idx]] = note_tmp

        self.store_note(note)

        if evolved:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self._consolidate_memories()
        return evolved

    def store_note(self, note: "MemoryNote") -> None:
        self.memories[note.id] = note
        self.retriever.add_documents([self._note_to_document(note)])

    # -------------------------------------------------------------------------
    # Retrieval API
    # -------------------------------------------------------------------------

    def find_related_memories(self, query: str, k: int = 5) -> str:
        """Return top-k memories without following linked neighbors."""
        if not self.memories:
            return ""
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            if i < len(all_memories):
                m = all_memories[i]
                dtime = parse_display_time(m.timestamp)
                memory_str += (
                    f"memory index:{i}\t"
                    f"talk start time:{dtime}\t"
                    f"memory content: {m.content}\t"
                    f"memory context: {m.context}\t"
                    f"memory keywords: {m.keywords}\t"
                    f"memory tags: {m.tags}\n"
                )
        return memory_str

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        memory_str, _ = self._retrieve_with_links(query, k)
        return memory_str

    def find_related_memories_with_metadata(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        return self._retrieve_with_links(query, k)

    def _retrieve_with_links(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        if not self.memories:
            return "", []
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        metadata_list = []
        for i in indices:
            if i < len(all_memories):
                m = all_memories[i]
                memory_str += self._format_note_str(m)
                metadata_list.append({
                    "dia_id": parse_dia_id(m.timestamp),
                    "content_preview": m.content[:120],
                })
                j = 0
                for neighbor_idx in m.links:
                    if neighbor_idx < len(all_memories):
                        n = all_memories[neighbor_idx]
                        memory_str += self._format_note_str(n)
                        metadata_list.append({
                            "dia_id": parse_dia_id(n.timestamp),
                            "content_preview": n.content[:120],
                        })
                        if j >= k:
                            break
                        j += 1
        return memory_str, metadata_list

    def retrieve_for_log(self, query: str, k: int = 5) -> Tuple[List[Dict], int]:
        if not self.memories:
            return [], 0

        indices, scores = self.retriever.search_with_scores(query, k)
        all_memories = list(self.memories.values())
        items = []
        num_linked = 0
        for i, score in zip(indices, scores):
            if i < len(all_memories):
                m = all_memories[i]
                items.append({
                    "memory_id": m.id,
                    "content_preview": m.content[:120],
                    "score": float(score),
                    "source_turn": {"dia_id": parse_dia_id(m.timestamp)},
                })
                num_linked += len([idx for idx in m.links if idx < len(all_memories)])
        return items, num_linked

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def clear(self):
        self.memories = {}
        self.retriever.clear()
        self.evo_cnt = 0
        self._evo_triggered = 0
        self._evo_actions = {"strengthen": 0, "update_neighbor": 0}
        self._note_parse_fallback_count = 0
        logger.info("Memory system cleared")

    def save_snapshot(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        memories_data = {k: v.to_dict() for k, v in self.memories.items()}
        with open(directory / "memories.json", "w", encoding="utf-8") as f:
            json.dump(memories_data, f, ensure_ascii=False, indent=2)
        self.retriever.save(
            str(directory / "retriever.json"),
            str(directory / "embeddings.npy"),
        )
        metadata = {
            "num_memories": len(self.memories),
            "evo_cnt": self.evo_cnt,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_snapshot(self, directory: Path):
        directory = Path(directory)
        memories_file = directory / "memories.json"
        if memories_file.exists():
            with open(memories_file, "r", encoding="utf-8") as f:
                memories_data = json.load(f)
            self.memories = {k: MemoryNote.from_dict(v) for k, v in memories_data.items()}
        retriever_file = directory / "retriever.json"
        embeddings_file = directory / "embeddings.npy"
        if retriever_file.exists():
            self.retriever.load(str(retriever_file), str(embeddings_file))
        metadata_file = directory / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            self.evo_cnt = metadata.get("evo_cnt", 0)

    def get_memory_count(self) -> int:
        return len(self.memories)

    def get_and_reset_token_counts_by_type(self) -> Dict[str, Dict]:
        return self.llm.get_and_reset_token_counts_by_type()

    def get_memory_stats(self) -> Dict[str, int]:
        num_memories = len(self.memories)
        total_chars = sum(len(note.content) for note in self.memories.values())
        return {
            "num_memories": num_memories,
            "total_content_tokens": total_chars // 4,
        }

    def get_and_reset_internal_stats(self) -> Dict:
        stats = {
            "evo_triggered_count": self._evo_triggered,
            "actions_taken": dict(self._evo_actions),
            "note_parse_fallback_count": self._note_parse_fallback_count,
        }
        self._evo_triggered = 0
        self._evo_actions = {"strengthen": 0, "update_neighbor": 0}
        self._note_parse_fallback_count = 0
        return stats

    def get_and_reset_token_counts(self) -> Dict[str, int]:
        return self.llm.get_and_reset_token_counts()

    def accumulate_token_counts(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_calls: int = 1,
        call_type: str = "call_1_note_construction",
    ):
        self.llm.accumulate_usage(input_tokens, output_tokens, llm_calls, call_type)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_note_str(note: "MemoryNote") -> str:
        dtime = parse_display_time(note.timestamp)
        return (
            f"talk start time:{dtime} "
            f"memory content: {note.content} "
            f"memory context: {note.context} "
            f"memory keywords: {note.keywords} "
            f"memory tags: {note.tags}\n"
        )

    def _find_neighbors(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        if not self.memories:
            return "", []
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            if i < len(all_memories):
                m = all_memories[i]
                dtime = parse_display_time(m.timestamp)
                memory_str += (
                    f"memory index:{i}\t"
                    f"talk start time:{dtime}\t"
                    f"memory content: {m.content}\t"
                    f"memory context: {m.context}\t"
                    f"memory keywords: {m.keywords}\t"
                    f"memory tags: {m.tags}\n"
                )
        return memory_str, indices

    def _consolidate_memories(self):
        self.retriever.clear()
        docs = [self._note_to_document(memory) for memory in self.memories.values()]
        if docs:
            self.retriever.add_documents(docs)
        logger.info(f"Consolidated {len(self.memories)} memories")

    def _note_to_document(self, note: MemoryNote) -> str:
        return (
            f"content:{note.content} "
            f"context:{note.context} "
            f"keywords: {', '.join(note.keywords)} "
            f"tags: {', '.join(note.tags)}"
        )
