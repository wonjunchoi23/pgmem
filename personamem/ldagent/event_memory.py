"""
Event Memory Module for LD-Agent — PersonaMem variant

Internal field names (`session_id`, `conv_id`, `turn_id`) are retained from
the ImplexConv variant. The runner injects PersonaMem identifiers into those
slots: context_index → session_id, block_idx → conv_id, local_msg_idx →
turn_id. Result metadata is remapped in run_experiment.py.

LTM storage uses numpy + SentenceTransformer (no ChromaDB).
STM finalization policy unchanged: STM accumulates within
FINALIZE_EVERY_N_CONVS conv_ids (= blocks in PersonaMem) and is summarised +
cleared at every group boundary.

Reference: "Hello Again! LLM-powered Personalized Agent for Long-term Dialogue"
           (Li et al., NAACL 2025)
"""

import math
import json
import logging
import numpy as np
import spacy
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_TIKTOKEN_ENC.encode(text))
except Exception:
    def _count_tokens(text: str) -> int:
        return len(text) // 4

def _truncate_oldest_first(text: str, max_tokens: int, log: logging.Logger = None) -> str:
    """Drop leading newline-separated lines until token count ≤ max_tokens."""
    lines = text.split("\n")
    original = len(lines)
    while len(lines) > 1 and _count_tokens("\n".join(lines)) > max_tokens:
        lines.pop(0)
    if len(lines) < original and log is not None:
        log.warning(
            "STM context truncated from %d to %d lines to fit %d-token budget",
            original, len(lines), max_tokens,
        )
    return "\n".join(lines)

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}
SUMMARY_GUIDED_JSON = SUMMARY_SCHEMA

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_DIM   = 384


# =============================================================================
# METADATA CLASS
# =============================================================================

@dataclass
class MetaData:
    idx:             int   = 0
    dialog:          str   = ""
    virtual_seconds: float = 0.0   # virtual elapsed seconds
    topics:          str   = ""
    datatype:        str   = "text"
    summary:         str   = ""
    conv_id:         int   = 0
    session_id:      int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx":             self.idx,
            "dialog":          self.dialog,
            "virtual_seconds": self.virtual_seconds,
            "topics":          self.topics,
            "datatype":        self.datatype,
            "summary":         self.summary,
            "conv_id":         self.conv_id,
            "session_id":      self.session_id,
        }


# =============================================================================
# EVENT MEMORY CLASS
# =============================================================================

class EventMemory:
    """
    Long-term and short-term memory module for LD-Agent.

    LTM storage uses numpy + SentenceTransformer (replaces ChromaDB):
      - Embeddings stored as list of (dim,) float32 arrays in memory.
      - Retrieval: L2 distance on normalized sentence-transformer embeddings,
        then noun-overlap × time-decay scoring (identical logic to ChromaDB version).
      - Snapshot: ltm_embeddings.npy  +  long_term_memory.json  +  memory_state.json.

    STM finalization policy (unchanged from v2):
      - STM accumulates turns within a group of FINALIZE_EVERY_N_CONVS conv_ids.
      - When the group is complete, STM is summarized → LTM and fully cleared.
      - flush_stm() handles the final incomplete group at end of Phase 1.
    """

    def __init__(
        self,
        llm_client,
        sample_id:               str,
        logger:                  logging.Logger,
        usr_name:                str   = "User",
        agent_name:              str   = "Agent",
        relevance_memory_number: int   = 1,
        dist_threshold:          float = 1.5,
        decay_temp:              float = 1e-4,
        ori_mem_query:           bool  = False,
        finalize_every_n_convs:        int   = 2,
        memory_cache_path:             Optional[str] = None,  # kept for compat; unused
        lemma_tokenizer=None,
        encoder=None,
        finalize_input_context_limit:  int   = 131_072,
        finalize_context_utilization:  float = 0.85,
    ):
        self.llm_client  = llm_client
        self.sample_id   = sample_id
        self.logger      = logger
        self._finalize_input_context_limit = finalize_input_context_limit
        self._finalize_context_utilization = finalize_context_utilization

        self.usr_name                = usr_name
        self.agent_name              = agent_name
        self.relevance_memory_number = relevance_memory_number
        self.dist_threshold          = dist_threshold
        self.decay_temp              = decay_temp
        self.ori_mem_query           = ori_mem_query
        self.finalize_every_n_convs  = finalize_every_n_convs

        # LLMCallLogger — injected externally; None = no logging
        self.llm_logger = None

        # Token info for the most recent _context_summarize call (zeros if no call)
        self.last_summarize_token_info: Dict[str, int] = {"input": 0, "output": 0}

        # Lemma tokenizer for noun extraction
        if lemma_tokenizer is not None:
            self.lemma_tokenizer = lemma_tokenizer
        else:
            try:
                self.lemma_tokenizer = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.lemma_tokenizer = spacy.load("en_core_web_sm")

        # Sentence encoder for LTM embedding-based retrieval
        if encoder is not None:
            self._encoder = encoder
        else:
            logger.info(f"Loading SentenceTransformer encoder ({_EMBEDDING_MODEL})...")
            self._encoder = SentenceTransformer(_EMBEDDING_MODEL)

        # LTM storage (replaces ChromaDB)
        self._ltm_embeddings: List[np.ndarray] = []   # (dim,) float32 per entry
        self._ltm_metadata:   List[Dict]       = []   # metadata dicts
        self._ltm_documents:  List[str]        = []   # noun strings (for re-encoding)

        # STM and virtual time state
        self.short_term_memory: List[Dict[str, Any]] = []
        self.current_virtual_seconds  = 0.0
        self.last_conv_id             = -1
        self._pending_conv_count      = 0
        self.overall_retrieve_score   = 0.0
        self.overall_retrieve_count   = 0

        logger.info(f"EventMemory initialized for sample {sample_id} "
                    f"(finalize_every={finalize_every_n_convs} convs)")

    # =========================================================================
    # LLM LOGGER INJECTION
    # =========================================================================

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

    def set_last_summarize_usage(self, input_tokens: int, output_tokens: int):
        self.last_summarize_token_info = {
            "input": input_tokens,
            "output": output_tokens,
        }

    # =========================================================================
    # EMBEDDING HELPERS
    # =========================================================================

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode text(s) into normalized (unit-norm) embedding vectors."""
        return self._encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def _ltm_add(self, document: str, metadata: Dict):
        """Append one entry to LTM storage."""
        emb = self._encode([document])[0]
        self._ltm_embeddings.append(emb)
        self._ltm_metadata.append(metadata)
        self._ltm_documents.append(document)

    def _ltm_query(self, query_texts: List[str], n_results: int) -> Dict:
        """
        Retrieve top-n LTM entries by L2 distance to the query embedding.

        Returns a dict matching the former ChromaDB query() output:
          {"metadatas": [[...]], "distances": [[...]]}
        Distances are Euclidean (L2) on normalized vectors, matching ChromaDB's
        default 'l2' metric with normalized embeddings.
        """
        if not self._ltm_metadata:
            return {"metadatas": [[]], "distances": [[]]}

        query_embs = self._encode(query_texts)           # (n_q, dim)
        query_emb  = query_embs.mean(axis=0)             # (dim,)

        emb_matrix = np.stack(self._ltm_embeddings)      # (n, dim)
        diffs      = emb_matrix - query_emb[np.newaxis, :]
        distances  = np.sqrt(np.sum(diffs ** 2, axis=1)) # (n,)

        n        = min(n_results, len(distances))
        top_idxs = np.argsort(distances)[:n]

        return {
            "metadatas": [[self._ltm_metadata[i] for i in top_idxs]],
            "distances": [[float(distances[i])   for i in top_idxs]],
        }

    # =========================================================================
    # STORAGE METHODS
    # =========================================================================

    def store(self, ids, key, metadata):
        """Add one LTM entry. `key` is the noun string; `ids` ignored (numpy index)."""
        if isinstance(key, list):
            key = key[0]
        if isinstance(metadata, list):
            metadata = metadata[0]
        self._ltm_add(key, metadata)

    # =========================================================================
    # RETRIEVAL METHODS
    # =========================================================================

    def relevance_retrieve(
        self,
        ori_query:               str,
        n_results:               int   = 10,
        dist_thres:              float = None,
        current_virtual_seconds: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on topic overlap and virtual-time decay.
        Scoring: overlap_score × exp(-decay_temp × elapsed_virtual_seconds)
        """
        dist_thres = dist_thres if dist_thres is not None else self.dist_threshold
        if current_virtual_seconds is None:
            current_virtual_seconds = self.current_virtual_seconds

        if not isinstance(ori_query, list):
            ori_query = [ori_query]

        query            = []
        query_nouns_item = []
        for query_item in ori_query:
            tokenized_item   = self.lemma_tokenizer(query_item)
            nouns            = list(set([
                token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"
            ]))
            query_nouns_item = nouns
            merged_nouns_str = ",".join(nouns)
            query.append(merged_nouns_str)

        if self.ori_mem_query:
            query = ori_query

        if not self._ltm_metadata:
            return []

        n_query = min(n_results, len(self._ltm_metadata))
        results = self._ltm_query(query, n_query)

        metadata_list = []
        best_memory   = {
            "idx": 0, "overall_score": 0.0,
            "overlap_score": 0.0, "overlap_count": 0, "distance": dist_thres,
        }
        empty_flag = False

        if results["metadatas"] and results["metadatas"][0]:
            for idx, retrieved_item in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][idx]

                retrieved_nouns_item = retrieved_item.get("topics", "").split(",")
                overlap_count        = len(set(query_nouns_item) & set(retrieved_nouns_item))

                if len(query_nouns_item) == 0 or len(retrieved_nouns_item) == 0:
                    overlap_score = 0.0
                else:
                    overlap_score = (
                        0.5 * (overlap_count / len(query_nouns_item))
                        + 0.5 * (overlap_count / len(retrieved_nouns_item))
                    )

                time_gap      = current_virtual_seconds - retrieved_item.get("virtual_seconds", 0.0)
                time_decay    = math.exp(-self.decay_temp * max(time_gap, 0.0))
                overall_score = time_decay * overlap_score

                if (self.ori_mem_query
                        and distance < dist_thres
                        and distance < best_memory["distance"]):
                    empty_flag  = True
                    best_memory = {
                        "idx": idx, "overall_score": overall_score,
                        "overlap_score": overlap_score,
                        "overlap_count": overlap_count, "distance": distance,
                    }
                elif overlap_count > 0 and overall_score >= best_memory["overall_score"]:
                    empty_flag  = True
                    best_memory = {
                        "idx": idx, "overall_score": overall_score,
                        "overlap_score": overlap_score,
                        "overlap_count": overlap_count, "distance": distance,
                    }

            if empty_flag:
                memory_metadata          = results["metadatas"][0][best_memory["idx"]].copy()
                memory_metadata["score"] = best_memory["distance"]
                metadata_list.append(memory_metadata)
                self.overall_retrieve_score += best_memory["overlap_score"]
                self.overall_retrieve_count += 1

        return metadata_list

    def context_retrieve(
        self,
        query:                   str,
        current_virtual_seconds: float = None,
        current_conv_id:         int   = None,
        current_session_id:      int   = None,
    ) -> List[Dict[str, Any]]:
        """
        Add user turn to STM; detect conv_id boundary and finalize every N conv_ids.
        Returns the full current STM (after appending current user turn).
        """
        job = self.prepare_boundary_summary(
            current_virtual_seconds=current_virtual_seconds,
            current_conv_id=current_conv_id,
            current_session_id=current_session_id,
        )
        if job is not None:
            last_session_summary = self._context_summarize(
                job["merged_context"], job["num_lines"]
            )
            self.apply_boundary_summary_result(job, last_session_summary)

        return self.append_user_query(
            query=query,
            current_virtual_seconds=current_virtual_seconds,
            current_conv_id=current_conv_id,
            current_session_id=current_session_id,
        )

    def _build_summary_prompts(self, context: str) -> Dict[str, str]:
        sys_prompt = (
            f"You are good at extracting events and summarizing them in brief sentences. "
            f"You will be shown a conversation between {self.usr_name} and {self.agent_name}.\n"
        )
        user_prompt = (
            f"#Conversation#:\n{context}.\n"
            f"Based on the Conversation, please summarize the main points of the "
            f"conversation with brief sentences in English, within 20 words.\n"
            f"Respond in JSON format with key \"summary\".\n"
        )
        return {"system_prompt": sys_prompt, "user_prompt": user_prompt}

    def _prepare_summary_job(
        self,
        current_virtual_seconds: float,
        current_session_id: int,
        clear_stm_after_store: bool,
    ) -> Dict[str, Any]:
        last_session_context = [
            f"(line {i + 1}) {mem['dialog']}."
            for i, mem in enumerate(self.short_term_memory)
        ]
        merged_context = "\n".join(last_session_context)
        # _context_summarize uses max_tokens=100; subtract that from the budget
        budget = int(self._finalize_input_context_limit * self._finalize_context_utilization) - 100
        merged_context = _truncate_oldest_first(merged_context, budget, log=self.logger)
        last_session_context = merged_context.split("\n")
        prompts = self._build_summary_prompts(merged_context)

        last_entry = self.short_term_memory[-1]
        last_vs = last_entry.get("virtual_seconds", current_virtual_seconds)
        last_session_id = last_entry.get("session_id", current_session_id)

        return {
            "merged_context": merged_context,
            "num_lines": len(last_session_context),
            "system_prompt": prompts["system_prompt"],
            "user_prompt": prompts["user_prompt"],
            "virtual_seconds": last_vs,
            "conv_id": self.last_conv_id,
            "session_id": last_session_id,
            "clear_stm_after_store": clear_stm_after_store,
        }

    def _store_summary_result(self, job: Dict[str, Any], summary_text: str):
        tokenized_item = self.lemma_tokenizer(job["merged_context"])
        context_nouns = list(set([
            token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"
        ]))
        merged_nouns_str = ",".join(context_nouns)

        metadata = MetaData(
            idx=len(self._ltm_metadata),
            dialog="",
            virtual_seconds=job["virtual_seconds"],
            topics=merged_nouns_str,
            datatype="text",
            summary=summary_text,
            conv_id=job["conv_id"],
            session_id=job["session_id"],
        )
        self.store(
            ids=len(self._ltm_metadata),
            key=merged_nouns_str,
            metadata=metadata.to_dict(),
        )

        if job["clear_stm_after_store"]:
            self.short_term_memory = []
            self._pending_conv_count = 0
            self.logger.debug(
                f"Finalized {self.finalize_every_n_convs} conv_ids "
                f"(last conv_id={job['conv_id']}) → "
                f"LTM entry #{len(self._ltm_metadata) - 1}. STM cleared."
            )
        else:
            self.logger.debug(
                f"flush_stm: {job['num_lines']} lines → "
                f"LTM entry #{len(self._ltm_metadata) - 1}. STM retained for QA."
            )

    def prepare_boundary_summary(
        self,
        current_virtual_seconds: float = None,
        current_conv_id: int = None,
        current_session_id: int = None,
    ) -> Optional[Dict[str, Any]]:
        self.last_summarize_token_info = {"input": 0, "output": 0}

        if current_virtual_seconds is None:
            current_virtual_seconds = self.current_virtual_seconds
        current_conv_id = (
            current_conv_id if current_conv_id is not None else self.last_conv_id
        )

        if not (
            len(self.short_term_memory) > 0
            and current_conv_id != self.last_conv_id
            and self.last_conv_id >= 0
        ):
            return None

        self._pending_conv_count += 1
        if self._pending_conv_count < self.finalize_every_n_convs:
            self.logger.debug(
                f"Conv boundary {self.last_conv_id}→{current_conv_id}: "
                f"pending={self._pending_conv_count}/{self.finalize_every_n_convs}, "
                f"accumulating."
            )
            return None

        return self._prepare_summary_job(
            current_virtual_seconds=current_virtual_seconds,
            current_session_id=current_session_id,
            clear_stm_after_store=True,
        )

    def apply_boundary_summary_result(self, job: Dict[str, Any], summary_text: str):
        self._store_summary_result(job, summary_text)

    def append_user_query(
        self,
        query: str,
        current_virtual_seconds: float = None,
        current_conv_id: int = None,
        current_session_id: int = None,
    ) -> List[Dict[str, Any]]:
        if current_virtual_seconds is None:
            current_virtual_seconds = self.current_virtual_seconds
        current_conv_id = (
            current_conv_id if current_conv_id is not None else self.last_conv_id
        )

        data = {
            "idx": len(self.short_term_memory),
            "virtual_seconds": current_virtual_seconds,
            "dialog": f"{self.usr_name}: {query}",
            "conv_id": current_conv_id,
            "session_id": current_session_id,
        }
        self.short_term_memory.append(data)
        self.last_conv_id = current_conv_id
        self.current_virtual_seconds = current_virtual_seconds

        return list(self.short_term_memory)

    def add_agent_response(
        self,
        response:                str,
        current_virtual_seconds: float = None,
        current_session_id:      int   = None,
    ):
        """Add GT agent response to short-term memory."""
        if current_virtual_seconds is None:
            current_virtual_seconds = self.current_virtual_seconds
        if current_session_id is None and self.short_term_memory:
            current_session_id = self.short_term_memory[-1].get("session_id", 0)

        data = {
            "idx":             len(self.short_term_memory),
            "virtual_seconds": current_virtual_seconds,
            "dialog":          f"{self.agent_name}: {response}",
            "conv_id":         self.last_conv_id,
            "session_id":      current_session_id,
        }
        self.short_term_memory.append(data)

    # =========================================================================
    # SUMMARIZATION
    # =========================================================================

    def _context_summarize(self, context: str, length: int) -> str:
        """
        Summarize the context using LLM.
        Stores token info in last_summarize_token_info.
        Logs to llm_logger as call_4_summarization if logger is set.
        """
        prompts = self._build_summary_prompts(context)
        sys_prompt = prompts["system_prompt"]
        user_prompt = prompts["user_prompt"]

        try:
            result = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                max_tokens=100,
                temperature=0.7,
                guided_json=SUMMARY_SCHEMA,
                return_usage=True,
            )

            if self.llm_logger is not None:
                self.llm_logger.log("call_4_summarization", sys_prompt, user_prompt, result)

            if isinstance(result, dict) and "_usage" in result:
                usage = result["_usage"]
                self.last_summarize_token_info = {
                    "input":  usage.get("prompt_tokens",     0),
                    "output": usage.get("completion_tokens", 0),
                }

            return str(result.get("summary", "")).strip()

        except Exception as e:
            self.logger.error(f"Error in context summarization: {e}")
            return "Summary generation failed."

    # =========================================================================
    # STM FLUSH (final batch, called before QA)
    # =========================================================================

    def flush_stm(self, current_session_id: int = 0) -> Dict[str, int]:
        """
        Force-commit remaining STM content to LTM.
        Called by ldagent_module before Phase 2 QA.
        STM is kept intact after flush so QA can use recent context.
        Returns token info dict from the summarize call.
        """
        job = self.prepare_flush_summary(current_session_id=current_session_id)
        if job is None:
            self.logger.debug("flush_stm: STM empty, nothing to flush.")
            return {"input": 0, "output": 0, "calls": 0}

        last_session_summary = self._context_summarize(
            job["merged_context"], job["num_lines"]
        )
        self.apply_flush_summary_result(job, last_session_summary)

        flush_tokens = dict(self.last_summarize_token_info)

        return {
            "input":  flush_tokens.get("input",  0),
            "output": flush_tokens.get("output", 0),
            "calls":  1 if flush_tokens.get("input", 0) > 0 else 0,
        }

    def prepare_flush_summary(self, current_session_id: int = 0) -> Optional[Dict[str, Any]]:
        self.last_summarize_token_info = {"input": 0, "output": 0}
        if not self.short_term_memory:
            return None
        return self._prepare_summary_job(
            current_virtual_seconds=self.current_virtual_seconds,
            current_session_id=current_session_id,
            clear_stm_after_store=False,
        )

    def apply_flush_summary_result(self, job: Dict[str, Any], summary_text: str):
        self._store_summary_result(job, summary_text)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_memory_count(self)     -> int: return len(self._ltm_metadata)
    def get_short_term_count(self) -> int: return len(self.short_term_memory)

    def get_memory_stats(self) -> Dict[str, int]:
        """Return LTM entry count and estimated total content tokens."""
        num_memories = len(self._ltm_metadata)
        total_chars  = sum(
            len(meta.get("summary", meta.get("dialog", "")))
            for meta in self._ltm_metadata
        )
        return {
            "num_memories":         num_memories,
            "total_content_tokens": total_chars // 4,
        }

    def clear(self):
        """Clear all memory (LTM + STM). Called between sessions."""
        self.short_term_memory  = []
        self._ltm_embeddings    = []
        self._ltm_metadata      = []
        self._ltm_documents     = []

        self.current_virtual_seconds   = 0.0
        self.last_conv_id              = -1
        self._pending_conv_count       = 0
        self.overall_retrieve_score    = 0.0
        self.overall_retrieve_count    = 0
        self.last_summarize_token_info = {"input": 0, "output": 0}

        self.logger.info(f"Memory cleared for sample {self.sample_id}")

    def save_snapshot(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # STM
        with open(directory / "short_term_memory.json", "w") as f:
            json.dump(self.short_term_memory, f, indent=2)

        # LTM metadata + documents
        with open(directory / "long_term_memory.json", "w") as f:
            json.dump({
                "documents": self._ltm_documents,
                "metadatas": self._ltm_metadata,
            }, f, indent=2)

        # LTM embeddings
        if self._ltm_embeddings:
            emb_matrix = np.stack(self._ltm_embeddings)
        else:
            emb_matrix = np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
        np.save(directory / "ltm_embeddings.npy", emb_matrix)

        # State
        state = {
            "sample_id":               self.sample_id,
            "current_virtual_seconds": self.current_virtual_seconds,
            "last_conv_id":            self.last_conv_id,
            "pending_conv_count":      self._pending_conv_count,
            "overall_retrieve_score":  self.overall_retrieve_score,
            "overall_retrieve_count":  self.overall_retrieve_count,
            "long_term_count":         len(self._ltm_metadata),
            "short_term_count":        len(self.short_term_memory),
        }
        with open(directory / "memory_state.json", "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Memory snapshot saved to {directory}")

    def load_snapshot(self, directory: Path):
        directory = Path(directory)

        stm_file = directory / "short_term_memory.json"
        if stm_file.exists():
            with open(stm_file) as f:
                self.short_term_memory = json.load(f)

        ltm_file = directory / "long_term_memory.json"
        emb_file = directory / "ltm_embeddings.npy"

        if ltm_file.exists():
            with open(ltm_file) as f:
                data = json.load(f)
            self._ltm_documents = data.get("documents", [])
            self._ltm_metadata  = data.get("metadatas", [])

            if emb_file.exists():
                emb_matrix = np.load(emb_file)
                self._ltm_embeddings = [emb_matrix[i] for i in range(len(emb_matrix))]
            elif self._ltm_documents:
                # Legacy snapshot without .npy: re-encode documents
                self.logger.warning(
                    "ltm_embeddings.npy not found; re-encoding LTM documents."
                )
                embs = self._encode(self._ltm_documents)
                self._ltm_embeddings = [embs[i] for i in range(len(embs))]
            else:
                self._ltm_embeddings = []

        state_file = directory / "memory_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self.current_virtual_seconds = state.get("current_virtual_seconds", 0.0)
            self.last_conv_id            = state.get("last_conv_id", -1)
            self._pending_conv_count     = state.get("pending_conv_count", 0)
            self.overall_retrieve_score  = state.get("overall_retrieve_score", 0.0)
            self.overall_retrieve_count  = state.get("overall_retrieve_count", 0)

        self.logger.info(
            f"Memory snapshot loaded from {directory} "
            f"(LTM={len(self._ltm_metadata)}, STM={len(self.short_term_memory)})"
        )
