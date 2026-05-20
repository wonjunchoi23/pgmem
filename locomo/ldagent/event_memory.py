"""
Event Memory Module for LD-Agent (LoComo) — Batch-enabled variant

LTM storage uses numpy + SentenceTransformer (replaces ChromaDB):
  - Embeddings stored as list of (dim,) float32 arrays in memory.
  - Retrieval: L2 distance on normalized sentence-transformer embeddings,
    then noun-overlap × time-decay scoring (identical logic to ChromaDB version).
  - Snapshot: ltm_embeddings.npy + long_term_memory.json + memory_state.json.

Batch API methods:
  should_flush(new_timestamp)   — check flush condition without side effects
  build_flush_context()         — pre-compute merged context, nouns, metadata
                                  for the pending flush; returns a flush_ctx dict
  apply_flush(summary, usage,
              flush_ctx,
              clear_stm)        — write LTM entry using pre-generated summary;
                                  clear STM if clear_stm=True (mid-Phase-1),
                                  keep STM if False (final flush before QA)
  append_turn_to_stm(...)       — STM append without flush check (use after
                                  apply_flush has already been called)
  get_memory_stats()            — LTM count + estimated content tokens

All original sequential methods (store_turn, flush_stm, etc.) are preserved
unchanged so the module remains drop-in compatible with ldagent_module.py.
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

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_DIM   = 384


# =============================================================================
# METADATA CLASS
# =============================================================================

@dataclass
class MetaData:
    idx:         int   = 0
    dialog:      str   = ""
    timestamp:   float = 0.0
    topics:      str   = ""
    datatype:    str   = "text"
    summary:     str   = ""
    session_num: int   = 0
    sample_id:   str   = ""
    date_time:   str   = ""
    dia_ids:     str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx":         self.idx,
            "dialog":      self.dialog,
            "timestamp":   self.timestamp,
            "topics":      self.topics,
            "datatype":    self.datatype,
            "summary":     self.summary,
            "session_num": self.session_num,
            "sample_id":   self.sample_id,
            "date_time":   self.date_time,
            "dia_ids":     self.dia_ids,
        }


# =============================================================================
# EVENT MEMORY CLASS
# =============================================================================

class EventMemory:
    """
    Long-term and short-term memory module for LD-Agent (LoComo batch variant).

    LTM storage uses numpy + SentenceTransformer (replaces ChromaDB):
      - self._ltm_embeddings: List[np.ndarray] — per-entry (dim,) float32 vectors
      - self._ltm_metadata:   List[Dict]       — per-entry metadata dicts
      - self._ltm_documents:  List[str]        — per-entry noun string (for re-encoding)
    """

    def __init__(
        self,
        llm_client,
        sample_id:               str,
        logger:                  logging.Logger,
        speaker_a:               str   = "Speaker A",
        speaker_b:               str   = "Speaker B",
        relevance_memory_number: int   = 1,
        dist_threshold:          float = 1.5,
        decay_temp:              float = 1e-7,
        ori_mem_query:           bool  = False,
        flush_gap_seconds:       float = 3600.0,
        memory_cache_path:       Optional[str] = None,  # kept for compat; unused
        lemma_tokenizer=None,
        encoder=None,
    ):
        self.llm_client  = llm_client
        self.sample_id   = sample_id
        self.logger      = logger

        self.speaker_a               = speaker_a
        self.speaker_b               = speaker_b
        self.relevance_memory_number = relevance_memory_number
        self.dist_threshold          = dist_threshold
        self.decay_temp              = decay_temp
        self.ori_mem_query           = ori_mem_query
        self.flush_gap_seconds       = flush_gap_seconds

        self.llm_logger = None
        self.last_summarize_token_info: Dict[str, int] = {"input": 0, "output": 0}

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

        if encoder is not None:
            self._encoder = encoder
        else:
            logger.info(f"Loading SentenceTransformer encoder ({_EMBEDDING_MODEL})...")
            self._encoder = SentenceTransformer(_EMBEDDING_MODEL)

        # LTM storage (replaces ChromaDB)
        self._ltm_embeddings: List[np.ndarray] = []
        self._ltm_metadata:   List[Dict]       = []
        self._ltm_documents:  List[str]        = []

        self.short_term_memory: List[Dict[str, Any]] = []
        self.current_timestamp      = 0.0
        self.overall_retrieve_score = 0.0
        self.overall_retrieve_count = 0

        logger.info(f"EventMemory initialized for sample {sample_id} "
                    f"(flush_gap={flush_gap_seconds}s, decay={decay_temp})")

    # =========================================================================
    # LLM LOGGER INJECTION
    # =========================================================================

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

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
        """
        if not self._ltm_metadata:
            return {"metadatas": [[]], "distances": [[]]}

        query_embs = self._encode(query_texts)        # (n_q, dim)
        query_emb  = query_embs.mean(axis=0)          # (dim,)

        emb_matrix = np.stack(self._ltm_embeddings)   # (n, dim)
        diffs      = emb_matrix - query_emb[np.newaxis, :]
        distances  = np.sqrt(np.sum(diffs ** 2, axis=1))  # (n,)

        n        = min(n_results, len(distances))
        top_idxs = np.argsort(distances)[:n]

        return {
            "metadatas": [[self._ltm_metadata[i] for i in top_idxs]],
            "distances": [[float(distances[i])   for i in top_idxs]],
        }

    # =========================================================================
    # STORAGE METHODS
    # =========================================================================

    def store(self, ids: Any, key: Any, metadata: Dict[str, Any]):
        if isinstance(key, list):
            key = key[0]
        if isinstance(metadata, list):
            metadata = metadata[0]
        self._ltm_add(key, metadata)

    # =========================================================================
    # TURN STORAGE  (sequential — unchanged)
    # =========================================================================

    def store_turn(
        self,
        speaker_name: str,
        text:         str,
        dia_id:       str,
        timestamp:    float,
        session_num:  int,
        date_time:    str,
        sample_id:    str,
    ):
        """Store a single turn in STM, flushing to LTM at session boundaries."""
        self.last_summarize_token_info = {"input": 0, "output": 0}

        if (len(self.short_term_memory) > 0 and
                timestamp - self.short_term_memory[-1]["timestamp"] > self.flush_gap_seconds):
            self._flush_stm_to_ltm(current_sample_id=sample_id)
            self.logger.debug(
                f"Session boundary detected at dia_id={dia_id}. STM flushed to LTM."
            )

        dialog = f"Speaker {speaker_name} says: {text}"
        entry = {
            "idx":         len(self.short_term_memory),
            "timestamp":   timestamp,
            "dialog":      dialog,
            "session_num": session_num,
            "dia_id":      dia_id,
            "date_time":   date_time,
            "sample_id":   sample_id,
        }
        self.short_term_memory.append(entry)
        self.current_timestamp = timestamp

    # =========================================================================
    # STM CONTEXT FOR QA
    # =========================================================================

    def get_stm_context(self) -> List[Dict[str, Any]]:
        return list(self.short_term_memory)

    # =========================================================================
    # RETRIEVAL METHODS
    # =========================================================================

    def relevance_retrieve(
        self,
        ori_query:         str,
        n_results:         int   = 10,
        dist_thres:        float = None,
        current_timestamp: float = None,
    ) -> List[Dict[str, Any]]:
        dist_thres = dist_thres if dist_thres is not None else self.dist_threshold
        if current_timestamp is None:
            current_timestamp = self.current_timestamp

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
                overlap_count = len(set(query_nouns_item) & set(retrieved_nouns_item))

                if len(query_nouns_item) == 0 or len(retrieved_nouns_item) == 0:
                    overlap_score = 0.0
                else:
                    overlap_score = (
                        0.5 * (overlap_count / len(query_nouns_item))
                        + 0.5 * (overlap_count / len(retrieved_nouns_item))
                    )

                time_gap      = current_timestamp - retrieved_item.get("timestamp", 0.0)
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

    # =========================================================================
    # SUMMARIZATION
    # =========================================================================

    def _context_summarize(self, context: str, speaker_a: str, speaker_b: str) -> str:
        sys_prompt = (
            f"You are good at extracting events and summarizing them in brief sentences. "
            f"You will be shown a conversation between {speaker_a} and {speaker_b}.\n"
        )
        user_prompt = (
            f"#Conversation#:\n{context}.\n"
            f"Based on the Conversation, please summarize the main points of the "
            f"conversation with brief sentences in English, within 20 words.\n"
            f"Respond in JSON format with key \"summary\".\n"
        )

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
                self.llm_logger.log("call_3_summarization", sys_prompt, user_prompt, result)

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
    # STM → LTM FLUSH (shared logic)
    # =========================================================================

    def _do_flush(self, current_sample_id: str, clear_stm: bool) -> Dict[str, int]:
        """Summarize STM and write one LTM entry. Clears STM iff clear_stm=True."""
        last_session_context = [
            f"(line {i + 1}) {mem['dialog']}."
            for i, mem in enumerate(self.short_term_memory)
        ]
        merged_context = "\n".join(last_session_context)

        summary = self._context_summarize(
            merged_context,
            speaker_a=self.speaker_a,
            speaker_b=self.speaker_b,
        )
        flush_tokens = dict(self.last_summarize_token_info)

        tokenized_item   = self.lemma_tokenizer(merged_context)
        context_nouns    = list(set([
            token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"
        ]))
        merged_nouns_str = ",".join(context_nouns)

        last_entry  = self.short_term_memory[-1]
        last_ts     = last_entry.get("timestamp",   self.current_timestamp)
        last_sess   = last_entry.get("session_num", 0)
        last_dt     = last_entry.get("date_time",   "")
        sample_id   = last_entry.get("sample_id",   current_sample_id)
        dia_ids_str = ",".join(m.get("dia_id", "") for m in self.short_term_memory if m.get("dia_id"))

        metadata = MetaData(
            idx=len(self._ltm_metadata),
            dialog="",
            timestamp=last_ts,
            topics=merged_nouns_str,
            datatype="text",
            summary=summary,
            session_num=last_sess,
            sample_id=sample_id,
            date_time=last_dt,
            dia_ids=dia_ids_str,
        )
        self.store(
            ids=len(self._ltm_metadata),
            key=merged_nouns_str,
            metadata=metadata.to_dict(),
        )

        label = "_flush_stm_to_ltm" if clear_stm else "flush_stm"
        suffix = "" if clear_stm else ". STM retained for QA."
        self.logger.debug(
            f"{label}: {len(last_session_context)} lines → "
            f"LTM entry #{len(self._ltm_metadata) - 1}{suffix}."
        )

        if clear_stm:
            self.short_term_memory = []

        return {
            "input":  flush_tokens.get("input",  0),
            "output": flush_tokens.get("output", 0),
            "calls":  1 if flush_tokens.get("input", 0) > 0 else 0,
        }

    def _flush_stm_to_ltm(self, current_sample_id: str = "") -> Dict[str, int]:
        if not self.short_term_memory:
            return {"input": 0, "output": 0, "calls": 0}
        return self._do_flush(current_sample_id, clear_stm=True)

    def flush_stm(self, current_sample_id: str = "") -> Dict[str, int]:
        """Force-commit remaining STM to LTM before Phase 2 QA. STM is kept."""
        self.last_summarize_token_info = {"input": 0, "output": 0}

        if not self.short_term_memory:
            self.logger.debug("flush_stm: STM empty, nothing to flush.")
            return {"input": 0, "output": 0, "calls": 0}

        return self._do_flush(current_sample_id, clear_stm=False)

    # =========================================================================
    # BATCH API
    # =========================================================================

    def should_flush(self, new_timestamp: float) -> bool:
        """Return True if adding a turn at new_timestamp would trigger STM→LTM flush."""
        return (
            len(self.short_term_memory) > 0
            and new_timestamp - self.short_term_memory[-1]["timestamp"] > self.flush_gap_seconds
        )

    def build_flush_context(self) -> Dict[str, Any]:
        """
        Pre-compute everything needed for the flush LLM call.
        Returns a flush_ctx dict. Pass to apply_flush() after the batch LLM call.
        Must only be called when should_flush() is True or STM is non-empty.
        """
        assert self.short_term_memory, "build_flush_context: STM is empty"

        last_session_context = [
            f"(line {i + 1}) {mem['dialog']}."
            for i, mem in enumerate(self.short_term_memory)
        ]
        merged_context = "\n".join(last_session_context)

        tokenized_item = self.lemma_tokenizer(merged_context)
        context_nouns  = list(set([
            token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"
        ]))
        nouns_str = ",".join(context_nouns)

        last_entry  = self.short_term_memory[-1]
        dia_ids_str = ",".join(
            m.get("dia_id", "") for m in self.short_term_memory if m.get("dia_id")
        )

        sys_prompt = (
            f"You are good at extracting events and summarizing them in brief sentences. "
            f"You will be shown a conversation between {self.speaker_a} and {self.speaker_b}.\n"
        )
        user_prompt = (
            f"#Conversation#:\n{merged_context}.\n"
            f"Based on the Conversation, please summarize the main points of the "
            f"conversation with brief sentences in English, within 20 words.\n"
            f"Respond in JSON format with key \"summary\".\n"
        )

        return {
            "nouns_str":       nouns_str,
            "last_entry":      last_entry,
            "dia_ids_str":     dia_ids_str,
            "sys_prompt":      sys_prompt,
            "user_prompt":     user_prompt,
            "combined_prompt": f"{sys_prompt}\n{user_prompt}",
            "num_stm_lines":   len(last_session_context),
        }

    def apply_flush(
        self,
        summary:   str,
        usage:     Dict[str, int],
        flush_ctx: Dict[str, Any],
        clear_stm: bool = True,
    ) -> None:
        """
        Complete a flush using a pre-generated summary from a batch LLM call.

        Args:
            summary:   Summary string from the batch call result.
            usage:     {'prompt_tokens': int, 'completion_tokens': int}
            flush_ctx: Context dict from build_flush_context().
            clear_stm: True  → mid-Phase-1 flush (clear STM after writing LTM).
                       False → final flush before QA (keep STM for context).
        """
        self.last_summarize_token_info = {
            "input":  usage.get("prompt_tokens",     0),
            "output": usage.get("completion_tokens", 0),
        }

        if self.llm_logger is not None:
            self.llm_logger.log(
                "call_3_summarization",
                flush_ctx["sys_prompt"],
                flush_ctx["user_prompt"],
                {"summary": summary, "_usage": self.last_summarize_token_info},
            )

        last_entry = flush_ctx["last_entry"]
        metadata   = MetaData(
            idx=len(self._ltm_metadata),
            dialog="",
            timestamp=last_entry.get("timestamp",   self.current_timestamp),
            topics=flush_ctx["nouns_str"],
            datatype="text",
            summary=summary,
            session_num=last_entry.get("session_num", 0),
            sample_id=last_entry.get("sample_id",    self.sample_id),
            date_time=last_entry.get("date_time",    ""),
            dia_ids=flush_ctx["dia_ids_str"],
        )
        self.store(
            ids=len(self._ltm_metadata),
            key=flush_ctx["nouns_str"],
            metadata=metadata.to_dict(),
        )

        if clear_stm:
            self.short_term_memory = []
            self.logger.debug(
                f"apply_flush (mid): {flush_ctx['num_stm_lines']} lines → "
                f"LTM entry #{len(self._ltm_metadata) - 1}. STM cleared."
            )
        else:
            self.logger.debug(
                f"apply_flush (final): {flush_ctx['num_stm_lines']} lines → "
                f"LTM entry #{len(self._ltm_metadata) - 1}. STM retained for QA."
            )

    def append_turn_to_stm(
        self,
        speaker_name: str,
        text:         str,
        dia_id:       str,
        timestamp:    float,
        session_num:  int,
        date_time:    str,
        sample_id:    str,
    ) -> None:
        """Append a turn to STM without flush check."""
        dialog = f"Speaker {speaker_name} says: {text}"
        entry  = {
            "idx":         len(self.short_term_memory),
            "timestamp":   timestamp,
            "dialog":      dialog,
            "session_num": session_num,
            "dia_id":      dia_id,
            "date_time":   date_time,
            "sample_id":   sample_id,
        }
        self.short_term_memory.append(entry)
        self.current_timestamp = timestamp

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_memory_count(self)     -> int: return len(self._ltm_metadata)
    def get_short_term_count(self) -> int: return len(self.short_term_memory)

    def get_memory_stats(self) -> Dict[str, int]:
        """Return LTM entry count and estimated total content tokens."""
        num_memories = len(self._ltm_metadata)
        total_chars  = sum(
            len(meta.get("summary", ""))
            for meta in self._ltm_metadata
        )
        return {
            "num_memories":         num_memories,
            "total_content_tokens": total_chars // 4,
        }

    def clear(self):
        self.short_term_memory  = []
        self._ltm_embeddings    = []
        self._ltm_metadata      = []
        self._ltm_documents     = []

        self.current_timestamp      = 0.0
        self.overall_retrieve_score = 0.0
        self.overall_retrieve_count = 0
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
            "sample_id":              self.sample_id,
            "current_timestamp":      self.current_timestamp,
            "overall_retrieve_score": self.overall_retrieve_score,
            "overall_retrieve_count": self.overall_retrieve_count,
            "long_term_count":        len(self._ltm_metadata),
            "short_term_count":       len(self.short_term_memory),
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
            self.current_timestamp      = state.get("current_timestamp",      0.0)
            self.overall_retrieve_score = state.get("overall_retrieve_score", 0.0)
            self.overall_retrieve_count = state.get("overall_retrieve_count", 0)

        self.logger.info(
            f"Memory snapshot loaded from {directory} "
            f"(LTM={len(self._ltm_metadata)}, STM={len(self.short_term_memory)})"
        )
