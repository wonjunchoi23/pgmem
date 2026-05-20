"""
Event Memory Module for LD-Agent

Changes from v2:
  - turn_index / TURN_DECAY_TEMP  →  virtual_seconds / DECAY_TEMP (seconds-based)
  - STM finalization every FINALIZE_EVERY_N_CONVS conv_ids (tracked internally)
  - STM fully cleared at finalization boundary (original LD-Agent behaviour)
  - Full STM returned from context_retrieve() — no n_results slicing
  - LTM write guard (_ltm_write_enabled) removed (Phase 2 never calls context_retrieve)
  - LLMCallLogger support added (call_4_summarization)

Reference: "Hello Again! LLM-powered Personalized Agent for Long-term Dialogue"
           (Li et al., NAACL 2025)
"""

import math
import json
import logging
import spacy
import chromadb
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)


# =============================================================================
# METADATA CLASS
# =============================================================================

@dataclass
class MetaData:
    idx:             int   = 0
    dialog:          str   = ""
    virtual_seconds: float = 0.0   # replaces turn_index; virtual elapsed seconds
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

    STM finalization policy (matches original LD-Agent session-boundary logic):
      - STM accumulates turns within a group of FINALIZE_EVERY_N_CONVS conv_ids.
      - When the group is complete (N conv_ids crossed), STM is summarized → LTM
        and then fully cleared.
      - flush_stm() handles the final incomplete group at end of Phase 1.

    Virtual time:
      - Each turn carries a virtual_seconds value (computed from conv_id + turn_id).
      - LTM entries store virtual_seconds for decay scoring.
      - Decay: exp(-DECAY_TEMP × elapsed_virtual_seconds).
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
        finalize_every_n_convs:  int   = 2,
        memory_cache_path:       Optional[str] = None,
    ):
        self.llm_client  = llm_client
        self.sample_id   = sample_id
        self.logger      = logger

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

        self.embedding_function = DefaultEmbeddingFunction()

        try:
            self.lemma_tokenizer = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.lemma_tokenizer = spacy.load("en_core_web_sm")

        if memory_cache_path:
            self.dbclient = chromadb.PersistentClient(path=memory_cache_path)
        else:
            self.dbclient = chromadb.Client()

        self.collection = self.dbclient.get_or_create_collection(
            name=f"collection_{sample_id}",
            embedding_function=self.embedding_function,
        )

        self.short_term_memory: List[Dict[str, Any]] = []
        self.current_virtual_seconds  = 0.0
        self.last_conv_id             = -1
        self._pending_conv_count      = 0   # completed conv_ids since last LTM flush
        self.overall_retrieve_score   = 0.0
        self.overall_retrieve_count   = 0

        logger.info(f"EventMemory initialized for sample {sample_id} "
                    f"(finalize_every={finalize_every_n_convs} convs)")

    # =========================================================================
    # LLM LOGGER INJECTION
    # =========================================================================

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

    # =========================================================================
    # STORAGE METHODS
    # =========================================================================

    def store(
        self,
        ids:      Any,
        key:      Any,
        metadata: Dict[str, Any],
    ):
        if not isinstance(key, list):
            key = [key]
        if not isinstance(ids, list):
            ids = [str(ids)]
        self.collection.add(
            ids=ids,
            documents=key,
            metadatas=[metadata] if not isinstance(metadata, list) else metadata,
        )

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

        if self.collection.count() == 0:
            return []

        n_query = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=query, n_results=n_query)

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

        Finalization policy (matches original LD-Agent at session boundary):
          - When _pending_conv_count reaches finalize_every_n_convs:
              1. Summarize current STM → store as LTM entry
              2. Fully clear STM
              3. Reset _pending_conv_count = 0

        Returns the full current STM (after appending current user turn).
        """
        # Reset summarize token info so non-finalizing turns expose zeros
        self.last_summarize_token_info = {"input": 0, "output": 0}

        if current_virtual_seconds is None:
            current_virtual_seconds = self.current_virtual_seconds
        current_conv_id = (
            current_conv_id if current_conv_id is not None else self.last_conv_id
        )

        # ── Conv boundary detection ──────────────────────────────────────────
        if (len(self.short_term_memory) > 0
                and current_conv_id != self.last_conv_id
                and self.last_conv_id >= 0):

            self._pending_conv_count += 1

            if self._pending_conv_count >= self.finalize_every_n_convs:
                # Finalize: summarize STM → LTM → clear STM (original LD-Agent style)
                last_session_context = [
                    f"(line {i + 1}) {mem['dialog']}."
                    for i, mem in enumerate(self.short_term_memory)
                ]
                merged_context       = "\n".join(last_session_context)
                last_session_summary = self._context_summarize(
                    merged_context, len(last_session_context)
                )

                tokenized_item   = self.lemma_tokenizer(merged_context)
                context_nouns    = list(set([
                    token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"
                ]))
                merged_nouns_str = ",".join(context_nouns)

                last_entry      = self.short_term_memory[-1]
                last_vs         = last_entry.get("virtual_seconds", current_virtual_seconds)
                last_session_id = last_entry.get("session_id", current_session_id)

                metadata = MetaData(
                    idx=self.collection.count(),
                    dialog="",
                    virtual_seconds=last_vs,
                    topics=merged_nouns_str,
                    datatype="text",
                    summary=last_session_summary,
                    conv_id=self.last_conv_id,
                    session_id=last_session_id,
                )
                self.store(
                    ids=self.collection.count(),
                    key=merged_nouns_str,
                    metadata=metadata.to_dict(),
                )

                # Fully clear STM (original LD-Agent behaviour)
                self.short_term_memory   = []
                self._pending_conv_count = 0

                self.logger.debug(
                    f"Finalized {self.finalize_every_n_convs} conv_ids "
                    f"(last conv_id={self.last_conv_id}) → "
                    f"LTM entry #{self.collection.count() - 1}. STM cleared."
                )
            else:
                self.logger.debug(
                    f"Conv boundary {self.last_conv_id}→{current_conv_id}: "
                    f"pending={self._pending_conv_count}/{self.finalize_every_n_convs}, "
                    f"accumulating."
                )

        # ── Append current user turn to STM ─────────────────────────────────
        data = {
            "idx":             len(self.short_term_memory),
            "virtual_seconds": current_virtual_seconds,
            "dialog":          f"{self.usr_name}: {query}",
            "conv_id":         current_conv_id,
            "session_id":      current_session_id,
        }
        self.short_term_memory.append(data)
        self.last_conv_id            = current_conv_id
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
        Summarize the context using LLM (exact prompt from original LD-Agent).
        Stores token info in last_summarize_token_info.
        Logs to llm_logger as call_4_summarization if logger is set.
        """
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
        self.last_summarize_token_info = {"input": 0, "output": 0}

        if not self.short_term_memory:
            self.logger.debug("flush_stm: STM empty, nothing to flush.")
            return {"input": 0, "output": 0, "calls": 0}

        last_session_context = [
            f"(line {i + 1}) {mem['dialog']}."
            for i, mem in enumerate(self.short_term_memory)
        ]
        merged_context       = "\n".join(last_session_context)
        last_session_summary = self._context_summarize(merged_context, len(last_session_context))

        flush_tokens = dict(self.last_summarize_token_info)

        tokenized_item   = self.lemma_tokenizer(merged_context)
        context_nouns    = list(set([
            token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"
        ]))
        merged_nouns_str = ",".join(context_nouns)

        last_entry      = self.short_term_memory[-1]
        last_vs         = last_entry.get("virtual_seconds", self.current_virtual_seconds)
        last_conv_id    = last_entry.get("conv_id",         self.last_conv_id)
        last_session_id = last_entry.get("session_id",      current_session_id)

        metadata = MetaData(
            idx=self.collection.count(),
            dialog="",
            virtual_seconds=last_vs,
            topics=merged_nouns_str,
            datatype="text",
            summary=last_session_summary,
            conv_id=last_conv_id,
            session_id=last_session_id,
        )
        self.store(
            ids=self.collection.count(),
            key=merged_nouns_str,
            metadata=metadata.to_dict(),
        )

        self.logger.debug(
            f"flush_stm: {len(last_session_context)} lines → "
            f"LTM entry #{self.collection.count() - 1}. STM retained for QA."
        )

        return {
            "input":  flush_tokens.get("input",  0),
            "output": flush_tokens.get("output", 0),
            "calls":  1 if flush_tokens.get("input", 0) > 0 else 0,
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_memory_count(self)     -> int: return self.collection.count()
    def get_short_term_count(self) -> int: return len(self.short_term_memory)

    def clear(self):
        """Clear all memory (LTM + STM). Called between sessions."""
        self.short_term_memory = []

        try:
            self.dbclient.delete_collection(name=f"collection_{self.sample_id}")
        except Exception:
            pass

        self.collection = self.dbclient.get_or_create_collection(
            name=f"collection_{self.sample_id}",
            embedding_function=self.embedding_function,
        )

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

        with open(directory / "short_term_memory.json", "w") as f:
            json.dump(self.short_term_memory, f, indent=2)

        long_term_file = directory / "long_term_memory.json"
        if self.collection.count() > 0:
            all_data = self.collection.get(include=["documents", "metadatas"])
            with open(long_term_file, "w") as f:
                json.dump({
                    "ids":       all_data["ids"],
                    "documents": all_data["documents"],
                    "metadatas": all_data["metadatas"],
                }, f, indent=2)
        else:
            with open(long_term_file, "w") as f:
                json.dump({"ids": [], "documents": [], "metadatas": []}, f)

        state = {
            "sample_id":               self.sample_id,
            "current_virtual_seconds": self.current_virtual_seconds,
            "last_conv_id":            self.last_conv_id,
            "pending_conv_count":      self._pending_conv_count,
            "overall_retrieve_score":  self.overall_retrieve_score,
            "overall_retrieve_count":  self.overall_retrieve_count,
            "long_term_count":         self.collection.count(),
            "short_term_count":        len(self.short_term_memory),
        }
        with open(directory / "memory_state.json", "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Memory snapshot saved to {directory}")

    def load_snapshot(self, directory: Path):
        directory = Path(directory)

        stm_file = directory / "short_term_memory.json"
        if stm_file.exists():
            with open(stm_file, "r") as f:
                self.short_term_memory = json.load(f)

        ltm_file = directory / "long_term_memory.json"
        if ltm_file.exists():
            with open(ltm_file, "r") as f:
                data = json.load(f)
            self.clear()
            if data["ids"]:
                self.collection.add(
                    ids=data["ids"],
                    documents=data["documents"],
                    metadatas=data["metadatas"],
                )

        state_file = directory / "memory_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            self.current_virtual_seconds = state.get("current_virtual_seconds", 0.0)
            self.last_conv_id            = state.get("last_conv_id",            -1)
            self._pending_conv_count     = state.get("pending_conv_count",       0)
            self.overall_retrieve_score  = state.get("overall_retrieve_score",  0.0)
            self.overall_retrieve_count  = state.get("overall_retrieve_count",  0)

        self.logger.info(f"Memory snapshot loaded from {directory}")
