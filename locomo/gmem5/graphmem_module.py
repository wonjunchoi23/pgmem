"""
GraphMem v5 module for LoComo.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from context_cache import ContextCache
from generator import GraphGenerator
from graph_store import HeterogeneousGraph, Node, NODE_C, embed_text, extract_kw_nouns
from retriever import GraphRetriever, RetrievalResult
from updater import GraphUpdater, PendingLLMCall

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    internal_token_info: Dict
    retrieval_result: RetrievalResult
    timing: Optional[Dict] = None


@dataclass
class QAResult:
    answer: str
    token_info: Dict
    retrieved_memories: List[Dict]
    retrieval_result: RetrievalResult
    prompt_snapshot: str
    timing: Optional[Dict] = None


@dataclass
class PreparedQA:
    question: str
    category: int
    temperature: float
    prompt: str
    system_prompt: str
    schema: Dict
    retrieval_result: RetrievalResult
    retrieved_memories: List[Dict]


class LLMCallLogger:
    CALL_DIRS = [
        "call_2_state",
        "call_2b_state_new_rel",
        "call_3_memory",
        "call_3b_memory_new_rel",
        "call_4_trait",
        "call_5a_trait_evidence",
        "call_5b_trait_extra_rel",
        "call_5c_state_state_rel",
        "call_5d_state_memory_rel",
        "call_6_qa",
    ]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "call_type": call_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output if not isinstance(output, dict) else {
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        with open(self._base / call_type / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class GraphMemModule:
    """Top-level GraphMem v5 orchestrator (LoComo)."""

    def __init__(
        self,
        llm_client,
        model_path: str = "",
        config=None,
        llm_log_dir=None,
        embed_model=None,
        nlp=None,
    ) -> None:
        if config is None:
            import config as default_cfg  # type: ignore
            config = default_cfg
        self._cfg = config
        self._llm = llm_client
        self._model_path = model_path

        if embed_model is None:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self._embed_model = embed_model

        if nlp is None:
            import spacy
            try:
                nlp = spacy.load(config.SPACY_MODEL)
            except OSError:
                spacy.cli.download(config.SPACY_MODEL)
                nlp = spacy.load(config.SPACY_MODEL)
        self._nlp = nlp

        self._graph = HeterogeneousGraph()
        self._context_cache = ContextCache(config.CONTEXT_CACHE_SIZE)
        self._llm_logger = LLMCallLogger(llm_log_dir) if llm_log_dir is not None else None
        self._retriever = GraphRetriever(self._graph, self._embed_model, self._nlp, config)
        self._generator = GraphGenerator(llm_client, model_path, config, llm_logger=self._llm_logger)
        self._updater = GraphUpdater(
            self._graph,
            llm_client,
            self._embed_model,
            model_path,
            config,
            llm_logger=self._llm_logger,
            context_cache=self._context_cache,
        )

        self._global_turn = 0
        self._current_session_id = 0
        self._current_turn_idx = 0
        self._current_timestamp_seconds = 0.0

    # ------------------------------------------------------------------
    # Batch-friendly step-wise API
    # ------------------------------------------------------------------

    def prepare_pre_turn_call(
        self,
        session_id: int,
        turn_idx: int,
        timestamp_seconds: float,
    ) -> Optional[PendingLLMCall]:
        return self._updater.prepare_pre_turn_call(
            session_id=session_id,
            turn_idx=turn_idx,
            global_turn=self._global_turn,
            anchor_timestamp_seconds=timestamp_seconds,
        )

    def process_turn_core(
        self,
        speaker: str,
        text: str,
        session_id: int,
        turn_idx: int,
        timestamp_seconds: float,
        dia_id: str = "",
    ) -> TurnResult:
        self._current_session_id = session_id
        self._current_turn_idx = turn_idx
        self._current_timestamp_seconds = timestamp_seconds
        self._updater.ensure_current_chunk(session_id)

        context_cache_str = self._context_cache.get_formatted_context(
            current_timestamp_seconds=timestamp_seconds,
        )

        ctx_text = f"{speaker} says: {text}"
        ctx_kw = self._extract_kw(text)
        ctx_emb = self._embed_text(ctx_text)
        ctx_id = HeterogeneousGraph.new_node_id()
        ctx_node = Node(
            node_id=ctx_id,
            node_type=NODE_C,
            content=ctx_text,
            keywords=ctx_kw,
            embedding=ctx_emb,
            created_at=self._global_turn,
            session_id=session_id,
            conv_id=session_id,
            turn_id=turn_idx,
            timestamp_seconds=timestamp_seconds,
            dia_id=dia_id,
            speaker=speaker,
            retrieval_count=1,
        )
        self._graph.add_node(ctx_node)

        retrieval_result = self._retriever.retrieve(
            query=text,
            global_turn=self._global_turn,
            context_cache_str=context_cache_str,
            current_timestamp_seconds=timestamp_seconds,
        )

        self._context_cache.add_turn(
            speaker=speaker,
            text=text,
            session_id=session_id,
            turn_idx=turn_idx,
            timestamp_seconds=timestamp_seconds,
        )
        self._updater.register_turn(
            speaker=speaker,
            text=text,
            session_id=session_id,
            turn_idx=turn_idx,
            context_node_id=ctx_id,
            timestamp_seconds=timestamp_seconds,
            dia_id=dia_id,
        )

        return TurnResult(
            internal_token_info={},
            retrieval_result=retrieval_result,
        )

    def prepare_post_turn_call(self, session_id: int) -> Optional[PendingLLMCall]:
        return self._updater.prepare_post_turn_call(
            session_id=session_id,
            global_turn=self._global_turn,
            current_turn_idx=self._current_turn_idx,
            current_timestamp_seconds=self._current_timestamp_seconds,
        )

    def prepare_finalize_call(self, session_id: int) -> Optional[PendingLLMCall]:
        return self._updater.prepare_finalize_call(
            session_id=session_id,
            global_turn=self._global_turn,
            anchor_timestamp_seconds=self._current_timestamp_seconds,
        )

    def apply_pending_call(self, call: PendingLLMCall, result: Dict) -> Optional[PendingLLMCall]:
        return self._updater.apply_call_result(call, result)

    def apply_irrelevant_fallback(self, call: PendingLLMCall) -> None:
        self._updater.apply_irrelevant_fallback(call)

    def execute_pending_call(self, call: PendingLLMCall) -> Dict:
        return self._updater.execute_call(call)

    def accumulate_internal_usage(self, input_tokens: int, output_tokens: int, call_type: str) -> None:
        self._updater.accumulate_usage(input_tokens, output_tokens, call_type)

    def get_and_reset_internal_tokens(self) -> Dict:
        return self._updater.get_and_reset_internal_tokens()

    def get_and_reset_internal_stats(self) -> Dict:
        return self._updater.get_and_reset_internal_stats()

    def get_memory_stats(self) -> Dict:
        nodes = self._graph.all_nodes()
        num_memories = len(nodes)
        total_chars = sum(len(n.content) for n in nodes)
        return {
            "num_memories": num_memories,
            "total_content_tokens": total_chars // 4,
        }

    def prepare_qa(
        self,
        question: str,
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
    ) -> PreparedQA:
        context_cache_str = self._context_cache.get_formatted_context(
            current_timestamp_seconds=self._current_timestamp_seconds,
            max_turns=getattr(self._cfg, "QA_CONTEXT_TURNS", None),
        )
        retrieval_result = self._retriever.retrieve(
            query=question,
            global_turn=self._global_turn,
            context_cache_str=context_cache_str,
            current_timestamp_seconds=self._current_timestamp_seconds,
            for_qa=True,
        )
        prompt, system_prompt, schema, temperature = self._generator.build_qa_prompt(
            question=question,
            retrieved_memory=retrieval_result.serialized,
            category=category,
            adversarial_answer=adversarial_answer,
            choice_order_seed=choice_order_seed,
        )
        retrieved_memories = [
            {
                "session_id": n.session_id,
                "conv_id": n.conv_id,
                "turn_id": n.turn_id,
                "dia_id": n.dia_id,
                "speaker": n.speaker,
                "node_type": n.node_type,
            }
            for n in retrieval_result.all_final_nodes
        ]
        return PreparedQA(
            question=question,
            category=category,
            temperature=temperature,
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            retrieval_result=retrieval_result,
            retrieved_memories=retrieved_memories,
        )

    def advance_turn(self) -> None:
        self._global_turn += 1

    def log_call(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        if self._llm_logger is not None:
            self._llm_logger.log(call_type, system_prompt, user_prompt, output)

    # ------------------------------------------------------------------
    # Sequential API
    # ------------------------------------------------------------------

    def process_turn(
        self,
        speaker: str,
        text: str,
        session_id: int,
        turn_idx: int,
        timestamp_seconds: float,
        dia_id: str = "",
        measure_timing: bool = False,
    ) -> TurnResult:
        timing = {} if measure_timing else None
        call = self.prepare_pre_turn_call(session_id, turn_idx, timestamp_seconds)
        while call is not None:
            result = self.execute_pending_call(call)
            call = self.apply_pending_call(call, result)

        turn_result = self.process_turn_core(
            speaker=speaker,
            text=text,
            session_id=session_id,
            turn_idx=turn_idx,
            timestamp_seconds=timestamp_seconds,
            dia_id=dia_id,
        )

        call = self.prepare_post_turn_call(session_id)
        while call is not None:
            result = self.execute_pending_call(call)
            call = self.apply_pending_call(call, result)

        self.advance_turn()
        turn_result.timing = timing
        return turn_result

    def finalize_chunk(self, session_id: int) -> Dict:
        call = self.prepare_finalize_call(session_id)
        while call is not None:
            result = self.execute_pending_call(call)
            call = self.apply_pending_call(call, result)
        return {"internal_token_info": self.get_and_reset_internal_tokens()}

    def get_qa_answer(
        self,
        question: str,
        category: int = 1,
        adversarial_answer: str = "",
        choice_order_seed: str = "",
    ) -> QAResult:
        prepared = self.prepare_qa(
            question, category=category,
            adversarial_answer=adversarial_answer,
            choice_order_seed=choice_order_seed,
        )
        answer, token_info, prompt_snapshot = self._generator.answer_qa(
            question=question,
            retrieved_memory=prepared.retrieval_result.serialized,
            category=category,
            adversarial_answer=adversarial_answer,
            choice_order_seed=choice_order_seed,
        )
        return QAResult(
            answer=answer,
            token_info=token_info,
            retrieved_memories=prepared.retrieved_memories,
            retrieval_result=prepared.retrieval_result,
            prompt_snapshot=prompt_snapshot,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_llm_logger(self, llm_logger) -> None:
        self._llm_logger = llm_logger
        self._generator.set_llm_logger(llm_logger)
        self._updater.set_llm_logger(llm_logger)

    def clear(self) -> None:
        self._graph = HeterogeneousGraph()
        self._retriever.set_graph(self._graph)
        self._updater.set_graph(self._graph)
        self._context_cache.clear()
        self._updater.clear()
        self._global_turn = 0
        self._current_session_id = 0
        self._current_turn_idx = 0
        self._current_timestamp_seconds = 0.0

    def save_snapshot(self, directory: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._graph.save_snapshot(directory / "graph")
        self._context_cache.save_snapshot(directory / "cache")
        metadata = {
            "global_turn": self._global_turn,
            "current_session_id": self._current_session_id,
            "current_turn_idx": self._current_turn_idx,
            "current_timestamp_seconds": self._current_timestamp_seconds,
        }
        with open(directory / "module_state.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)

    def load_snapshot(self, directory: str) -> None:
        directory = Path(directory)
        self._graph.load_snapshot(directory / "graph")
        self._context_cache.load_snapshot(directory / "cache")
        state_file = directory / "module_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self._global_turn = metadata.get("global_turn", 0)
            self._current_session_id = metadata.get("current_session_id", 0)
            self._current_turn_idx = metadata.get("current_turn_idx", 0)
            self._current_timestamp_seconds = metadata.get("current_timestamp_seconds", 0.0)

    def get_node_counts(self) -> Dict[str, int]:
        return self._graph.node_count_by_type()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> np.ndarray:
        return embed_text(self._embed_model, text)

    def _extract_kw(self, text: str) -> List[str]:
        return extract_kw_nouns(self._nlp, text)
