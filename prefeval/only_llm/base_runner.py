"""
Shared OnlyLLM utilities for PrefEval (single-chain, cumulative-checkpoint).

Adapted from exp_locomo/only_llm/base_runner.py:
- ContextTurn wraps PrefEval's Turn (role/utterance/conv_id/turn_id/...).
- Session header uses conv_id + virtual time (CONV_IDS_PER_DAY / MINUTES_PER_TURN).
- Persona/speaker name is intentionally hidden — turns render as "User: ..." / "Assistant: ...".
- Single QA prompt template, no category branching, no choice shuffle.
- 200-word answer cap (PrefEval spec §8).
- LLM call dir name follows PrefEval convention: `call_4_qa`.
"""

import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))
MODULE_SLUG = _MODULE_DIR.name

cfg = None  # type: ignore

from load_dataset import QAPair, Session, Turn


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# I/O HELPERS
# =============================================================================

def atomic_write_json(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_jsonl(path: Path, entry: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def write_retrieval_log(log_path: Path, entry: Dict) -> None:
    append_jsonl(log_path, entry)


def build_retrieval_log_entry(
    k: int,
    question_session: int,
    query: str,
    turns_in_prompt: int,
    sessions_in_prompt: int,
    token_budget: Optional[int],
) -> Dict:
    mode = "all_given" if cfg.HISTORY_ALL_GIVEN else "window"
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": "qa",
        "k": k,
        "question_session": question_session,
        "query": query,
        "memory_type": "context",
        "num_retrieved": turns_in_prompt,
        "module_specific": {
            "module": MODULE_SLUG,
            "mode": mode,
            "turns_in_prompt": turns_in_prompt,
            "sessions_in_prompt": sessions_in_prompt,
            "token_budget": token_budget,
            "max_context_turns": None if cfg.HISTORY_ALL_GIVEN else cfg.MAX_CONTEXT_TURNS,
        },
    }


# =============================================================================
# LLM CALL LOGGER
# =============================================================================

class LLMCallLogger:
    CALL_DIRS = ["call_4_qa"]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for call_dir in self.CALL_DIRS:
            (self._base / call_dir).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output if not isinstance(output, dict) else {
                key: value for key, value in output.items() if key != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# JSON PARSE HELPERS
# =============================================================================

def _escape_control_chars_in_strings(text: str) -> str:
    result = []
    in_string = False
    prev_backslash = False
    for ch in text:
        if prev_backslash:
            result.append(ch)
            prev_backslash = False
        elif ch == "\\" and in_string:
            result.append(ch)
            prev_backslash = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _parse_json_robust(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    if not text:
        raise json.JSONDecodeError("Empty response from model", "", 0)

    original_error = None

    def _try_all(raw_text: str):
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_escape_control_chars_in_strings(raw_text))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(raw_text))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(_escape_control_chars_in_strings(raw_text)))
        except json.JSONDecodeError:
            pass
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        original_error = exc

    result = _try_all(text)
    if result is not None:
        return result

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        result = _try_all(match.group())
        if result is not None:
            return result

    raise original_error


# =============================================================================
# QA PROMPT
# =============================================================================

QA_PROMPT = """\
Based on the conversation history:
{context}

Question from user: {question}

Answer the question based only on the information provided in the conversation history above.
Be concise (maximum 200 words)."""

QA_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


# =============================================================================
# VIRTUAL TIME
# =============================================================================

def format_virtual_time(conv_id: int, turn_id: int) -> str:
    """Render PrefEval (conv_id, turn_id) as a 'Day D, HH:MM' string.

    Mirrors `format_virtual_time` in amem/memory_layer.py.
    """
    day = conv_id // cfg.CONV_IDS_PER_DAY + 1
    minute_offset = turn_id * cfg.MINUTES_PER_TURN
    h, m = divmod(minute_offset, 60)
    return f"Day {day}, {h:02d}:{m:02d}"


# =============================================================================
# DIALOGUE CONTEXT
# =============================================================================

@dataclass
class ContextTurn:
    conv_id: int
    turn_id: int
    global_turn_id: int
    role: str
    utterance: str

    def session_header(self) -> str:
        return f"Conversation {self.conv_id} | {format_virtual_time(self.conv_id, 0)}"

    def turn_line(self) -> str:
        role_label = "User" if self.role == "user" else "Assistant"
        return f"{role_label}: {self.utterance}"

    def to_dict(self) -> Dict:
        return {
            "conv_id": self.conv_id,
            "turn_id": self.turn_id,
            "global_turn_id": self.global_turn_id,
            "role": self.role,
            "utterance": self.utterance,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ContextTurn":
        return cls(
            conv_id=int(data["conv_id"]),
            turn_id=int(data["turn_id"]),
            global_turn_id=int(data["global_turn_id"]),
            role=str(data["role"]),
            utterance=str(data["utterance"]),
        )


class DialogueContext:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._turns: List[ContextTurn] = []

    def add_turn(self, turn: Turn) -> None:
        if not cfg.HISTORY_ALL_GIVEN and cfg.MAX_CONTEXT_TURNS <= 0:
            self._turns = []
            return

        self._turns.append(ContextTurn(
            conv_id=turn.conv_id,
            turn_id=turn.turn_id,
            global_turn_id=turn.global_turn_id,
            role=turn.role,
            utterance=turn.utterance,
        ))
        if not cfg.HISTORY_ALL_GIVEN and len(self._turns) > cfg.MAX_CONTEXT_TURNS:
            self._turns = self._turns[-cfg.MAX_CONTEXT_TURNS:]

    def add_context_turn(self, ct: ContextTurn) -> None:
        """Append a pre-built ContextTurn (used on snapshot resume)."""
        if not cfg.HISTORY_ALL_GIVEN and cfg.MAX_CONTEXT_TURNS <= 0:
            self._turns = []
            return
        self._turns.append(ct)
        if not cfg.HISTORY_ALL_GIVEN and len(self._turns) > cfg.MAX_CONTEXT_TURNS:
            self._turns = self._turns[-cfg.MAX_CONTEXT_TURNS:]

    def clear(self) -> None:
        self._turns = []

    def __len__(self) -> int:
        if not cfg.HISTORY_ALL_GIVEN and cfg.MAX_CONTEXT_TURNS <= 0:
            return 0
        return len(self._turns)

    def to_dict_list(self) -> List[Dict]:
        return [ct.to_dict() for ct in self._turns]

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def format_for_prompt(self, token_budget: Optional[int]) -> Tuple[str, int, int]:
        if cfg.HISTORY_ALL_GIVEN:
            assert token_budget is not None, "token_budget required in all_given mode"
            selected_turns = self._select_suffix_with_budget(token_budget)
        else:
            selected_turns = self._select_window()

        if not selected_turns:
            return "No previous conversation.", 0, 0

        lines: List[str] = []
        last_conv_id: Optional[int] = None
        sessions_in_prompt = 0

        for item in selected_turns:
            if item.conv_id != last_conv_id:
                lines.append(item.session_header())
                last_conv_id = item.conv_id
                sessions_in_prompt += 1
            lines.append(item.turn_line())

        return "\n".join(lines), len(selected_turns), sessions_in_prompt

    def _select_window(self) -> List[ContextTurn]:
        if not self._turns:
            return []
        if cfg.MAX_CONTEXT_TURNS <= 0:
            return []
        return self._turns[-cfg.MAX_CONTEXT_TURNS:]

    def _encode_turn(self, item: ContextTurn, include_header: bool) -> Tuple[str, int]:
        lines = []
        if include_header:
            lines.append(item.session_header())
        lines.append(item.turn_line())
        text = "\n".join(lines)
        tokens = self._count_tokens(text)
        return text, tokens

    def _select_suffix_with_budget(self, token_budget: int) -> List[ContextTurn]:
        if not self._turns:
            return []

        n = len(self._turns)

        def enc(idx: int, include_header: Optional[bool] = None) -> Tuple[str, int]:
            if include_header is None:
                include_header = (
                    idx == 0
                    or self._turns[idx].conv_id != self._turns[idx - 1].conv_id
                )
            return self._encode_turn(self._turns[idx], include_header)

        sample_indices = sorted(set(
            max(0, min(n - 1, idx))
            for idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ))
        avg_tokens = sum(enc(idx)[1] for idx in sample_indices) / len(sample_indices)
        avg_tokens = max(avg_tokens, 1.0)

        rough_keep = max(1, int(token_budget / avg_tokens * 0.8))
        rough_start = max(0, n - rough_keep)

        def suffix_cost(start_idx: int) -> int:
            used_tokens = 0
            for idx in range(start_idx, n):
                include_header = idx == start_idx or self._turns[idx].conv_id != self._turns[idx - 1].conv_id
                _, tokens = enc(idx, include_header=include_header)
                used_tokens += tokens
            return used_tokens

        start_idx = rough_start
        used = suffix_cost(start_idx)

        while used > token_budget and start_idx < n - 1:
            start_idx += 1
            used = suffix_cost(start_idx)

        return self._turns[start_idx:]


# =============================================================================
# TOKEN HELPERS
# =============================================================================

def extract_token_info(response, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(response, dict) and isinstance(response.get("_usage"), dict):
        usage = response["_usage"]
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def usage_to_token_info(usage: Dict, model_path: str = "") -> Dict:
    return {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "model": model_path,
    }


def token_info_to_usage(token_info: Dict) -> Dict:
    return {
        "prompt_tokens": token_info.get("input", 0),
        "completion_tokens": token_info.get("output", 0),
    }


# =============================================================================
# ONLY-LLM RUNNER (single instance per chain)
# =============================================================================

class OnlyLLMRunner:
    QA_SYSTEM_PROMPT = (
        "You are a helpful assistant answering a question about a user "
        "based on their conversation history. "
        "Respond in JSON format with an 'answer' field."
    )

    def __init__(
        self,
        llm_client,
        tokenizer,
        model_path: str,
        max_model_len: Optional[int],
    ):
        self.client = llm_client
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.ctx = DialogueContext(tokenizer)
        self.llm_logger: Optional[LLMCallLogger] = None
        self._qa_parse_fallback_count = 0

    def set_llm_logger(self, llm_logger: Optional[LLMCallLogger]) -> None:
        self.llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Snapshot save / load (turn list serialization)
    # ------------------------------------------------------------------

    def save_snapshot(self, snap_dir: Path, k: int) -> None:
        snap_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "k": k,
            "num_turns": len(self.ctx),
            "turns": self.ctx.to_dict_list(),
            "timestamp": datetime.now().isoformat(),
        }
        atomic_write_json(snap_dir / "context.json", payload)

    def load_snapshot(self, snap_dir: Path) -> None:
        with open(snap_dir / "context.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.ctx.clear()
        for turn_dict in payload.get("turns", []):
            self.ctx.add_context_turn(ContextTurn.from_dict(turn_dict))

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _compute_context_budget(self, fixed_text: str) -> int:
        if self.max_model_len is None:
            return 0
        fixed_tokens = self._count_tokens(fixed_text)
        budget = (
            self.max_model_len
            - cfg.OUTPUT_TOKEN_RESERVE
            - cfg.CONTEXT_SAFETY_MARGIN
            - fixed_tokens
        )
        return max(budget, 0)

    def build_qa_job(self, qa: QAPair) -> Dict:
        if cfg.HISTORY_ALL_GIVEN:
            fixed_text = QA_PROMPT.format(context="", question=qa.question)
            token_budget = self._compute_context_budget(fixed_text)
        else:
            token_budget = None

        context_str, turns_in_prompt, sessions_in_prompt = self.ctx.format_for_prompt(token_budget)
        prompt = QA_PROMPT.format(context=context_str, question=qa.question)
        return {
            "prompt": prompt,
            "turns_in_prompt": turns_in_prompt,
            "sessions_in_prompt": sessions_in_prompt,
            "token_budget": token_budget,
        }

    # ------------------------------------------------------------------
    # Sequential QA fallback (used on batch failure)
    # ------------------------------------------------------------------

    def answer_qa_sequential(self, qa: QAPair) -> Tuple[Dict, Dict, bool]:
        """Generate one QA answer outside the batch path.

        Returns (parsed_result, usage_dict, already_logged_flag).
        """
        job = self.build_qa_job(qa)
        prompt = job["prompt"]

        fallback_used = False
        try:
            raw = self.client.generate(
                prompt=prompt,
                system_prompt=self.QA_SYSTEM_PROMPT,
                guided_json=QA_SCHEMA,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_TOKENS,
                json_retry=cfg.JSON_RETRY,
                return_usage=True,
            )
        except json.JSONDecodeError:
            fallback_used = True
            logger.warning("QA: guided_json failed; retrying without guided decoding")
            fallback = self.client.generate(
                prompt=prompt,
                system_prompt=self.QA_SYSTEM_PROMPT,
                guided_json=None,
                temperature=cfg.TEMPERATURE,
                max_tokens=cfg.MAX_TOKENS,
                json_retry=1,
                return_usage=True,
            )
            text = fallback.get("content", "") if isinstance(fallback, dict) else str(fallback)
            try:
                raw = _parse_json_robust(text)
            except json.JSONDecodeError:
                logger.warning("QA: fallback also failed to produce valid JSON; using empty answer")
                raw = {"answer": ""}
            raw["_usage"] = fallback.get("_usage", {}) if isinstance(fallback, dict) else {}

        if fallback_used:
            self._qa_parse_fallback_count += 1

        if self.llm_logger is not None:
            self.llm_logger.log("call_4_qa", "", prompt, raw)

        usage = raw.pop("_usage", {}) if isinstance(raw, dict) else {}
        return raw, usage, True


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def create_llm_client(
    model_path: str,
    tensor_parallel: int,
    gpu_memory: float,
    max_model_len: Optional[int],
):
    from llm_client import create_llm_client as _create

    engine = cfg.LLM_ENGINE
    if engine == "vllm":
        vllm_kwargs = dict(
            engine="vllm",
            model_path=model_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_memory,
            download_dir=None,
        )
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len
        return _create(**vllm_kwargs)
    if engine == "together":
        return _create(engine="together", **cfg.TOGETHER_CONFIG)
    if engine == "openai":
        return _create(engine="openai", **cfg.OPENAI_CONFIG)
    raise ValueError(f"Unknown LLM engine: {engine}")
