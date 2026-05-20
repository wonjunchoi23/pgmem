"""
Shared OnlyLLM utilities used by both sequential and batched runners.
PersonaMem variant.
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

from load_dataset import (
    PersonaMemContext,
    PersonaMemMessage,
    PersonaMemQAPair,
    load_personamem_dataset,
)


logger: logging.Logger = logging.getLogger(__name__)


def load_existing_results(results_file: Path) -> List[Dict]:
    if not results_file.exists():
        return []
    try:
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results from {results_file}")
        return results
    except Exception as exc:
        logger.warning(f"Failed to load existing results ({exc}). Starting fresh.")
        return []


def save_results(results_file: Path, results: List[Dict]) -> None:
    atomic_write_json(results_file, results)


def atomic_write_json(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_retrieval_log(log_path: Path, entry: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    context_index: int,
    query: str,
    messages_in_prompt: int,
    blocks_in_prompt: int,
    token_budget: Optional[int],
) -> Dict:
    mode = "all_given" if cfg.HISTORY_ALL_GIVEN else "window"
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": "qa",
        "context_index": context_index,
        "block_idx": -1,
        "local_msg_idx": -1,
        "query": query,
        "memory_type": "context",
        "num_retrieved": messages_in_prompt,
        "module_specific": {
            "module": MODULE_SLUG,
            "mode": mode,
            "messages_in_prompt": messages_in_prompt,
            "blocks_in_prompt": blocks_in_prompt,
            "token_budget": token_budget,
            "max_context_messages": None if cfg.HISTORY_ALL_GIVEN else cfg.MAX_CONTEXT_MESSAGES,
        },
    }


class LLMCallLogger:
    CALL_DIRS = ["call_1_qa"]

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


QA_PROMPT_MULTICHOICE = """\
You are a helpful assistant. Answer the multiple-choice question based only on the conversation history provided.

Conversation history:
{context}

Question: {question}

Options:
{options_text}

Choose the single best answer (a, b, c, or d).
Answer in JSON format:
{{"answer": "<a | b | c | d>"}}"""


QA_SCHEMA_MULTICHOICE = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["a", "b", "c", "d"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass
class ContextMessage:
    block_idx: int
    message: PersonaMemMessage

    def block_header(self) -> str:
        return f"Block {self.block_idx}"

    def message_line(self) -> str:
        # content already contains "User: ..." / "Assistant: ..." prefix
        return self.message.content


class DialogueContext:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._messages: List[ContextMessage] = []

    def add_message(self, msg: PersonaMemMessage) -> None:
        if not cfg.HISTORY_ALL_GIVEN and cfg.MAX_CONTEXT_MESSAGES <= 0:
            self._messages = []
            return

        self._messages.append(ContextMessage(block_idx=msg.block_idx, message=msg))
        if not cfg.HISTORY_ALL_GIVEN and len(self._messages) > cfg.MAX_CONTEXT_MESSAGES:
            self._messages = self._messages[-cfg.MAX_CONTEXT_MESSAGES :]

    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        if not cfg.HISTORY_ALL_GIVEN and cfg.MAX_CONTEXT_MESSAGES <= 0:
            return 0
        return len(self._messages)

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def format_for_prompt(self, token_budget: Optional[int]) -> Tuple[str, int, int]:
        if cfg.HISTORY_ALL_GIVEN:
            assert token_budget is not None, "token_budget required in all_given mode"
            selected = self._select_suffix_with_budget(token_budget)
        else:
            selected = self._select_window()

        if not selected:
            return "No previous conversation.", 0, 0

        lines: List[str] = []
        last_block_idx: Optional[int] = None
        blocks_in_prompt = 0

        for item in selected:
            if item.block_idx != last_block_idx:
                lines.append(item.block_header())
                last_block_idx = item.block_idx
                blocks_in_prompt += 1
            lines.append(item.message_line())

        return "\n".join(lines), len(selected), blocks_in_prompt

    def _select_window(self) -> List[ContextMessage]:
        if not self._messages:
            return []
        if cfg.MAX_CONTEXT_MESSAGES <= 0:
            return []
        return self._messages[-cfg.MAX_CONTEXT_MESSAGES :]

    def _encode_item(self, item: ContextMessage, include_header: bool) -> Tuple[str, int]:
        lines = []
        if include_header:
            lines.append(item.block_header())
        lines.append(item.message_line())
        text = "\n".join(lines)
        tokens = self._count_tokens(text)
        return text, tokens

    def _select_suffix_with_budget(self, token_budget: int) -> List[ContextMessage]:
        if not self._messages:
            return []

        n = len(self._messages)

        def enc(idx: int, include_header: Optional[bool] = None) -> Tuple[str, int]:
            if include_header is None:
                include_header = (
                    idx == 0
                    or self._messages[idx].block_idx != self._messages[idx - 1].block_idx
                )
            return self._encode_item(self._messages[idx], include_header)

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
                include_header = (
                    idx == start_idx
                    or self._messages[idx].block_idx != self._messages[idx - 1].block_idx
                )
                _, tokens = enc(idx, include_header=include_header)
                used_tokens += tokens
            return used_tokens

        start_idx = rough_start
        used = suffix_cost(start_idx)

        while used > token_budget and start_idx < n - 1:
            start_idx += 1
            used = suffix_cost(start_idx)

        return self._messages[start_idx:]


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


def build_memory_at_qa_start() -> Dict:
    return {
        "num_memories": 0,
        "total_content_tokens": 0,
    }


def normalize_answer(raw: str) -> str:
    label = (raw or "").strip().lower()
    return label if label in ("a", "b", "c", "d") else "unknown"


class OnlyLLMRunner:
    QA_SYSTEM_PROMPT = None  # instruction is in the user prompt

    def __init__(self, llm_client, tokenizer, model_path: str, max_model_len: Optional[int]):
        self.client = llm_client
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.ctx = DialogueContext(tokenizer)
        self.llm_logger: Optional[LLMCallLogger] = None
        self._qa_parse_fallback_count = 0

    def set_llm_logger(self, llm_logger: Optional[LLMCallLogger]) -> None:
        self.llm_logger = llm_logger

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _compute_context_budget(self, fixed_text: str) -> int:
        fixed_tokens = self._count_tokens(fixed_text)
        budget = (
            self.max_model_len
            - cfg.OUTPUT_TOKEN_RESERVE
            - cfg.CONTEXT_SAFETY_MARGIN
            - fixed_tokens
        )
        return max(budget, 0)

    def _build_qa_prompt(self, qa: PersonaMemQAPair, context_str: str) -> str:
        options_text = "\n".join(qa.all_options)
        return QA_PROMPT_MULTICHOICE.format(
            context=context_str,
            question=qa.question,
            options_text=options_text,
        )

    def build_qa_job(self, qa: PersonaMemQAPair) -> Dict:
        if cfg.HISTORY_ALL_GIVEN:
            fixed_text = self._build_qa_prompt(qa, context_str="")
            token_budget = self._compute_context_budget(fixed_text)
        else:
            token_budget = None

        context_str, messages_in_prompt, blocks_in_prompt = self.ctx.format_for_prompt(token_budget)
        prompt = self._build_qa_prompt(qa, context_str=context_str)
        return {
            "prompt": prompt,
            "temperature": cfg.TEMPERATURE,
            "messages_in_prompt": messages_in_prompt,
            "blocks_in_prompt": blocks_in_prompt,
            "token_budget": token_budget,
        }

    def _generate_qa_raw(self, prompt: str, temperature: float) -> Tuple[Dict, bool]:
        fallback_used = False
        try:
            raw = self.client.generate(
                prompt=prompt,
                system_prompt=self.QA_SYSTEM_PROMPT,
                guided_json=QA_SCHEMA_MULTICHOICE,
                temperature=temperature,
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
                temperature=temperature,
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
        return raw, fallback_used

    def answer_qa(self, qa: PersonaMemQAPair, include_prompt: bool = False) -> Dict:
        job = self.build_qa_job(qa)
        prompt = job["prompt"]
        temperature = job["temperature"]
        raw, fallback_used = self._generate_qa_raw(prompt, temperature)

        if fallback_used:
            self._qa_parse_fallback_count += 1

        if self.llm_logger is not None:
            self.llm_logger.log("call_1_qa", "", prompt, raw)

        answer_raw = raw.get("answer", "") if isinstance(raw, dict) else ""
        result = {
            "generated_answer": normalize_answer(answer_raw),
            "qa_tokens": extract_token_info(raw, self.model_path),
            "_messages_in_prompt": job["messages_in_prompt"],
            "_blocks_in_prompt": job["blocks_in_prompt"],
            "_token_budget": job["token_budget"],
        }
        if include_prompt:
            result["_prompt"] = prompt
        return result


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
