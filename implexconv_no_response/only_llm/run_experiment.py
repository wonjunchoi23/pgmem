"""
OnlyLLM Experiment Runner — ImplexConv (Batched Main Variant, QA-Only)

Runs multiple sessions in parallel within a single GPU by batching their QA
LLM calls together. Phase 1 performs no model inference, so sessions are
interleaved turn-by-turn only to preserve the original session-local context
construction order while improving throughput at the batch level.

Context strategy is still controlled by HISTORY_ALL_GIVEN:

  HISTORY_ALL_GIVEN = True   →  All accumulated turns provided as context,
                                trimmed from the oldest end when the token
                                budget is exceeded.
                                Trimming is two-phase:
                                  1. Rough cut  – bulk removal based on avg
                                                  tokens/turn estimate
                                  2. Fine trim  – one turn at a time until
                                                  the budget is satisfied

  HISTORY_ALL_GIVEN = False  →  Sliding window: only the last
                                MAX_CONTEXT_TURNS turns are kept.

Batch execution flow:
  For each batch of N sessions (processed simultaneously):

  Phase 1 — Context Construction (interleaved across sessions):
    For turn_idx = 0, 1, ..., max_turns:
      1. Write retrieval log for the current context state
      2. Add GT user/assistant turns to the per-session context

  Phase 2 — QA Answering (all sessions batched together):
    1. Build QA prompts for all sessions
    2. [BATCH] vLLM generate in QA_BATCH_SIZE chunks
    3. Distribute results back to each session

Checkpoint format (new — set-based):
  {"completed_session_ids": [0, 1, 2, ...], ...}
  Old format {"last_completed_session_index": N} is auto-converted using the
  target session range.

Usage:
    python run_experiment.py \\
        --start-session 0 --end-session 99 \\
        --subset opposed \\
        --model Qwen/Qwen3-1.7B \\
        --tensor-parallel 1 --gpu-memory 0.9 \\
        --batch-size 4 \\
        --config config_lb
"""

import os
import re
import sys
import json
import logging
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from tqdm import tqdm

# Suppress noisy logs before any imports
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(_MODULE_DIR))

# cfg is loaded dynamically in main() based on --config argument.
# All references to cfg inside functions are resolved at call time, not import time.
cfg = None  # type: ignore

from load_dataset import (
    load_implexconv_dataset,
    Session,
    Turn,
    QAPair,
    compute_virtual_minutes,
    format_elapsed_label,
)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if cfg.LOG_TO_FILE:
        log_file = log_dir / f"onlyllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT & RESULTS I/O
# =============================================================================

def load_checkpoint(checkpoint_file: Path, target_session_ids: Optional[List[int]] = None) -> Set[int]:
    """Load completed session IDs from checkpoint.

    Supports both the new set-based format and the old sequential format
    storing last_completed_session_index. When old format is detected, the
    index is mapped onto the provided target_session_ids range.
    """
    if not checkpoint_file.exists():
        return set()
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        if "completed_session_ids" in data:
            return set(data["completed_session_ids"])

        last = data.get("last_completed_session_index")
        if last is None:
            return set()

        if target_session_ids is None:
            return set(range(last + 1))

        capped = max(0, min(last + 1, len(target_session_ids)))
        return set(target_session_ids[:capped])
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def save_checkpoint(checkpoint_file: Path, completed_ids: Set[int],
                    model_path: str, subset: str,
                    start_session: int, end_session: int,
                    config_name: str = "config"):
    data = {
        "completed_session_ids": sorted(completed_ids),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_name": config_name,
            "model": model_path,
            "subset": subset,
            "start_session": start_session,
            "end_session": end_session,
        },
    }
    _atomic_write(checkpoint_file, data)


def load_existing_results(results_file: Path) -> List[Dict]:
    if not results_file.exists():
        return []
    try:
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results from {results_file}")
        return results
    except Exception as e:
        logger.warning(f"Failed to load existing results ({e}). Starting fresh.")
        return []


def save_results(results_file: Path, results: List[Dict]):
    _atomic_write(results_file, results)


def _atomic_write(path: Path, data):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# =============================================================================
# RETRIEVAL LOGGING
# =============================================================================

def write_retrieval_log(log_path: Path, entry: Dict):
    """Append one retrieval log entry (JSONL) to the session log file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_retrieval_log_entry(
    phase: str,
    session_id: int,
    conv_id: int,
    global_turn: int,
    query: str,
    turns_in_prompt: int,
    # all_given mode extras
    total_accumulated_turns: Optional[int] = None,
    token_budget: Optional[int] = None,
    # window mode extras
    max_context_turns: Optional[int] = None,
) -> Dict:
    """Build a retrieval log entry for OnlyLLM (no memory store)."""
    mode = "all_given" if cfg.HISTORY_ALL_GIVEN else "window"
    entry = {
        "timestamp":     datetime.now().isoformat(),
        "phase":         phase,
        "session_id":    session_id,
        "conv_id":       conv_id,
        "global_turn":   global_turn,
        "query":         query[:500],
        "memory_type":   "context",
        "num_retrieved": turns_in_prompt,
        "module_specific": {
            "module":          "only_llm",
            "mode":            mode,
            "turns_in_prompt": turns_in_prompt,
        },
    }
    if mode == "all_given":
        entry["module_specific"]["total_accumulated_turns"] = total_accumulated_turns
        entry["module_specific"]["token_budget"] = token_budget
    else:
        entry["module_specific"]["max_context_turns"] = max_context_turns
    return entry


# =============================================================================
# LLM CALL LOGGING
# =============================================================================

class LLMCallLogger:
    """
    Logs all LLM calls (input + output) to call-type-specific folders as JSONL files.

    Folder structure:
        {base_dir}/call_5_qa/calls.jsonl
    """

    CALL_DIRS = ["call_5_qa"]

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir)
        for d in self.CALL_DIRS:
            (self._base / d).mkdir(parents=True, exist_ok=True)

    def log(self, call_type: str, system_prompt: str, user_prompt: str, output) -> None:
        entry = {
            "timestamp":     datetime.now().isoformat(),
            "call_type":     call_type,
            "system_prompt": system_prompt,
            "user_prompt":   user_prompt,
            "output":        output if not isinstance(output, dict) else {
                k: v for k, v in output.items() if k != "_usage"
            },
        }
        log_file = self._base / call_type / "calls.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# JSON PARSING UTILITIES
# =============================================================================

def _escape_control_chars_in_strings(text: str) -> str:
    """State-machine: escape literal newlines/tabs inside JSON string values.

    xgrammar occasionally emits raw control characters inside string tokens
    instead of their escaped forms, producing invalid JSON.  This function
    fixes only characters that are *inside* a string value without touching
    the structural whitespace between keys/values.
    """
    result = []
    in_string = False
    prev_backslash = False
    for ch in text:
        if prev_backslash:
            result.append(ch)
            prev_backslash = False
        elif ch == '\\' and in_string:
            result.append(ch)
            prev_backslash = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
    return ''.join(result)


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM mistake)."""
    return re.sub(r',(\s*[}\]])', r'\1', text)


def _parse_json_robust(text: str) -> dict:
    """Parse JSON from raw LLM text with multiple fallback strategies.

    Strategy order:
    1. Direct json.loads after stripping markdown fences.
    2. Escape literal control characters inside string values.
    3. Remove trailing commas.
    4. Both fixes combined.
    5. Extract the first {...} block, then apply strategies 1–4.
    """
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

    def _try_all(t: str):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_escape_control_chars_in_strings(t))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(t))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(_escape_control_chars_in_strings(t)))
        except json.JSONDecodeError:
            pass
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        original_error = e

    result = _try_all(text)
    if result is not None:
        return result

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        result = _try_all(match.group())
        if result is not None:
            return result

    raise original_error


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# Used in Phase 2 (LLM IS called)
QA_PROMPT_OPPOSED = """\
You are a helpful assistant. Answer the question based only on the conversation history provided. Be concise (max 100 words).

Conversation history:
{context}

Question: {question}

Answer in JSON format:
{{"answer": "<your answer here>"}}"""

QA_SCHEMA_OPPOSED = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

QA_PROMPT_SUPPORTIVE = """\
You are a helpful assistant. Answer the yes/no question based only on the conversation history provided. You MUST answer with exactly one of: "yes" or "no".

Conversation history:
{context}

Question: {question}

Answer in JSON format using exactly one of: "yes", "no"
{{"answer": "<yes | no>"}}"""

QA_SCHEMA_SUPPORTIVE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]}
    },
    "required": ["answer"],
    "additionalProperties": False,
}


# =============================================================================
# DIALOGUE CONTEXT
# =============================================================================

class DialogueContext:
    """
    Manages dialogue history for the OnlyLLM experiment.

    HISTORY_ALL_GIVEN = True  (all_given mode)
        All past turns are accumulated without limit.
        When formatting, turns are trimmed from the oldest end to fit within
        a given token budget using a two-phase strategy:
          1. Rough cut  – sample a few turns to estimate avg tokens/turn,
                          then bulk-remove more turns than the estimate
                          (overshoot to err on the safe side).
          2. Fine trim  – compute exact token counts from the rough-cut
                          start; remove one turn at a time until the
                          remaining turns fit within the budget.

    HISTORY_ALL_GIVEN = False  (window mode)
        Only the last MAX_CONTEXT_TURNS turns are kept in memory.
        All stored turns are always included in the formatted context
        (no token-budget trimming).

    Rules (both modes):
        • All assistant turns stored must be GT agent responses.
        • QA exchanges must never be added.
        • Context is cleared between sessions.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._turns: List[Turn] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, turn: Turn) -> None:
        if not cfg.HISTORY_ALL_GIVEN and cfg.MAX_CONTEXT_TURNS <= 0:
            self._turns = []
            return

        self._turns.append(turn)
        if not cfg.HISTORY_ALL_GIVEN:
            if len(self._turns) > cfg.MAX_CONTEXT_TURNS:
                self._turns = self._turns[-cfg.MAX_CONTEXT_TURNS:]

    def clear(self) -> None:
        self._turns = []

    def __len__(self) -> int:
        if not cfg.HISTORY_ALL_GIVEN and cfg.MAX_CONTEXT_TURNS <= 0:
            return 0
        return len(self._turns)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_prompt(
        self,
        token_budget: Optional[int],
        current_vm: float,
        conv_ids_per_day: int,
        minutes_per_turn: int,
    ) -> Tuple[str, int]:
        """
        Format stored turns as a context string for use in a prompt.

        Args:
            token_budget: Maximum token count for the returned string.
                          Used only in all_given mode; ignored in window mode.
            current_vm:   Virtual time (minutes) of the current turn, used
                          as reference for elapsed-time labels.
            conv_ids_per_day, minutes_per_turn: Time-model parameters.

        Returns:
            (context_str, num_turns_included)
        """
        if cfg.HISTORY_ALL_GIVEN:
            assert token_budget is not None, "token_budget required in all_given mode"
            return self._format_with_budget(
                token_budget, current_vm, conv_ids_per_day, minutes_per_turn
            )
        else:
            return self._format_window(current_vm, conv_ids_per_day, minutes_per_turn)

    # ------------------------------------------------------------------
    # Window mode
    # ------------------------------------------------------------------

    def _format_window(
        self,
        current_vm: float,
        conv_ids_per_day: int,
        minutes_per_turn: int,
    ) -> Tuple[str, int]:
        if cfg.MAX_CONTEXT_TURNS <= 0:
            return "No previous conversation.", 0
        if not self._turns:
            return "No previous conversation.", 0
        lines = []
        for t in self._turns:
            turn_vm = compute_virtual_minutes(t.conv_id, t.turn_id, conv_ids_per_day, minutes_per_turn)
            elapsed = current_vm - turn_vm
            label = format_elapsed_label(elapsed)
            lines.append(t.to_message(elapsed_label=label))
        return "\n".join(lines), len(self._turns)

    # ------------------------------------------------------------------
    # All-given mode (token-budget trimming)
    # ------------------------------------------------------------------

    def _encode_turn(
        self,
        t: Turn,
        current_vm: float,
        conv_ids_per_day: int,
        minutes_per_turn: int,
    ) -> Tuple[str, int]:
        """Return (formatted_text, token_count) for a single turn."""
        turn_vm = compute_virtual_minutes(t.conv_id, t.turn_id, conv_ids_per_day, minutes_per_turn)
        elapsed = current_vm - turn_vm
        label = format_elapsed_label(elapsed)
        text = t.to_message(elapsed_label=label)
        tokens = len(self._tokenizer.encode(text, add_special_tokens=False))
        return text, tokens

    def _format_with_budget(
        self,
        token_budget: int,
        current_vm: float,
        conv_ids_per_day: int,
        minutes_per_turn: int,
    ) -> Tuple[str, int]:
        """
        Return (context_str, num_turns_included) trimmed to fit token_budget.

        Two-phase trimming strategy:
          Phase 1 – Rough cut
            • Sample a small number of turns spread across the accumulated
              history and compute their average token count.
            • Estimate how many turns fit in the budget; apply an overshoot
              factor of 0.8 (keep 80% of the estimated count) so that the
              rough cut removes more than strictly necessary.
            • Jump to that start index in one step.
          Phase 2 – Fine trim
            • Compute exact token counts for all turns from the rough-cut
              start onward.
            • Remove turns one at a time from the oldest end until the
              remaining turns fit within the budget.
        """
        if not self._turns:
            return "No previous conversation.", 0

        n = len(self._turns)

        def enc(t: Turn) -> Tuple[str, int]:
            return self._encode_turn(t, current_vm, conv_ids_per_day, minutes_per_turn)

        # ── Phase 1: rough cut ────────────────────────────────────────
        # Sample 5 turns spread evenly across the history
        sample_indices = sorted(set(
            max(0, min(n - 1, i))
            for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ))
        avg_tokens = (
            sum(enc(self._turns[i])[1] for i in sample_indices) / len(sample_indices)
        )
        avg_tokens = max(avg_tokens, 1.0)

        # Estimate how many turns fit; multiply by 0.8 to overshoot removal
        rough_keep = max(1, int(token_budget / avg_tokens * 0.8))
        rough_start = max(0, n - rough_keep)

        # ── Phase 2: fine trim ────────────────────────────────────────
        # Pre-compute (text, tokens) for turns[rough_start:] in one pass
        entries: List[Tuple[str, int]] = [enc(t) for t in self._turns[rough_start:]]
        used = sum(c for _, c in entries)

        # Remove one turn at a time from the oldest end while over budget
        trim = 0
        while used > token_budget and trim < len(entries) - 1:
            used -= entries[trim][1]
            trim += 1

        selected = entries[trim:]
        if not selected:
            return "No previous conversation.", 0

        return "\n".join(text for text, _ in selected), len(selected)


# =============================================================================
# TOKEN UTILITIES
# =============================================================================

def extract_token_info(response, model_path: str = "") -> Dict:
    token_info = {"input": 0, "output": 0, "model": model_path}
    if isinstance(response, dict) and "_usage" in response:
        usage = response["_usage"] or {}
        token_info["input"] = usage.get("prompt_tokens", 0)
        token_info["output"] = usage.get("completion_tokens", 0)
    return token_info


def build_memory_at_qa_start() -> Dict:
    """OnlyLLM has no memory store; keep a stable zeroed schema."""
    return {
        "num_memories": 0,
        "total_content_tokens": 0,
    }


def generate_json_with_fallback(client, prompt: str, schema: Dict) -> Tuple[Dict, bool]:
    """Run one JSON generation and report whether fallback parsing was used."""
    fallback_used = False
    try:
        raw = client.generate(
            prompt=prompt,
            guided_json=schema,
            temperature=cfg.TEMPERATURE,
            max_tokens=cfg.MAX_TOKENS,
            json_retry=cfg.JSON_RETRY,
            return_usage=True,
        )
    except json.JSONDecodeError:
        fallback_used = True
        logger.warning("QA: guided_json failed; retrying without guided decoding")
        fallback = client.generate(
            prompt=prompt,
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
        raw["_usage"] = fallback.get("_usage") if isinstance(fallback, dict) else None
    return raw, fallback_used


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class OnlyLLMRunner:
    """
    Runs the OnlyLLM QA-only experiment on a single session.

    Phase 1 – Context Construction:
        For each (user_turn, assistant_turn) in the session:
        1. Write retrieval/context log for the current state
        2. Add user_turn + GT assistant_turn to context

    Phase 2 – QA Answering:
        For each QA pair (context is FROZEN from Phase 1):
        1. Format context → build QA prompt
        2. Call LLM → record answer
        3. Track tokens and fallback counts
    """

    def __init__(self, llm_client, tokenizer, subset: str, model_path: str,
                 max_model_len: Optional[int], config_metadata: Optional[Dict] = None):
        self.client = llm_client
        self.tokenizer = tokenizer
        self.subset = subset
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.config_metadata = config_metadata or {}
        self.ctx = DialogueContext(tokenizer)
        self.llm_logger: Optional[LLMCallLogger] = None
        self._qa_parse_fallback_count = 0

    def set_llm_logger(self, llm_logger: Optional[LLMCallLogger]) -> None:
        self.llm_logger = llm_logger

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _compute_context_budget(self, fixed_text: str) -> int:
        """
        Available token budget for the context string.

        Budget = max_model_len - OUTPUT_TOKEN_RESERVE - CONTEXT_SAFETY_MARGIN
                 - tokens(fixed_text)

        Only used in all_given mode where max_model_len is defined.
        """
        fixed_tokens = self._count_tokens(fixed_text)
        budget = (
            self.max_model_len
            - cfg.OUTPUT_TOKEN_RESERVE
            - cfg.CONTEXT_SAFETY_MARGIN
            - fixed_tokens
        )
        return max(budget, 0)

    # ------------------------------------------------------------------
    # Phase 2: QA answering
    # ------------------------------------------------------------------

    def build_qa_prompt(self, qa: QAPair) -> Tuple[str, Dict, int, Optional[int]]:
        """Build the QA prompt and metadata from the current frozen context."""
        # Reference time = just after the last stored turn
        if self.ctx._turns:
            last = self.ctx._turns[-1]
            current_vm = compute_virtual_minutes(
                last.conv_id, last.turn_id,
                cfg.CONV_IDS_PER_DAY, cfg.MINUTES_PER_TURN,
            ) + cfg.MINUTES_PER_TURN
        else:
            current_vm = 0.0

        if cfg.HISTORY_ALL_GIVEN:
            if self.subset == "supportive":
                fixed_text = QA_PROMPT_SUPPORTIVE.format(context="", question=qa.question)
            else:
                fixed_text = QA_PROMPT_OPPOSED.format(context="", question=qa.question)
            token_budget = self._compute_context_budget(fixed_text)
        else:
            token_budget = None

        context_str, turns_in_prompt = self.ctx.format_for_prompt(
            token_budget, current_vm, cfg.CONV_IDS_PER_DAY, cfg.MINUTES_PER_TURN,
        )

        if self.subset == "supportive":
            prompt = QA_PROMPT_SUPPORTIVE.format(context=context_str, question=qa.question)
            schema = QA_SCHEMA_SUPPORTIVE
        else:
            prompt = QA_PROMPT_OPPOSED.format(context=context_str, question=qa.question)
            schema = QA_SCHEMA_OPPOSED

        return prompt, schema, turns_in_prompt, token_budget

    def build_qa_result(
        self,
        raw,
        qa_tokens: Dict,
        turns_in_prompt: int,
        token_budget: Optional[int],
    ) -> Dict:
        """Convert one raw QA generation into the standard result payload."""
        answer = raw.get("answer", "") if isinstance(raw, dict) else ""
        if self.subset == "supportive":
            label = answer.strip().lower()
            if label not in ("yes", "no"):
                label = "unknown"
            answer = label

        return {
            "generated_answer": answer,
            "qa_tokens": qa_tokens,
            "_turns_in_prompt": turns_in_prompt,
            "_token_budget": token_budget,
        }

    def answer_qa(self, qa: QAPair) -> Tuple[Dict, str]:
        """
        Answer a QA pair using the current frozen context.

        Returns:
            (result_dict, prompt_snapshot)
            result_dict keys: generated_answer, qa_tokens,
                              _turns_in_prompt, _token_budget
        """
        prompt, schema, turns_in_prompt, token_budget = self.build_qa_prompt(qa)

        raw, fallback_used = generate_json_with_fallback(self.client, prompt, schema)
        if fallback_used:
            self._qa_parse_fallback_count += 1
        token_info = extract_token_info(raw, self.model_path)

        if self.llm_logger is not None:
            self.llm_logger.log("call_5_qa", "", prompt, raw)

        result = self.build_qa_result(
            raw=raw,
            qa_tokens=token_info,
            turns_in_prompt=turns_in_prompt,
            token_budget=token_budget,
        )
        return result, prompt

    # ------------------------------------------------------------------
    # Session runner
    # ------------------------------------------------------------------

    def run_session(self, session: Session, retrieval_log_path: Path) -> Dict:
        """
        Run Phases 1 and 2 on a single session.

        Returns a session result dict matching the QA-only output schema.
        """
        logger.info(f"{'='*60}")
        logger.info(f"Session {session.session_id}  "
                    f"({len(session.get_turn_pairs())} turn pairs, "
                    f"{len(session.qa)} QA)  "
                    f"[mode={'all_given' if cfg.HISTORY_ALL_GIVEN else 'window'}]")
        logger.info(f"{'='*60}")

        self.ctx.clear()
        self._qa_parse_fallback_count = 0

        # ── Phase 1: context construction ─────────────────────────────
        phase1_global_turn = 0
        turn_pairs = session.get_turn_pairs()
        for user_turn, assistant_turn in tqdm(turn_pairs, desc=f"S{session.session_id} Phase1"):
            log_entry = build_retrieval_log_entry(
                phase="context_construction",
                session_id=session.session_id,
                conv_id=user_turn.conv_id,
                global_turn=phase1_global_turn,
                query=user_turn.utterance,
                turns_in_prompt=len(self.ctx),
                total_accumulated_turns=len(self.ctx) if cfg.HISTORY_ALL_GIVEN else None,
                token_budget=None,
                max_context_turns=cfg.MAX_CONTEXT_TURNS if not cfg.HISTORY_ALL_GIVEN else None,
            )
            write_retrieval_log(retrieval_log_path, log_entry)
            phase1_global_turn += 1

            # Add GT turns to context (NOT any generated response)
            gt_response = assistant_turn.utterance if assistant_turn else ""
            self.ctx.add(user_turn)
            if assistant_turn:
                gt_turn = Turn(
                    session_id=assistant_turn.session_id,
                    conv_id=assistant_turn.conv_id,
                    turn_id=assistant_turn.turn_id,
                    global_turn_id=assistant_turn.global_turn_id,
                    role="assistant",
                    utterance=gt_response,
                )
                self.ctx.add(gt_turn)

        logger.info(f"Phase 1 done: {len(turn_pairs)} turns processed")

        # ── Phase 2: QA answering ─────────────────────────────────────
        qa_results: List[Dict] = []
        total_qa_input = total_qa_output = 0
        num_qa_calls = 0

        for qa in tqdm(session.qa, desc=f"S{session.session_id} Phase2"):
            qa_gen, _ = self.answer_qa(qa)

            record = {
                "question": qa.question,
                "generated_answer": qa_gen["generated_answer"],
                "ground_truth_answer": qa.answer,
                "retrieved_memories": [],   # OnlyLLM has no memory store
                "qa_tokens": qa_gen["qa_tokens"],
            }
            qa_results.append(record)

            tok = qa_gen["qa_tokens"]
            total_qa_input += tok.get("input", 0)
            total_qa_output += tok.get("output", 0)
            num_qa_calls += 1

            # Retrieval log
            log_entry = build_retrieval_log_entry(
                phase="qa",
                session_id=session.session_id,
                conv_id=-1,
                global_turn=-1,
                query=qa.question,
                turns_in_prompt=qa_gen["_turns_in_prompt"],
                total_accumulated_turns=len(self.ctx) if cfg.HISTORY_ALL_GIVEN else None,
                token_budget=qa_gen["_token_budget"],
                max_context_turns=cfg.MAX_CONTEXT_TURNS if not cfg.HISTORY_ALL_GIVEN else None,
            )
            write_retrieval_log(retrieval_log_path, log_entry)

        logger.info(f"Phase 2 done: {len(qa_results)} QA answers")

        # ── Aggregate ─────────────────────────────────────────────────
        qa_call_stats = {
            "input": total_qa_input,
            "output": total_qa_output,
            "llm_calls": num_qa_calls,
            "parse_fallback_count": self._qa_parse_fallback_count,
        }
        token_stats = {
            "call_5_qa": qa_call_stats,
            "total_input": total_qa_input,
            "total_output": total_qa_output,
            "total_llm_calls": num_qa_calls,
        }

        return {
            "session_id": session.session_id,
            "config_metadata": self.config_metadata,
            "memory_at_qa_start": build_memory_at_qa_start(),
            "qa_results": qa_results,
            "token_statistics": token_stats,
            "memory_snapshot_path": None,
        }


# =============================================================================
# BATCHED EXPERIMENT RUNNER
# =============================================================================

class BatchedOnlyLLMRunner:
    """Processes multiple OnlyLLM sessions in parallel with batched QA calls."""

    def __init__(self, llm_client, tokenizer, subset: str, model_path: str,
                 max_model_len: Optional[int], config_metadata: Optional[Dict] = None):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.subset = subset
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.config_metadata = config_metadata or {}

    def run_batch(
        self,
        sessions: List[Session],
        retrieval_log_paths: List[Path],
        llm_call_log_dirs: List[Path],
    ) -> List[Dict]:
        runners = [
            OnlyLLMRunner(
                self.llm_client,
                tokenizer=self.tokenizer,
                subset=self.subset,
                model_path=self.model_path,
                max_model_len=self.max_model_len,
                config_metadata=self.config_metadata,
            )
            for _ in sessions
        ]

        for runner, llm_call_log_dir in zip(runners, llm_call_log_dirs):
            if cfg.ENABLE_LLM_CALL_LOGGING:
                runner.set_llm_logger(LLMCallLogger(llm_call_log_dir))

        for session in sessions:
            logger.info(f"{'='*60}")
            logger.info(
                f"Session {session.session_id}  "
                f"({len(session.get_turn_pairs())} turn pairs, "
                f"{len(session.qa)} QA)  "
                f"[mode={'all_given' if cfg.HISTORY_ALL_GIVEN else 'window'}]"
            )
            logger.info(f"{'='*60}")

        self._run_phase1_batched(sessions, runners, retrieval_log_paths)
        qa_results_list, phase2_stats = self._run_phase2_batched(
            sessions, runners, retrieval_log_paths
        )

        results = []
        for i, session in enumerate(sessions):
            p2 = phase2_stats[i]
            results.append({
                "session_id": session.session_id,
                "config_metadata": self.config_metadata,
                "memory_at_qa_start": build_memory_at_qa_start(),
                "qa_results": qa_results_list[i],
                "token_statistics": {
                    "call_5_qa": p2["call_5_qa"],
                    "total_input": p2["call_5_qa"]["input"],
                    "total_output": p2["call_5_qa"]["output"],
                    "total_llm_calls": p2["call_5_qa"]["llm_calls"],
                },
                "memory_snapshot_path": None,
            })

        return results

    def _run_phase1_batched(
        self,
        sessions: List[Session],
        runners: List[OnlyLLMRunner],
        retrieval_log_paths: List[Path],
    ) -> None:
        """Context construction — interleaved turn-by-turn across sessions."""
        max_turns = max((len(session.get_turn_pairs()) for session in sessions), default=0)

        for turn_idx in tqdm(range(max_turns), desc="Phase1 turns"):
            active = [
                (i, sessions[i], runners[i])
                for i in range(len(sessions))
                if turn_idx < len(sessions[i].get_turn_pairs())
            ]

            for i, session, runner in active:
                user_turn, assistant_turn = session.get_turn_pairs()[turn_idx]

                write_retrieval_log(
                    retrieval_log_paths[i],
                    build_retrieval_log_entry(
                        phase="context_construction",
                        session_id=session.session_id,
                        conv_id=user_turn.conv_id,
                        global_turn=turn_idx,
                        query=user_turn.utterance,
                        turns_in_prompt=len(runner.ctx),
                        total_accumulated_turns=len(runner.ctx) if cfg.HISTORY_ALL_GIVEN else None,
                        token_budget=None,
                        max_context_turns=cfg.MAX_CONTEXT_TURNS if not cfg.HISTORY_ALL_GIVEN else None,
                    ),
                )

                runner.ctx.add(user_turn)
                if assistant_turn:
                    runner.ctx.add(Turn(
                        session_id=assistant_turn.session_id,
                        conv_id=assistant_turn.conv_id,
                        turn_id=assistant_turn.turn_id,
                        global_turn_id=assistant_turn.global_turn_id,
                        role="assistant",
                        utterance=assistant_turn.utterance,
                    ))

    def _run_phase2_batched(
        self,
        sessions: List[Session],
        runners: List[OnlyLLMRunner],
        retrieval_log_paths: List[Path],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """QA answering — all sessions' QA prompts are batched together."""
        schema = QA_SCHEMA_SUPPORTIVE if self.subset == "supportive" else QA_SCHEMA_OPPOSED

        qa_jobs = []
        for i, (session, runner) in enumerate(zip(sessions, runners)):
            for qa in session.qa:
                prompt, _, turns_in_prompt, token_budget = runner.build_qa_prompt(qa)
                write_retrieval_log(
                    retrieval_log_paths[i],
                    build_retrieval_log_entry(
                        phase="qa",
                        session_id=session.session_id,
                        conv_id=-1,
                        global_turn=-1,
                        query=qa.question,
                        turns_in_prompt=turns_in_prompt,
                        total_accumulated_turns=len(runner.ctx) if cfg.HISTORY_ALL_GIVEN else None,
                        token_budget=token_budget,
                        max_context_turns=cfg.MAX_CONTEXT_TURNS if not cfg.HISTORY_ALL_GIVEN else None,
                    ),
                )
                qa_jobs.append((i, qa, prompt, turns_in_prompt, token_budget))

        qa_results_per_session: List[List[Dict]] = [[] for _ in sessions]
        phase2_stats = [
            {"qa_input": 0, "qa_output": 0, "num_qa_calls": 0, "qa_parse_fallback_count": 0}
            for _ in sessions
        ]

        for chunk_start in tqdm(range(0, len(qa_jobs), cfg.QA_BATCH_SIZE), desc="Phase2 QA chunks"):
            chunk = qa_jobs[chunk_start:chunk_start + cfg.QA_BATCH_SIZE]
            if not chunk:
                continue

            chunk_prompts = [job[2] for job in chunk]
            chunk_results, chunk_usages, chunk_fallbacks = self._batch_generate_with_retry(
                chunk_prompts, schema
            )

            for (i, qa, prompt, turns_in_prompt, token_budget), raw, usage, fallback_used in zip(
                chunk, chunk_results, chunk_usages, chunk_fallbacks
            ):
                if runners[i].llm_logger is not None:
                    runners[i].llm_logger.log("call_5_qa", "", prompt, raw)

                qa_tokens = {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "model": self.model_path,
                }
                qa_gen = runners[i].build_qa_result(
                    raw=raw,
                    qa_tokens=qa_tokens,
                    turns_in_prompt=turns_in_prompt,
                    token_budget=token_budget,
                )

                qa_results_per_session[i].append({
                    "question": qa.question,
                    "generated_answer": qa_gen["generated_answer"],
                    "ground_truth_answer": qa.answer,
                    "retrieved_memories": [],
                    "qa_tokens": qa_gen["qa_tokens"],
                })

                phase2_stats[i]["qa_input"] += qa_tokens["input"]
                phase2_stats[i]["qa_output"] += qa_tokens["output"]
                phase2_stats[i]["num_qa_calls"] += 1
                phase2_stats[i]["qa_parse_fallback_count"] += int(fallback_used)

        phase2_final = [
            {
                "call_5_qa": {
                    "input": stats["qa_input"],
                    "output": stats["qa_output"],
                    "llm_calls": stats["num_qa_calls"],
                    "parse_fallback_count": stats["qa_parse_fallback_count"],
                },
            }
            for stats in phase2_stats
        ]
        return qa_results_per_session, phase2_final

    def _batch_generate_with_retry(
        self, prompts: List[str], schema: Dict
    ) -> Tuple[List[Dict], List[Dict], List[bool]]:
        """Batch generate QA answers and retry bad JSON items sequentially."""
        if not prompts:
            return [], [], []

        texts, usages = self.llm_client.generate_batch_raw(
            prompts=prompts,
            system_prompt=None,
            max_tokens=cfg.MAX_TOKENS,
            temperature=cfg.TEMPERATURE,
            guided_json=schema,
            return_usage=True,
        )

        from llm_client import _parse_json_response

        parsed: List[Optional[Dict]] = []
        retry_indices = []
        fallback_flags = [False] * len(prompts)
        for idx, text in enumerate(texts):
            try:
                parsed.append(_parse_json_response(text) if isinstance(text, str) else text)
            except (json.JSONDecodeError, ValueError):
                parsed.append(None)
                retry_indices.append(idx)

        for idx in retry_indices:
            logger.warning(f"Batch item {idx} failed JSON parse — retrying sequentially")
            fallback_flags[idx] = True
            try:
                raw, _ = generate_json_with_fallback(self.llm_client, prompts[idx], schema)
                usage_info = (raw.pop("_usage", {}) or {}) if isinstance(raw, dict) else {}
                usages[idx] = {
                    "prompt_tokens": usage_info.get("prompt_tokens", 0),
                    "completion_tokens": usage_info.get("completion_tokens", 0),
                }
                parsed[idx] = raw if isinstance(raw, dict) else {}
            except Exception as e:
                logger.error(f"Sequential retry for batch item {idx} failed: {e}")
                parsed[idx] = {}

        return [item if isinstance(item, dict) else {} for item in parsed], usages, fallback_flags


# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def create_llm_client(model_path: str, tensor_parallel: int, gpu_memory: float,
                      max_model_len: Optional[int]):
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
    elif engine == "together":
        return _create(engine="together", **cfg.TOGETHER_CONFIG)
    elif engine == "openai":
        return _create(engine="openai", **cfg.OPENAI_CONFIG)
    raise ValueError(f"Unknown LLM engine: {engine}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Step 1: parse --config first so we can load the right module ──
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config_lb")
    pre_args, _ = pre_parser.parse_known_args()
    config_name = Path(pre_args.config).stem  # strip .py if provided

    # Dynamically load the config module (supports renamed config files)
    import importlib.util
    config_file = _MODULE_DIR / f"{config_name}.py"
    if not config_file.exists():
        print(f"[error] Config file not found: {config_file}")
        return 1
    spec = importlib.util.spec_from_file_location(config_name, config_file)
    global cfg
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # ── Step 2: full argument parsing ─────────────────────────────────
    parser = argparse.ArgumentParser(
        description="OnlyLLM QA-Only Baseline Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-session", type=int, required=True,
                        help="First session index (inclusive)")
    parser.add_argument("--end-session", type=int, required=True,
                        help="Last session index (inclusive)")
    parser.add_argument("--subset", type=str, required=True,
                        choices=["opposed", "supportive"],
                        help="Dataset subset to use")
    parser.add_argument("--model", type=str,
                        default=cfg.DEFAULT_VLLM_CONFIG["model_path"])
    parser.add_argument("--tensor-parallel", type=int,
                        default=cfg.DEFAULT_VLLM_CONFIG["tensor_parallel_size"])
    parser.add_argument("--gpu-memory", type=float,
                        default=cfg.DEFAULT_VLLM_CONFIG["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", type=int, default=None,
                        help=(
                            "Model max context length. "
                            "REQUIRED when HISTORY_ALL_GIVEN=True (used for token budget). "
                            "Optional when HISTORY_ALL_GIVEN=False (passed to vLLM if provided)."
                        ))
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE,
                        help=f"Sessions to process in parallel (default: {cfg.BATCH_SIZE})")
    parser.add_argument("--config", type=str, default="config_lb",
                        help="Config file name (without .py). Used to load settings and as output directory prefix.")
    args = parser.parse_args()

    # ── Step 3: validate ──────────────────────────────────────────────
    if not 0.0 < args.gpu_memory <= 1.0:
        parser.error("--gpu-memory must be between 0.0 and 1.0")
    if args.start_session > args.end_session:
        parser.error("--start-session must be <= --end-session")
    if cfg.HISTORY_ALL_GIVEN and args.max_model_len is None:
        parser.error("--max-model-len is required when HISTORY_ALL_GIVEN=True")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if cfg.LLM_ENGINE != "vllm":
        parser.error("Batched only_llm requires LLM_ENGINE='vllm' because it uses generate_batch_raw().")

    # ── Step 4: setup directories and logging ─────────────────────────
    cfg.ensure_directories(args.model, args.subset, args.start_session, args.end_session, config_name)

    session_dir = cfg.get_session_dir(
        args.model, args.subset, args.start_session, args.end_session, config_name
    )
    global logger
    logger = setup_logging(session_dir / "logs")

    results_file    = cfg.get_results_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    checkpoint_file = cfg.get_checkpoint_file(args.model, args.subset, args.start_session, args.end_session, config_name)
    retrieval_log_dir = cfg.get_retrieval_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)
    llm_call_log_dir  = cfg.get_llm_call_log_dir(args.model, args.subset, args.start_session, args.end_session, config_name)

    mode_label = "all_given (token-budget trim)" if cfg.HISTORY_ALL_GIVEN else f"window (last {cfg.MAX_CONTEXT_TURNS} turns)"

    logger.info("=" * 60)
    logger.info("OnlyLLM Batch QA-Only Baseline Experiment")
    logger.info(f"  Config          : {config_name}")
    logger.info(f"  Subset          : {args.subset}")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Sessions        : [{args.start_session}, {args.end_session}]")
    logger.info(f"  History mode    : {mode_label}")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  QA batch size   : {cfg.QA_BATCH_SIZE}")
    if cfg.HISTORY_ALL_GIVEN:
        logger.info(f"  Max model len   : {args.max_model_len}")
        logger.info(f"  Output reserve  : {cfg.OUTPUT_TOKEN_RESERVE} tokens")
        logger.info(f"  Safety margin   : {cfg.CONTEXT_SAFETY_MARGIN} tokens")
    else:
        logger.info(f"  Max model len   : {args.max_model_len if args.max_model_len else 'auto'}")
    logger.info(f"  LLM call logging: {cfg.ENABLE_LLM_CALL_LOGGING}")
    logger.info("=" * 60)

    # ── Step 5: load dataset ──────────────────────────────────────────
    logger.info("Loading dataset...")
    dataset_path = cfg.DATASET_OPPOSED if args.subset == "opposed" else cfg.DATASET_SUPPORTIVE
    sessions = load_implexconv_dataset(dataset_path, args.subset)

    if args.end_session >= len(sessions):
        logger.error(f"end_session={args.end_session} out of range (dataset has {len(sessions)} sessions)")
        return 1

    target_sessions = sessions[args.start_session: args.end_session + 1]
    target_session_ids = [session.session_id for session in target_sessions]

    # ── Step 6: checkpoint resume ─────────────────────────────────────
    completed_ids: Set[int] = set()
    if cfg.ENABLE_CHECKPOINTING:
        completed_ids = load_checkpoint(checkpoint_file, target_session_ids)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} sessions already completed")

    pending_sessions = [session for session in target_sessions if session.session_id not in completed_ids]
    if not pending_sessions:
        logger.info("All sessions already completed.")
        return 0

    # ── Step 7: load tokenizer ────────────────────────────────────────
    # Always loaded — needed for:
    #   • token-budget trimming for QA prompts (HISTORY_ALL_GIVEN=True)
    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )
    logger.info("Tokenizer ready.")

    # ── Step 8: init LLM ──────────────────────────────────────────────
    logger.info("Initialising LLM client...")
    llm_client = create_llm_client(
        args.model, args.tensor_parallel, args.gpu_memory, args.max_model_len
    )
    logger.info("LLM client ready.")

    runner = BatchedOnlyLLMRunner(
        llm_client=llm_client,
        tokenizer=tokenizer,
        subset=args.subset,
        model_path=args.model,
        max_model_len=args.max_model_len,
        config_metadata={
            "config_name": config_name,
            "model": args.model,
            "subset": args.subset,
            "llm_engine": cfg.LLM_ENGINE,
            "temperature": cfg.TEMPERATURE,
            "max_tokens": cfg.MAX_TOKENS,
            "max_model_len": args.max_model_len,
            "session_range": [args.start_session, args.end_session],
            "batch_size": args.batch_size,
            "qa_batch_size": cfg.QA_BATCH_SIZE,
            "history_all_given": cfg.HISTORY_ALL_GIVEN,
            "max_context_turns": cfg.MAX_CONTEXT_TURNS,
            "output_token_reserve": cfg.OUTPUT_TOKEN_RESERVE,
            "context_safety_margin": cfg.CONTEXT_SAFETY_MARGIN,
        },
    )
    results = load_existing_results(results_file)

    print(f"\n{'#'*60}")
    print(f"# OnlyLLM Batch QA-Only  |  subset={args.subset}")
    print(f"# Model  : {cfg.extract_model_name(args.model)}")
    print(f"# Mode   : {mode_label}")
    print(f"# Batch size : {args.batch_size}")
    print(f"# Sessions [{args.start_session}, {args.end_session}]  "
          f"({len(pending_sessions)} to process)")
    print(f"{'#'*60}\n")

    # ── Step 9: main loop ─────────────────────────────────────────────
    num_batches = (len(pending_sessions) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch = pending_sessions[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        session_ids = [session.session_id for session in batch]
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: sessions {session_ids}")

        batch_retrieval_log_paths = [
            retrieval_log_dir / f"session_{session.session_id}_retrieval_log.jsonl"
            for session in batch
        ]
        batch_llm_call_log_dirs = [
            llm_call_log_dir / f"session_{session.session_id}"
            for session in batch
        ]

        try:
            batch_results = runner.run_batch(
                sessions=batch,
                retrieval_log_paths=batch_retrieval_log_paths,
                llm_call_log_dirs=batch_llm_call_log_dirs,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed with hard error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        for result in batch_results:
            results.append(result)
            save_results(results_file, results)

            if cfg.ENABLE_CHECKPOINTING:
                completed_ids.add(result["session_id"])
                save_checkpoint(
                    checkpoint_file, completed_ids,
                    args.model, args.subset,
                    args.start_session, args.end_session,
                    config_name=config_name,
                )

            logger.info(
                f"Session {result['session_id']} done. "
                f"QA: {len(result['qa_results'])}, "
                f"Total LLM calls: {result['token_statistics']['total_llm_calls']}"
            )

    print(f"\n{'#'*60}")
    print(f"# Batch experiment complete!  Results -> {results_file}")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
