"""
evaluation_llm_judge.py

Usage:
    # vLLM (default)
    python evaluation/evaluation_llm_judge.py \
        --root_dir qwen3_1.7b_results \
        --judge_model meta-llama/Llama-3.1-8B-Instruct \
        --tensor_parallel 1 --gpu-memory 0.9 --max-model-len 32768 --batch-size 50

    # OpenAI
    python evaluation/evaluation_llm_judge.py \
        --root_dir qwen3_1.7b_results --judge_model gpt-4o \
        --engine openai --api-key-env OPENAI_API_KEY

    # OpenAI Batch API (~50% cheaper, up to 24h latency; sync fallback for parse failures)
    python evaluation/evaluation_llm_judge.py \
        --root_dir qwen3_1.7b_results --judge_model gpt-4o-mini \
        --engine openai-batch --api-key-env OPENAI_API_KEY \
        --poll-interval 60 --completion-window 24h
    # Notes (openai-batch):
    #   - One OpenAI batch per (model, results_*.json) source file, submitted in parallel.
    #     Each batch's input is written to {output_root}/{model}/batch_input_{stem}.jsonl.
    #   - All in-flight batch ids are saved to {output_root}/batch_state.json so re-running
    #     resumes from where prior run left off (waits on in-progress batches, resubmits any
    #     that ended in failed/expired/cancelled).
    #   - Items returning API errors or unparseable output are retried via the synchronous
    #     OpenAI client (fallback_single_judge with guided_json + temperature ramp).

    # Elice (OpenAI-compatible reasoning endpoint)
    python evaluation/evaluation_llm_judge.py \
        --root_dir qwen3_1.7b_results --judge_model <model-id> \
        --engine elice --base-url https://mlapi.run/<id> --api-key-env ELICE_API_KEY

Outputs:
    evaluation/{llm}_judge_{judge}/
      {model}/judge_results_*.json
      judge_summary.csv
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ["HF_TOKEN"] = "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"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pandas as pd


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LLM_MODULE_DIR = PROJECT_ROOT / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))


CATEGORIES = [1, 2, 3, 4, 5]
JUDGE_MAX_TOKENS = 2048
JUDGE_TEMPERATURE = 0.0
JUDGE_JSON_RETRY = 10
JUDGE_JSON_RETRY_TEMPERATURE_STEP = 0.05
PROGRESS_BAR_WIDTH = 24

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the model answer is correct, 0 if it is incorrect.",
        },
        "reasoning": {
            "type": "string",
            "description": "A concise reason for the 0/1 correctness judgment.",
        },
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a strict but fair evaluator for LoCoMo question answering. "
    "Use only the provided reference conversation sessions and answer in valid JSON."
)


@dataclass
class JudgeItem:
    prompt: str
    output_qa: dict[str, Any]
    category: int
    sample_id: str
    qa_index: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        "--root-dir",
        required=True,
        dest="root_dir",
        help="Results root folder name or path, e.g. qwen3_1.7b_results",
    )
    parser.add_argument(
        "--judge_model",
        "--judge-model",
        required=True,
        dest="judge_model",
        help="Judge model path/name (vllm: HF path, openai/elice: model name)",
    )
    parser.add_argument(
        "--engine",
        choices=["vllm", "openai", "openai-batch", "elice"],
        default="vllm",
        help="Inference engine (default: vllm). elice = OpenAI-compatible reasoning model "
             "via custom base_url (temp=1, max_completion_tokens). openai-batch = submits one "
             "OpenAI Batch per source file (~50%% cheaper, up to 24h latency); parse failures "
             "are sync-retried via the OpenAI client.",
    )
    # vLLM-only options
    parser.add_argument(
        "--tensor_parallel",
        "--tensor-parallel",
        type=int,
        default=1,
        dest="tensor_parallel",
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--gpu_memory",
        "--gpu-memory",
        type=float,
        default=0.9,
        dest="gpu_memory",
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--max_model_len",
        "--max-model-len",
        type=int,
        default=32768,
        dest="max_model_len",
        help="vLLM max model length",
    )
    # OpenAI/elice-only options
    parser.add_argument(
        "--base-url",
        default=None,
        dest="base_url",
        help="OpenAI-compatible endpoint base URL (e.g. https://mlapi.run/<id>)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        dest="api_key_env",
        help="Env var name holding the API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=50,
        dest="batch_size",
        help="Number of judge prompts per batch (default: 50)",
    )
    # openai-batch-only options
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        dest="poll_interval",
        help="(openai-batch only) seconds between batch status polls (default: 60)",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        dest="completion_window",
        help="(openai-batch only) batch completion window (default: 24h)",
    )
    return parser.parse_args()


def resolve_root_dir(root_dir_arg: str) -> Path:
    root_dir = Path(root_dir_arg)
    if root_dir.is_absolute():
        return root_dir
    if root_dir.exists():
        return root_dir.resolve()
    eval_relative = SCRIPT_DIR / root_dir
    if eval_relative.exists():
        return eval_relative.resolve()
    return eval_relative.resolve()


def model_slug(model_name: str) -> str:
    name = model_name.rstrip("/").split("/")[-1].lower()
    name = re.sub(r"[-_]?instruct$", "", name)
    name = re.sub(r"[-_]?chat$", "", name)
    name = re.sub(r"[^a-z0-9.]+", "_", name).strip("_")
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"^llama_(\d)", r"llama\1", name)
    name = re.sub(r"^qwen_(\d)", r"qwen\1", name)
    return name or "judge"


def output_root_for(root_dir: Path, judge_model: str) -> Path:
    root_name = root_dir.name
    llm_name = root_name[:-8] if root_name.endswith("_results") else root_name
    return root_dir.parent / f"{llm_name}_judge_{model_slug(judge_model)}"


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_dataset(dataset_path: Path) -> dict[str, dict]:
    dataset = load_json(dataset_path)
    return {sample["sample_id"]: sample for sample in dataset}


def get_dataset_qa(dataset_sample: Optional[dict], question: str, qa_index: int) -> dict:
    if not dataset_sample:
        return {}

    qa_list = dataset_sample.get("qa", [])
    for qa in qa_list:
        if qa.get("question") == question:
            return qa

    if 0 <= qa_index < len(qa_list):
        return qa_list[qa_index]

    return {}


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def parse_evidence_dia_ids(evidence: list[Any]) -> dict[int, list[str] | None]:
    """Returns {session_id: [dia_id, ...] or None}.
    None means include all turns (fallback when no turn number is specified).
    """
    if isinstance(evidence, str):
        evidence = [evidence]

    result: dict[int, list[str] | None] = {}
    seen: set[str] = set()

    for item in evidence or []:
        for match in re.finditer(r"D(\d+):(\d+)?", str(item)):
            session_id = int(match.group(1))
            turn_num = match.group(2)
            if turn_num is None:
                result[session_id] = None
            else:
                dia_id = f"D{session_id}:{turn_num}"
                if dia_id not in seen:
                    seen.add(dia_id)
                    if session_id not in result:
                        result[session_id] = [dia_id]
                    elif result[session_id] is not None:
                        result[session_id].append(dia_id)
    return result


def build_ref_conv(dataset_sample: Optional[dict], evidence: list[Any]) -> list[dict[str, Any]]:
    if not dataset_sample:
        return []

    conv = dataset_sample.get("conversation", {})
    ref_conv = []
    for session_id, dia_ids in parse_evidence_dia_ids(evidence).items():
        session_key = f"session_{session_id}"
        all_turns = conv.get(session_key, [])
        if dia_ids is None:
            turns = all_turns
        else:
            dia_id_set = set(dia_ids)
            turns = [t for t in all_turns if t.get("dia_id") in dia_id_set]
        ref_conv.append(
            {
                "session_id": session_id,
                "date_time": conv.get(f"{session_key}_date_time", ""),
                "turns": turns,
            }
        )
    return ref_conv


def format_ref_conv(ref_conv: list[dict[str, Any]]) -> str:
    if not ref_conv:
        return ""

    lines = []
    for session in ref_conv:
        session_id = session.get("session_id", "")
        date_time = session.get("date_time", "")
        lines.append(f"Session {session_id} | {date_time}".strip())
        for turn in session.get("turns", []):
            dia_id = turn.get("dia_id", "")
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            lines.append(f"{dia_id} {speaker}: {text}".strip())
        lines.append("")
    return "\n".join(lines).strip()


CATEGORY_DESCRIPTIONS = {
    1: "Multi-hop reasoning — requires synthesizing evidence across multiple sessions.",
    2: "Temporal reasoning — requires reasoning about time, order, or temporal cues.",
    3: "Open-domain knowledge — requires combining conversation evidence with commonsense or world knowledge.",
    4: "Single-hop fact recall — the answer is found from a single conversation session.",
    5: "Adversarial/unanswerable — the question cannot be answered from the conversation.",
}


def build_judge_prompt(
    category: int,
    ref_conv: list[dict[str, Any]],
    question: str,
    ground_truth_answer: Any,
    generated_answer: Any,
    adversarial_answer: Any,
) -> str:
    category_desc = CATEGORY_DESCRIPTIONS.get(category, f"Category {category}")

    formatted_ref_conv = format_ref_conv(ref_conv)
    ref_conv_block = (
        f"Reference conversation sessions:\n{formatted_ref_conv}\n\n"
        if formatted_ref_conv
        else ""
    )

    has_reference_conversation = bool(formatted_ref_conv)

    if category == 5:
        answer_block = (
            "Expected behavior:\n"
            "The model should abstain because the answer is not available in the "
            "conversation. Do not reward concrete factual answers, even if "
            "they appear plausible."
        )
        rubric = (
            "This is a category 5 adversarial/unanswerable question. "
            "Score 1 ONLY if the model answer clearly states that the information is "
            "unavailable, not mentioned, unknown, or cannot be determined from the "
            "conversation. Score 0 if the model provides any substantive "
            "answer, hallucinated answer, unsupported guess, vague non-answer, or "
            "anything short of a clear abstention. Use only 0/1 scoring: "
            "1 for correct, 0 for incorrect."
        )
    else:
        answer_block = f"Ground-truth/reference answer:\n{ground_truth_answer}"
        if has_reference_conversation:
            rubric = (
                "Judge whether the model answer is correct or incorrect compared with the "
                "ground-truth answer, using the reference conversation as supporting evidence. "
                "Score 1 ONLY if the model answer fully and unambiguously answers the question, "
                "is semantically equivalent to the ground-truth answer, and is supported by "
                "the reference conversation. Score 0 if the answer is wrong, unsupported, "
                "contradicted by the reference conversation, incomplete, partially correct, "
                "overly vague, ambiguous, or does not answer the question. Do not give partial "
                "credit. Use only 0/1 scoring: 1 for correct, 0 for incorrect."
            )
        else:
            rubric = (
                "Judge whether the model answer is correct or incorrect compared with the "
                "ground-truth answer. Score 1 ONLY if the model answer fully and unambiguously "
                "answers the question and is semantically equivalent to the ground-truth answer. "
                "Score 0 if the answer is wrong, incomplete, partially correct, overly vague, "
                "ambiguous, or does not answer the question. Do not give partial credit. "
                "Use only 0/1 scoring: 1 for correct, 0 for incorrect."
            )

        if category == 3:
            rubric += (
                " For this open-domain category, commonsense or widely known world knowledge "
                "may be used only when it is needed to interpret the conversation-grounded "
                "question and reference answer; do not reward unsupported speculation."
            )

    score_format = '{"score": 0 or 1, "reasoning": "short explanation"}'

    return f"""Evaluate the model answer for one LoCoMo QA item.

{ref_conv_block}Question:
{question}

Category: {category} — {category_desc}

{answer_block}

Model answer:
{generated_answer}

Rubric:
{rubric}

Return JSON only:
{score_format}
"""


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_judge_json(text: str) -> dict[str, Any]:
    text = strip_json_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def normalize_judge_result(parsed: dict[str, Any], category: int) -> dict[str, Any]:
    raw_score = parsed.get("score")
    score = int(raw_score)
    adjusted = False

    if score not in (0, 1):
        score = 0
        adjusted = True

    result = {
        "score": score,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }
    if adjusted:
        result["raw_score"] = raw_score
        result["score_adjusted"] = True
    return result


def make_error_judge_result(error: Exception, raw_output: Optional[str] = None) -> dict[str, Any]:
    result = {
        "score": 0,
        "reasoning": f"Judge output could not be parsed; assigned fallback score 0: {error}",
        "error": str(error),
        "fallback_score": True,
    }
    if raw_output is not None:
        result["raw_output"] = raw_output
    return result


def is_retryable_judge_error(error: Exception) -> bool:
    if isinstance(error, (json.JSONDecodeError, ValueError)):
        return True
    message = str(error).lower()
    return "json" in message or "validation" in message


def attach_usage(judge_result: dict[str, Any], usage: Optional[dict[str, int]], judge_model: str):
    prompt_tokens = 0 if usage is None else usage.get("prompt_tokens", 0) or 0
    completion_tokens = 0 if usage is None else usage.get("completion_tokens", 0) or 0

    judge_result["judge_model"] = judge_model
    judge_result["prompt_tokens"] = prompt_tokens
    judge_result["completion_tokens"] = completion_tokens
    judge_result["total_tokens"] = prompt_tokens + completion_tokens
    return judge_result


def prepare_judge_output(
    source_file: Path,
    model_name: str,
    judge_model: str,
    results: list[dict[str, Any]],
    dataset_by_sample: dict[str, dict],
) -> tuple[dict[str, Any], list[JudgeItem]]:
    output = {
        "source_file": str(source_file),
        "model": model_name,
        "judge_model": judge_model,
        "score_scale": {
            "categories_1_to_4": {
                "1": "correct; semantically equivalent to the reference answer and supported by the conversation",
                "0": "incorrect, unsupported, incomplete, partially correct, ambiguous, or non-answer",
            },
            "category_5": {
                "1": "correct abstention for adversarial/unanswerable question",
                "0": "non-abstention, substantive unsupported answer, hallucination, or vague non-answer",
            },
        },
        "samples": [],
    }

    judge_items: list[JudgeItem] = []

    for sample in results:
        sample_id = sample.get("sample_id", "")
        dataset_sample = dataset_by_sample.get(sample_id)
        sample_output = {
            "sample_id": sample_id,
            "qa_judgments": [],
        }

        for qa_index, qa_result in enumerate(sample.get("qa_results", [])):
            question = qa_result.get("question", "")
            dataset_qa = get_dataset_qa(dataset_sample, question, qa_index)
            category = int(coalesce(dataset_qa.get("category"), qa_result.get("category"), 0))
            evidence = coalesce(dataset_qa.get("evidence"), qa_result.get("evidence"), [])
            ground_truth_answer = coalesce(
                qa_result.get("ground_truth_answer"),
                dataset_qa.get("answer"),
                dataset_qa.get("adversarial_answer"),
            )
            adversarial_answer = dataset_qa.get("adversarial_answer")
            generated_answer = qa_result.get("generated_answer", "")
            ref_conv = build_ref_conv(dataset_sample, evidence)

            output_qa = {
                "qa_index": qa_index,
                "question": question,
                "category": category,
                "evidence": evidence,
                "ground_truth_answer": ground_truth_answer,
                "adversarial_answer": adversarial_answer,
                "generated_answer": generated_answer,
                "ref_conv": ref_conv,
                "judge": None,
            }
            sample_output["qa_judgments"].append(output_qa)

            prompt = build_judge_prompt(
                category=category,
                ref_conv=ref_conv,
                question=question,
                ground_truth_answer=ground_truth_answer,
                generated_answer=generated_answer,
                adversarial_answer=adversarial_answer,
            )
            judge_items.append(
                JudgeItem(
                    prompt=prompt,
                    output_qa=output_qa,
                    category=category,
                    sample_id=sample_id,
                    qa_index=qa_index,
                )
            )

        output["samples"].append(sample_output)

    return output, judge_items


def fallback_single_judge(
    client,
    item: JudgeItem,
    judge_model: str,
    initial_temperature_step: int = 0,
    engine: str = "vllm",
) -> dict[str, Any]:
    # elice (reasoning mode) requires temperature=1 and ignores any other value.
    fixed_temperature = engine == "elice"

    def _temperature_for(retry_index: int) -> float:
        if fixed_temperature:
            return 1.0
        return JUDGE_TEMPERATURE + (
            JUDGE_JSON_RETRY_TEMPERATURE_STEP * (initial_temperature_step + retry_index)
        )

    last_error: Optional[Exception] = None
    for retry_index in range(JUDGE_JSON_RETRY):
        temperature = _temperature_for(retry_index)
        try:
            response = client.generate(
                prompt=item.prompt,
                system_prompt=SYSTEM_PROMPT,
                guided_json=JUDGE_SCHEMA,
                temperature=temperature,
                max_tokens=JUDGE_MAX_TOKENS,
                json_retry=1,
                return_usage=True,
            )
            usage = response.pop("_usage", None) if isinstance(response, dict) else None
            parsed = response if isinstance(response, dict) else parse_judge_json(str(response))
            result = normalize_judge_result(parsed, item.category)
            result["judge_temperature"] = temperature
            result["json_retry_attempt"] = retry_index + 1
            return attach_usage(result, usage, judge_model)
        except Exception as exc:
            last_error = exc
            if not is_retryable_judge_error(exc):
                result = make_error_judge_result(exc)
                result["judge_temperature"] = temperature
                result["json_retry_attempt"] = retry_index + 1
                return attach_usage(result, None, judge_model)

    error = last_error or RuntimeError("Judge retry failed without an exception")
    result = make_error_judge_result(error)
    result["judge_temperature"] = _temperature_for(JUDGE_JSON_RETRY - 1)
    result["json_retry_attempt"] = JUDGE_JSON_RETRY
    return attach_usage(result, None, judge_model)


def run_judge_batches(
    client,
    judge_items: list[JudgeItem],
    judge_model: str,
    batch_size: int,
    progress_label: str,
    engine: str = "vllm",
):
    total = len(judge_items)
    if total == 0:
        print(f"  {progress_label} {format_progress_bar(0, 0)}", flush=True)
        return

    file_start_time = time.perf_counter()
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = judge_items[start:end]
        batch_start_time = time.perf_counter()
        print_judge_progress(
            progress_label=progress_label,
            done=start,
            total=total,
            status=f"running QA {start + 1}-{end}",
            current_item=batch[-1],
        )

        if engine in ("openai", "elice"):
            # Concurrent individual calls via ThreadPoolExecutor
            def _call_one(item: JudgeItem):
                return fallback_single_judge(client, item, judge_model, engine=engine)

            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                batch_results = list(ex.map(_call_one, batch))

            for item_offset, (item, result) in enumerate(
                zip(batch, batch_results), start=start + 1
            ):
                item.output_qa["judge"] = result
        else:
            # vLLM batch generation
            try:
                texts, usages = client.generate_batch_raw(
                    prompts=[item.prompt for item in batch],
                    system_prompt=SYSTEM_PROMPT,
                    guided_json=JUDGE_SCHEMA,
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=JUDGE_MAX_TOKENS,
                    return_usage=True,
                )
            except Exception as exc:
                print(flush=True)
                logging.warning(
                    "[%s] Batch judge failed at QA %s-%s / %s; falling back item-by-item: %s",
                    progress_label,
                    start + 1,
                    end,
                    total,
                    exc,
                )
                for item_offset, item in enumerate(batch, start=start + 1):
                    print_judge_progress(
                        progress_label=progress_label,
                        done=item_offset - 1,
                        total=total,
                        status=f"fallback QA {item_offset}",
                        current_item=item,
                    )
                    item.output_qa["judge"] = fallback_single_judge(
                        client, item, judge_model, engine=engine
                    )
                    batch_elapsed = time.perf_counter() - batch_start_time
                    total_elapsed = time.perf_counter() - file_start_time
                    print_judge_progress(
                        progress_label=progress_label,
                        done=item_offset,
                        total=total,
                        status=build_progress_status(
                            done=item_offset,
                            total=total,
                            label=f"fallback QA {item_offset} done",
                            batch_elapsed=batch_elapsed,
                            total_elapsed=total_elapsed,
                        ),
                        current_item=item,
                    )
                continue

            for item_offset, (item, text, usage) in enumerate(
                zip(batch, texts, usages),
                start=start + 1,
            ):
                try:
                    parsed = parse_judge_json(text)
                    result = normalize_judge_result(parsed, item.category)
                except Exception as exc:
                    print(flush=True)
                    logging.warning(
                        "[%s] Judge JSON parse failed at QA %s / %s (%s); retrying single item: %s",
                        progress_label,
                        item_offset,
                        total,
                        describe_judge_item(item),
                        exc,
                    )
                    result = fallback_single_judge(
                        client,
                        item,
                        judge_model,
                        initial_temperature_step=1,
                        engine=engine,
                    )
                else:
                    result = attach_usage(result, usage, judge_model)
                item.output_qa["judge"] = result
        batch_elapsed = time.perf_counter() - batch_start_time
        total_elapsed = time.perf_counter() - file_start_time
        print_judge_progress(
            progress_label=progress_label,
            done=end,
            total=total,
            status=build_progress_status(
                done=end,
                total=total,
                label=f"completed QA {start + 1}-{end}",
                batch_elapsed=batch_elapsed,
                total_elapsed=total_elapsed,
            ),
            current_item=batch[-1],
        )
    print(flush=True)


def format_progress_bar(done: int, total: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    if total <= 0:
        return f"[{'-' * width}] 0/0 (  0.0%)"

    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} ({ratio * 100:5.1f}%)"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def build_progress_status(
    done: int,
    total: int,
    label: str,
    batch_elapsed: float,
    total_elapsed: float,
) -> str:
    eta = 0.0
    if done > 0 and total > done:
        eta = total_elapsed * ((total - done) / done)
    return (
        f"{label} | batch {format_duration(batch_elapsed)} "
        f"| elapsed {format_duration(total_elapsed)} | eta {format_duration(eta)}"
    )


def print_judge_progress(
    progress_label: str,
    done: int,
    total: int,
    status: str,
    current_item: Optional[JudgeItem] = None,
):
    location = ""
    if current_item is not None:
        location = f" | up to {describe_judge_item(current_item)}"
    line = f"  {progress_label} | {format_progress_bar(done, total)} | {status}{location}"
    last_len = getattr(print_judge_progress, "_last_line_len", 0)
    padding = " " * max(0, last_len - len(line))
    print(
        f"\r{line}{padding}",
        end="",
        flush=True,
    )
    print_judge_progress._last_line_len = len(line)


def describe_judge_item(item: JudgeItem) -> str:
    sample = item.sample_id or "unknown_sample"
    return f"{sample} qa={item.qa_index} cat={item.category}"


def add_judge_token_summary_columns(summary: dict[str, Any]) -> dict[str, Any]:
    input_tokens = summary.get("judge_input_tokens")
    if input_tokens is None:
        input_tokens = summary.get("prompt_tokens", 0) or 0

    output_tokens = summary.get("judge_output_tokens")
    if output_tokens is None:
        output_tokens = summary.get("completion_tokens", 0) or 0

    total_tokens = summary.get("judge_total_tokens")
    if total_tokens is None:
        total_tokens = summary.get("total_tokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    summary["judge_input_tokens"] = input_tokens
    summary["judge_output_tokens"] = output_tokens
    summary["judge_total_tokens"] = total_tokens
    summary.setdefault("total_tokens", total_tokens)
    return summary


def summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    scores_by_cat: dict[int, list[int]] = {cat: [] for cat in CATEGORIES}
    prompt_tokens = 0
    completion_tokens = 0
    num_qa = 0
    num_scored = 0
    num_failed = 0

    for sample in output.get("samples", []):
        for qa in sample.get("qa_judgments", []):
            num_qa += 1
            judge = qa.get("judge") or {}
            prompt_tokens += judge.get("prompt_tokens", 0) or 0
            completion_tokens += judge.get("completion_tokens", 0) or 0
            score = judge.get("score")
            category = qa.get("category")
            if score is None:
                num_failed += 1
                continue
            num_scored += 1
            if category in scores_by_cat:
                scores_by_cat[category].append(int(score))

    all_scores = [score for scores in scores_by_cat.values() for score in scores]
    summary = {
        "model": output.get("model", ""),
        "source_file": output.get("source_file", ""),
        "judge_model": output.get("judge_model", ""),
        "num_qa": num_qa,
        "num_scored": num_scored,
        "num_failed": num_failed,
        "llm_judge_accuracy": sum(all_scores) / len(all_scores) if all_scores else None,
        "accuracy": sum(all_scores) / len(all_scores) if all_scores else None,  # backward-compatible alias
        "score_0_count": all_scores.count(0),
        "score_1_count": all_scores.count(1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    add_judge_token_summary_columns(summary)
    for cat in CATEGORIES:
        cat_scores = scores_by_cat[cat]
        summary[f"cat{cat}_count"] = len(cat_scores)
        summary[f"cat{cat}_correct"] = sum(cat_scores)
        cat_accuracy = sum(cat_scores) / len(cat_scores) if cat_scores else None
        summary[f"cat{cat}_llm_judge_accuracy"] = cat_accuracy
        summary[f"cat{cat}_accuracy"] = cat_accuracy  # backward-compatible alias
    return summary


def write_summary_csv(output_root: Path) -> Path:
    rows = []
    for path in sorted(output_root.glob("*/judge_results_*.json")):
        data = load_json(path)
        summary = data.get("summary") or summarize_output(data)
        add_judge_token_summary_columns(summary)
        rows.append(summary)

    csv_path = output_root / "judge_summary.csv"
    if rows:
        col_order = [
            "model", "judge_model",
            "num_qa", "num_failed", "accuracy",
            "score_0_count", "score_1_count",
            "judge_input_tokens", "judge_output_tokens", "judge_total_tokens",
            "prompt_tokens", "completion_tokens",
        ]
        for cat in CATEGORIES:
            col_order.extend([
                f"cat{cat}_count",
                f"cat{cat}_correct",
                f"cat{cat}_accuracy",
            ])
        pd.DataFrame(rows)[col_order].to_csv(csv_path, index=False)
    else:
        pd.DataFrame().to_csv(csv_path, index=False)
    return csv_path


def discover_result_files(root_dir: Path) -> list[tuple[str, Path]]:
    files = []
    for model_dir in sorted(d for d in root_dir.iterdir() if d.is_dir()):
        for result_file in sorted(model_dir.glob("results_*.json")):
            files.append((model_dir.name, result_file))
    return files


# ---------------- OpenAI Batch (hybrid) ----------------

_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_REASONING_MAX_TOKENS_FLOOR = 4096
_BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def is_reasoning_model(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    m = model_name.lower()
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def adjust_for_reasoning_model(api_kwargs: dict[str, Any]) -> dict[str, Any]:
    if not is_reasoning_model(api_kwargs.get("model", "")):
        return api_kwargs
    if "max_tokens" in api_kwargs:
        mt = api_kwargs.pop("max_tokens")
        api_kwargs["max_completion_tokens"] = max(int(mt or 0), _REASONING_MAX_TOKENS_FLOOR)
    elif "max_completion_tokens" in api_kwargs:
        api_kwargs["max_completion_tokens"] = max(
            int(api_kwargs["max_completion_tokens"] or 0), _REASONING_MAX_TOKENS_FLOOR
        )
    api_kwargs.pop("temperature", None)
    api_kwargs.pop("top_p", None)
    return api_kwargs


def build_batch_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "locomo_judge",
            "strict": True,
            "schema": JUDGE_SCHEMA,
        },
    }


def build_batch_input_file(
    judge_items: list[JudgeItem],
    judge_model: str,
    path: Path,
    temperature: float = JUDGE_TEMPERATURE,
) -> None:
    is_reasoning = is_reasoning_model(judge_model)
    if is_reasoning:
        logging.info(
            "  reasoning model detected (%s): using max_completion_tokens, dropping temperature",
            judge_model,
        )
    response_format = build_batch_response_format()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(judge_items):
            body: dict[str, Any] = {
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item.prompt},
                ],
                "max_tokens": JUDGE_MAX_TOKENS,
                "temperature": temperature,
                "response_format": response_format,
            }
            adjust_for_reasoning_model(body)
            line = {
                "custom_id": f"req-{idx:08d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def submit_batch(client, batch_input_path: Path, completion_window: str):
    with open(batch_input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch


def download_batch_records(client, output_file_id: str) -> list[dict[str, Any]]:
    resp = client.files.content(output_file_id)
    if hasattr(resp, "text") and isinstance(resp.text, str):
        text = resp.text
    elif hasattr(resp, "content"):
        raw = resp.content
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    else:
        text = str(resp)
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logging.warning("  bad batch output line skipped: %s", exc)
    return records


def extract_batch_result(
    record: dict[str, Any],
) -> tuple[Optional[str], Optional[dict[str, int]], Optional[str]]:
    if record.get("error"):
        return None, None, str(record["error"])
    response = record.get("response") or {}
    status_code = response.get("status_code")
    if status_code is not None and status_code != 200:
        return None, None, f"status={status_code}"
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None, None, "no choices"
    text = (choices[0].get("message") or {}).get("content", "")
    usage = body.get("usage") or {}
    usage_dict = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }
    return text, usage_dict, None


def save_batch_state(state_path: Path, jobs: list[dict[str, Any]], judge_model: str) -> None:
    payload = {
        "judge_model": judge_model,
        "batches": [
            {
                "model_name": j["model_name"],
                "source_file": str(j["source_file"]),
                "batch_id": j["batch_id"],
                "input_file_id": j.get("input_file_id"),
                "n_requests": len(j["judge_items"]),
            }
            for j in jobs if j.get("batch_id")
        ],
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state_path, payload)


def load_batch_state(state_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not state_path.exists():
        return {}
    try:
        raw = load_json(state_path)
    except Exception as exc:
        logging.warning("Could not load %s: %s", state_path.name, exc)
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in raw.get("batches", []):
        out[(entry["model_name"], entry["source_file"])] = entry
    return out


def process_completed_batch(
    client,
    job: dict[str, Any],
    output_file_id: str,
    judge_model: str,
    sync_client,
) -> None:
    label = f"{job['model_name']}/{job['source_file'].name}"
    logging.info("  [%s] downloading results...", label)
    records = download_batch_records(client, output_file_id)
    logging.info("  [%s] parsed %d records", label, len(records))

    results_by_cid: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("custom_id")
        if not cid:
            continue
        text, usage_dict, err = extract_batch_result(rec)
        results_by_cid[cid] = {"text": text, "usage": usage_dict, "error": err}

    judge_items: list[JudgeItem] = job["judge_items"]
    sync_retry_queue: list[tuple[int, JudgeItem]] = []
    for idx, item in enumerate(judge_items):
        cid = f"req-{idx:08d}"
        res = results_by_cid.get(cid)
        if res is None or res["text"] is None:
            sync_retry_queue.append((idx, item))
            continue
        try:
            parsed = parse_judge_json(res["text"])
            result = normalize_judge_result(parsed, item.category)
        except Exception as exc:
            logging.warning(
                "  [%s] parse failed for %s (%s); queuing sync retry",
                label,
                describe_judge_item(item),
                exc,
            )
            sync_retry_queue.append((idx, item))
            continue
        result["judge_temperature"] = JUDGE_TEMPERATURE
        result["json_retry_attempt"] = 0
        result["via_batch"] = True
        attach_usage(result, res["usage"], judge_model)
        item.output_qa["judge"] = result

    n_filled = len(judge_items) - len(sync_retry_queue)
    logging.info(
        "  [%s] filled %d/%d from batch; %d need sync retry",
        label,
        n_filled,
        len(judge_items),
        len(sync_retry_queue),
    )

    if sync_retry_queue:
        for _, item in sync_retry_queue:
            result = fallback_single_judge(sync_client, item, judge_model, engine="openai")
            item.output_qa["judge"] = result

    output = job["output"]
    output["summary"] = summarize_output(output)
    atomic_write_json(job["output_file"], output)
    logging.info("  [%s] wrote %s", label, job["output_file"])


def run_openai_batch_mode(
    args,
    pending: list[tuple[str, Path, Path]],
    dataset_by_sample: dict[str, dict],
    output_root: Path,
) -> None:
    try:
        import openai
    except ImportError as exc:
        raise ImportError("openai package required for --engine openai-batch") from exc

    from llm_client import UnifiedLLMClient

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in env var '{args.api_key_env}'")
    client = openai.OpenAI(api_key=api_key)

    state_path = output_root / "batch_state.json"
    existing_by_key = load_batch_state(state_path)
    if existing_by_key:
        logging.info(
            "Loaded %d prior batch entries from %s", len(existing_by_key), state_path.name
        )

    # -------- Phase 1: prepare per-file jobs --------
    logging.info("== Phase 1: preparing per-file batches ==")
    jobs: list[dict[str, Any]] = []
    for model_name, source_file, output_file in pending:
        label = f"{model_name}/{source_file.name}"
        logging.info("── %s", label)
        results = load_json(source_file)
        output, judge_items = prepare_judge_output(
            source_file=source_file,
            model_name=model_name,
            judge_model=args.judge_model,
            results=results,
            dataset_by_sample=dataset_by_sample,
        )
        if not judge_items:
            logging.info("  no judge items — writing summary-only output")
            output["summary"] = summarize_output(output)
            atomic_write_json(output_file, output)
            continue

        jobs.append({
            "model_name": model_name,
            "source_file": source_file,
            "output_file": output_file,
            "output": output,
            "judge_items": judge_items,
            "input_path": output_root / model_name / f"batch_input_{source_file.stem}.jsonl",
            "batch_id": None,
            "input_file_id": None,
        })
        logging.info("  queued %d prompts", len(judge_items))

    if not jobs:
        logging.info("Nothing to submit — done")
        return

    # -------- Phase 2: submit (or resume) per-file batches --------
    logging.info("== Phase 2: submitting/resuming %d batches ==", len(jobs))
    for job in jobs:
        key = (job["model_name"], str(job["source_file"]))
        label = f"{job['model_name']}/{job['source_file'].name}"
        prior = existing_by_key.get(key)
        if prior and prior.get("batch_id"):
            try:
                batch = client.batches.retrieve(prior["batch_id"])
                if batch.status in ("expired", "cancelled", "failed"):
                    logging.warning(
                        "  [%s] prior batch %s is %s — resubmitting",
                        label, prior["batch_id"], batch.status,
                    )
                else:
                    job["batch_id"] = prior["batch_id"]
                    job["input_file_id"] = prior.get("input_file_id")
                    logging.info(
                        "  [%s] resumed batch %s (status=%s)", label, batch.id, batch.status
                    )
                    continue
            except Exception as exc:
                logging.warning("  [%s] could not retrieve prior batch: %s", label, exc)

        build_batch_input_file(job["judge_items"], args.judge_model, job["input_path"])
        batch = submit_batch(client, job["input_path"], args.completion_window)
        job["batch_id"] = batch.id
        job["input_file_id"] = batch.input_file_id
        logging.info(
            "  [%s] submitted batch %s (%d requests)",
            label, batch.id, len(job["judge_items"]),
        )

    save_batch_state(state_path, jobs, args.judge_model)

    # -------- Phase 3: poll all batches; process each on completion --------
    logging.info(
        "== Phase 3: polling %d batches every %ds ==", len(jobs), args.poll_interval
    )
    sync_client = UnifiedLLMClient(
        engine="openai",
        model_name=args.judge_model,
        api_key=api_key,
        base_url=args.base_url,
    )
    pending_map: dict[tuple[str, str], dict[str, Any]] = {
        (j["model_name"], str(j["source_file"])): j for j in jobs
    }
    failed_labels: list[str] = []

    while pending_map:
        finished_keys: list[tuple[str, str]] = []
        for key, job in list(pending_map.items()):
            label = f"{job['model_name']}/{job['source_file'].name}"
            try:
                batch = client.batches.retrieve(job["batch_id"])
            except Exception as exc:
                logging.warning("  [%s] retrieve failed: %s", label, exc)
                continue
            counts = getattr(batch, "request_counts", None)
            completed = getattr(counts, "completed", "?") if counts else "?"
            total = getattr(counts, "total", "?") if counts else "?"
            failed_n = getattr(counts, "failed", "?") if counts else "?"
            logging.info(
                "  [%s] batch=%s status=%s progress=%s/%s failed=%s",
                label, job["batch_id"], batch.status, completed, total, failed_n,
            )
            if batch.status in _BATCH_TERMINAL_STATUSES:
                if batch.status == "completed":
                    try:
                        process_completed_batch(
                            client, job, batch.output_file_id,
                            args.judge_model, sync_client,
                        )
                    except Exception as exc:
                        logging.error(
                            "  [%s] processing failed: %s", label, exc, exc_info=True
                        )
                        failed_labels.append(label)
                else:
                    logging.error(
                        "  [%s] ended with status=%s — leaving for next run",
                        label, batch.status,
                    )
                    failed_labels.append(label)
                finished_keys.append(key)

        for key in finished_keys:
            del pending_map[key]
        save_batch_state(state_path, jobs, args.judge_model)

        if pending_map:
            time.sleep(args.poll_interval)

    # -------- Phase 4: cleanup + summary CSV --------
    all_saved = all(j["output_file"].exists() for j in jobs)
    if all_saved and not failed_labels and state_path.exists():
        try:
            state_path.unlink()
        except Exception:
            pass

    if failed_labels:
        logging.warning(
            "Done with %d failed file(s): %s. Re-run to retry.",
            len(failed_labels), failed_labels,
        )

    summary_path = write_summary_csv(output_root)
    logging.info("Summary CSV  : %s", summary_path)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")
    
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    root_dir = resolve_root_dir(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Results root not found: {root_dir}")

    dataset_path = root_dir.parent.parent / "dataset" / "locomo10.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_root = output_root_for(root_dir, args.judge_model)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Results root : {root_dir}")
    print(f"Dataset      : {dataset_path}")
    print(f"Output root  : {output_root}")
    print(f"Judge model  : {args.judge_model}")
    print(f"Engine       : {args.engine}")
    print(f"Batch size   : {args.batch_size}")
    print(f"JSON retry   : {JUDGE_JSON_RETRY}")
    if args.engine == "elice":
        print("Retry temp   : fixed at 1.0 (elice reasoning mode)")
    else:
        print(f"Retry temp   : +{JUDGE_JSON_RETRY_TEMPERATURE_STEP} per JSON retry")

    result_files = discover_result_files(root_dir)
    if not result_files:
        raise FileNotFoundError(f"No results_*.json files found under: {root_dir}")

    dataset_by_sample = load_dataset(dataset_path)

    pending = []
    for model_name, source_file in result_files:
        output_file = output_root / model_name / f"judge_{source_file.name}"
        if output_file.exists():
            print(f"Skip existing: {output_file}")
            continue
        pending.append((model_name, source_file, output_file))

    if args.engine == "openai-batch":
        if not pending:
            print("No new files to judge.")
            summary_path = write_summary_csv(output_root)
            print(f"\nSummary CSV  : {summary_path}")
            print("Done.")
            return
        run_openai_batch_mode(
            args=args,
            pending=pending,
            dataset_by_sample=dataset_by_sample,
            output_root=output_root,
        )
        print("Done.")
        return

    if pending:
        from llm_client import UnifiedLLMClient

        if args.engine == "vllm":
            client = UnifiedLLMClient(
                engine="vllm",
                model_path=args.judge_model,
                tensor_parallel_size=args.tensor_parallel,
                gpu_memory_utilization=args.gpu_memory,
                max_model_len=args.max_model_len,
            )
        else:  # openai or elice
            api_key = os.getenv(args.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in env var '{args.api_key_env}'")
            client = UnifiedLLMClient(
                engine=args.engine,
                model_name=args.judge_model,
                api_key=api_key,
                base_url=args.base_url,
                reasoning_mode=(args.engine == "elice"),
            )
    else:
        client = None
        print("No new files to judge.")

    total_pending = len(pending)
    for file_index, (model_name, source_file, output_file) in enumerate(pending, start=1):
        progress_label = f"file {file_index}/{total_pending}"
        source_label = f"{model_name}/{source_file.name}"
        print(f"\nProcessing {progress_label}: {source_label}", flush=True)
        results = load_json(source_file)
        output, judge_items = prepare_judge_output(
            source_file=source_file,
            model_name=model_name,
            judge_model=args.judge_model,
            results=results,
            dataset_by_sample=dataset_by_sample,
        )
        print(f"  Prepared {len(judge_items)} QA items", flush=True)
        assert client is not None
        run_judge_batches(
            client=client,
            judge_items=judge_items,
            judge_model=args.judge_model,
            batch_size=args.batch_size,
            progress_label=progress_label,
            engine=args.engine,
        )
        output["summary"] = summarize_output(output)
        atomic_write_json(output_file, output)
        print(f"  {progress_label} wrote {output_file}", flush=True)

    summary_path = write_summary_csv(output_root)
    print(f"\nSummary CSV  : {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
