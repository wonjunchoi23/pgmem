"""
evaluation_llm_judge.py — PrefBench-style 4-criterion LLM-as-judge for PrefEval results.

Usage:
    # vLLM (local)
    python evaluation_llm_judge.py \
        --llm Qwen3-1.7B \
        --engine vllm \
        --judge-model meta-llama/Llama-3.1-8B-Instruct \
        --batch-size 32

    # OpenAI API (synchronous)
    python evaluation_llm_judge.py \
        --llm Qwen3-1.7B \
        --engine openai \
        --judge-model gpt-4o-mini \
        --api-key sk-... \
        --batch-size 32 \
        --max-concurrent 10

    # OpenAI Batch API (~50% cheaper, up to 24h latency; sync fallback for parse failures)
    python evaluation_llm_judge.py \
        --llm Qwen3-1.7B \
        --engine openai-batch \
        --judge-model gpt-4o-mini \
        --api-key sk-...
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_module.llm_client import (  # noqa: E402
    UnifiedLLMClient,
    _adjust_for_reasoning_model,
    _is_reasoning_model,
)
from prompt_prefeval import (  # noqa: E402
    USER_PROMPT_TEMPLATE_VIOLATION,
    USER_PROMPT_TEMPLATE_ACKNOWLEDGEMENT,
    USER_PROMPT_TEMPLATE_HALLUCINATION,
    USER_PROMPT_TEMPLATE_HELPFUL,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

DEFAULT_MODULES = ["pgmem"]


CONFIG_FOLDER_RE = re.compile(r"^config_(\d+)_outputs_(.+)$")

_DEFAULT_TEMP     = 0.0    # first-pass temperature (batch input + sync first attempt)
_RETRY_TEMP_START = 0.3    # sync-retry start (after a parse failure); diversified vs. first pass
_RETRY_TEMP_STEP  = 0.025
_RETRY_MAX        = 30


# ---------------- sub-dim spec ----------------
#
# For each criterion:
#   template     : user prompt template (single .format() pass)
#   inputs       : placeholder names; values pulled from the results.jsonl row
#   yes_is_good  : True  -> Answer="Yes" maps to 1, "No" maps to 0
#                  False -> Answer="Yes" maps to 0, "No" maps to 1
#   max_tokens   : generation cap

SUBDIM_SPEC: Dict[str, Dict[str, Any]] = {
    "violation": {
        "template":    USER_PROMPT_TEMPLATE_VIOLATION,
        "inputs":      ["preference", "question", "response"],
        "yes_is_good": False,
        "max_tokens":  1000,
    },
    "acknowledgement": {
        "template":    USER_PROMPT_TEMPLATE_ACKNOWLEDGEMENT,
        "inputs":      ["question", "response"],
        "yes_is_good": True,
        "max_tokens":  1000,
    },
    "hallucination": {
        "template":    USER_PROMPT_TEMPLATE_HALLUCINATION,
        "inputs":      ["preference", "restatement"],
        "yes_is_good": False,
        "max_tokens":  1000,
    },
    "helpful": {
        "template":    USER_PROMPT_TEMPLATE_HELPFUL,
        "inputs":      ["question", "response"],
        "yes_is_good": True,
        "max_tokens":  1000,
    },
}

ALL_SUBDIMS: List[str] = list(SUBDIM_SPEC.keys())


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(description="PrefBench-style 4-criterion LLM-as-judge")
    p.add_argument("--llm", required=True,
                   help="LLM tag in folder names, e.g. Qwen3-1.7B")
    p.add_argument("--engine", choices=["vllm", "openai", "openai-batch"], default="vllm",
                   help="Inference engine for the judge model (default: vllm). "
                        "'openai-batch' submits a single batch via OpenAI's Batch API "
                        "(~50%% cheaper, up to 24h latency) and falls back to the "
                        "synchronous OpenAI client for any parse failures.")
    p.add_argument("--judge-model", required=True,
                   help="Judge model. vLLM: HF path (e.g. meta-llama/Llama-3.1-8B-Instruct). "
                        "OpenAI: model name (e.g. gpt-4o-mini)")
    p.add_argument("--api-key", default=None,
                   help="API key (OpenAI engine only). Falls back to OPENAI_API_KEY env var")
    p.add_argument("--criteria", nargs="+", choices=ALL_SUBDIMS, default=ALL_SUBDIMS,
                   help="Which criteria to score (default: all four)")
    p.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    p.add_argument("--root", default=None,
                   help="exp_prefeval root (default: parent of this script)")
    # vLLM-only options
    p.add_argument("--tensor-parallel", type=int, default=1,
                   help="(vLLM only) tensor parallel size")
    p.add_argument("--gpu-memory", type=float, default=0.9,
                   help="(vLLM only) GPU memory utilization")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="(vLLM only) max model length")
    # OpenAI-only options
    p.add_argument("--max-concurrent", type=int, default=10,
                   help="(OpenAI only) max concurrent in-flight API requests per batch")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="(openai-batch only) seconds between batch status polls (default: 60)")
    p.add_argument("--completion-window", default="24h",
                   help="(openai-batch only) batch completion window (default: 24h)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-judge even sub-dims already filled in the detail JSON")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip judging entirely; just (re)build judge_summary.csv from the "
                        "detail JSONs already present in the output directory.")
    return p.parse_args()


# ---------------- helpers ----------------

def find_config_dirs(module_dir: Path, llm: str) -> List[Tuple[int, Path]]:
    out = []
    if not module_dir.is_dir():
        return out
    for sub in sorted(module_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = CONFIG_FOLDER_RE.match(sub.name)
        if not m or m.group(2) != llm:
            continue
        out.append((int(m.group(1)), sub))
    return out


def load_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"  bad JSON line skipped: {e}")
    return rows


def is_valid_answer(ans: Any) -> bool:
    return ans is not None and str(ans).strip() not in ("", "N/A")


# Match `<answer>Yes</answer>` / `<answer>[Yes]</answer>` / `<answer> No </answer>` etc.
_ANSWER_RE = re.compile(r"<answer>\s*\[?\s*(Yes|No)\s*\]?\s*</answer>", re.IGNORECASE)
# Fallback: a trailing standalone Yes/No (after stripping any XML tags).
_FALLBACK_YESNO_RE = re.compile(r"\b(Yes|No)\b\s*\.?\s*$", re.IGNORECASE)


def _parse_yesno(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).capitalize()
    stripped = re.sub(r"<[^>]+>", "", text).strip()
    m2 = _FALLBACK_YESNO_RE.search(stripped)
    if m2:
        return m2.group(1).capitalize()
    return None


def _yesno_to_score(yesno: Optional[str], yes_is_good: bool) -> Optional[int]:
    if yesno is None:
        return None
    is_yes = yesno.lower() == "yes"
    if yes_is_good:
        return 1 if is_yes else 0
    return 0 if is_yes else 1


def make_parser(subdim: str):
    yes_is_good = SUBDIM_SPEC[subdim]["yes_is_good"]
    def _parse(text: Any) -> Optional[int]:
        if isinstance(text, dict):
            text = text.get("content", "")
        if not isinstance(text, str):
            text = str(text or "")
        return _yesno_to_score(_parse_yesno(text), yes_is_good)
    return _parse


def _accumulate_usage(usage: Dict, u: Optional[Dict]) -> None:
    if u is None:
        return
    usage["calls"] = usage.get("calls", 0) + 1
    usage["prompt_tokens"]     = usage.get("prompt_tokens", 0)     + int(u.get("prompt_tokens", 0))
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + int(u.get("completion_tokens", 0))


def _retry_with_temp(prompt, parse_fn, judge_client, max_tokens, usage):
    temperature = _RETRY_TEMP_START
    for attempt in range(_RETRY_MAX):
        try:
            out = judge_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                return_usage=True,
            )
            text = out.get("content", "") if isinstance(out, dict) else str(out)
            u = out.get("_usage") if isinstance(out, dict) else None
            _accumulate_usage(usage, u)
            score = parse_fn(text)
            if score is not None:
                return score
        except Exception as e:
            logging.warning(f"    retry attempt {attempt+1} failed (T={temperature:.3f}): {e}")
        temperature += _RETRY_TEMP_STEP
    return None


def build_prompt_for_subdim(subdim: str, qa_row: Dict[str, Any]) -> str:
    spec = SUBDIM_SPEC[subdim]
    template = spec["template"]
    question   = qa_row.get("question", "") or ""
    response   = qa_row.get("model_answer", "") or ""
    preference = qa_row.get("preference", "") or ""
    # Hallucination uses the full model_answer as the "restatement" proxy.
    fmt_kwargs = {
        "question":    question,
        "response":    response,
        "preference":  preference,
        "restatement": response,
    }
    return template.format(**{k: fmt_kwargs[k] for k in spec["inputs"]})


def _load_folder_qa(
    cfg_dir: Path,
    detail_path: Path,
    active_subdims: List[str],
    overwrite: bool,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, List[Dict[str, Any]]]]]:
    """Load results.jsonl + prior detail (for resume), produce qa_entries and
    a per-subdim list of items still needing judging.

    Returns:
        (qa_entries, flat_items) where flat_items[subdim] = [{"qa_idx", "prompt"}, ...].
        Returns (None, None) if results.jsonl is missing.
    """
    results_path = cfg_dir / "results.jsonl"
    if not results_path.exists():
        return None, None
    rows = load_results_jsonl(results_path)

    prior_by_key: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    if detail_path.exists() and not overwrite:
        try:
            with open(detail_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            for q in prior.get("qa_entries", []):
                key = (q.get("k"), q.get("question_session"), q.get("question", ""))
                prior_by_key[key] = q
            logging.info(f"  resuming with {len(prior_by_key)} prior entries")
        except Exception as e:
            logging.warning(f"  could not load prior detail file: {e}")

    qa_entries: List[Dict[str, Any]] = []
    flat_items: Dict[str, List[Dict[str, Any]]] = {n: [] for n in active_subdims}

    for row in rows:
        question_session = row.get("question_session")
        question         = row.get("question", "")
        model_answer     = row.get("model_answer", "")

        eligible = is_valid_answer(model_answer)

        prior = prior_by_key.get((row.get("k"), question_session, question), {})
        entry = {
            "k":                 row.get("k"),
            "question_session":  question_session,
            "question":          question,
            "model_answer":      model_answer,
            "topic":             row.get("topic", ""),
            "persona":           row.get("persona", ""),
            "preference":        row.get("preference", ""),
            "explanation":       row.get("explanation", ""),
            "judge_eligible":    eligible,
        }
        for name in ALL_SUBDIMS:
            entry[name] = prior.get(name) if isinstance(prior.get(name), int) else None

        qa_idx = len(qa_entries)
        qa_entries.append(entry)

        if not eligible:
            continue

        for name in active_subdims:
            cur = entry.get(name)
            if isinstance(cur, int) and cur >= 0:
                continue
            try:
                prompt_text = build_prompt_for_subdim(name, row)
            except Exception as e:
                logging.warning(f"  prompt build failed for {name}: {e}")
                continue
            flat_items[name].append({"qa_idx": qa_idx, "prompt": prompt_text})

    return qa_entries, flat_items


# ---------------- summary (post-hoc, from detail JSONs) ----------------
#
# The summary CSV is derived entirely from the per-folder detail JSONs
# (judge_*.json) rather than accumulated during the run. This mirrors the
# standalone "Detail JSON → CSV" cell in plot_judge_scores_.ipynb, and means a
# consistent summary is produced for EVERY engine (vllm / openai / openai-batch)
# and can be rebuilt at any time via --summary-only.
#
# Columns (no `avg_` prefix; `total` is the mean per-QA sum of the four sub-dims):
#   module_config, total, violation, acknowledgement, hallucination, helpful,
#   {subdim}_score_0, {subdim}_score_1, num_qa, num_valid_qa, num_failed_qa


def _is_filled(v: Any) -> bool:
    return isinstance(v, int) and v >= 0


def build_summary_csv_cols() -> List[str]:
    return (
        ["module_config", "total"]
        + ALL_SUBDIMS
        + [f"{s}_score_{v}" for s in ALL_SUBDIMS for v in (0, 1)]
        + ["num_qa", "num_valid_qa", "num_failed_qa"]
    )


def summary_row_from_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Build one CSV row from a single judge_*.json detail dict.

    A QA is "valid" only if it is judge-eligible AND every one of the four
    sub-dims is filled (>= 0). `total` and each sub-dim average are computed
    over the valid QAs only; empty string when there are none.
    """
    qa = detail.get("qa_entries", [])
    eligible = [q for q in qa if q.get("judge_eligible")]
    valid    = [q for q in eligible if all(_is_filled(q.get(s)) for s in ALL_SUBDIMS)]
    row: Dict[str, Any] = {
        "module_config": f"{detail['module']}_{detail['config_num']}",
        "num_qa":        len(qa),
        "num_valid_qa":  len(valid),
        "num_failed_qa": len(eligible) - len(valid),
    }
    for s in ALL_SUBDIMS:
        scores = [int(q[s]) for q in valid]
        row[s] = round(float(np.mean(scores)), 4) if scores else ""
        row[f"{s}_score_0"] = sum(1 for v in scores if v == 0)
        row[f"{s}_score_1"] = sum(1 for v in scores if v == 1)
    if valid:
        totals = [sum(int(q[s]) for s in ALL_SUBDIMS) for q in valid]
        row["total"] = round(float(np.mean(totals)), 4)
    else:
        row["total"] = ""
    return row


def build_summary_csv_from_details(out_dir: Path, summary_csv: Optional[Path] = None) -> int:
    """Scan judge_*.json detail files in out_dir and (re)write judge_summary.csv.

    Returns the number of rows written. Mirrors the standalone notebook cell, so
    the CSV can be regenerated post-hoc for any engine or after an interrupted run.
    """
    summary_csv = summary_csv or (out_dir / "judge_summary.csv")
    cols = build_summary_csv_cols()
    rows: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("judge_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                detail = json.load(f)
        except Exception as e:
            logging.warning(f"  [skip] {path.name} — could not read: {e}")
            continue
        if not all(k in detail for k in ("qa_entries", "module", "config_num")):
            logging.warning(f"  [skip] {path.name} — missing module/config_num/qa_entries")
            continue
        rows.append(summary_row_from_detail(detail))

    rows.sort(key=lambda r: r["module_config"])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"wrote {len(rows)} rows → {summary_csv}")
    for r in rows:
        logging.info(
            f"  {r['module_config']}: total={r['total']}  "
            f"valid={r['num_valid_qa']}/{r['num_qa']}"
        )
    return len(rows)


def build_summary_row(module: str, config_num: int, llm: str,
                      qa_entries: List[Dict[str, Any]],
                      active_subdims: List[str],
                      usage: Dict[str, int]) -> Dict[str, Any]:
    valid = [q for q in qa_entries if q.get("judge_eligible")]
    valid_filled = [
        q for q in valid
        if all(_is_filled(q.get(name)) for name in active_subdims)
    ]
    row: Dict[str, Any] = {
        "module_config": f"{module}_{config_num}",
        "module":        module,
        "config_num":    config_num,
        "llm":           llm,
        "num_qa":        len(qa_entries),
        "num_valid_qa":  len(valid_filled),
        "num_failed_qa": len(valid) - len(valid_filled),
    }
    for name in ALL_SUBDIMS:
        scores = [int(q[name]) for q in valid_filled
                  if _is_filled(q.get(name)) and name in active_subdims]
        if scores:
            row[f"avg_{name}"]      = round(float(np.mean(scores)), 4)
            row[f"{name}_score_0"] = sum(1 for v in scores if v == 0)
            row[f"{name}_score_1"] = sum(1 for v in scores if v == 1)
        else:
            row[f"avg_{name}"]      = ""
            row[f"{name}_score_0"] = ""
            row[f"{name}_score_1"] = ""
    if active_subdims and valid_filled:
        totals = [sum(int(q[n]) for n in active_subdims) for q in valid_filled]
        row["avg_total"] = round(float(np.mean(totals)), 4)
    else:
        row["avg_total"] = ""
    row["judge_calls"]             = usage.get("calls", 0)
    row["judge_prompt_tokens"]     = usage.get("prompt_tokens", 0)
    row["judge_completion_tokens"] = usage.get("completion_tokens", 0)
    return row


# ---------------- main per-folder eval ----------------

def evaluate_folder(
    module: str,
    config_num: int,
    cfg_dir: Path,
    llm: str,
    active_subdims: List[str],
    detail_path: Path,
    judge_client_factory,
    batch_size: int,
    overwrite: bool,
    max_concurrent: int = 10,
) -> Tuple[Dict[str, Any], int]:
    """Returns (summary_row, num_new_judgments)."""
    qa_entries, flat_items = _load_folder_qa(cfg_dir, detail_path, active_subdims, overwrite)
    if qa_entries is None:
        logging.warning(f"  no results.jsonl in {cfg_dir}")
        return {}, 0

    n_new = sum(len(v) for v in flat_items.values())
    if n_new == 0:
        logging.info("  nothing new to judge — emitting summary only")
    else:
        per = "  ".join(f"{n}={len(flat_items[n])}" for n in active_subdims)
        logging.info(f"  to judge: {per}")

    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    if n_new > 0:
        judge_client = judge_client_factory()

        for name in active_subdims:
            items = flat_items[name]
            if not items:
                continue
            spec = SUBDIM_SPEC[name]
            parse_fn = make_parser(name)
            n_items = len(items)
            scores: List[Optional[int]] = [None] * n_items
            failed: List[int] = []

            for batch_start in tqdm(
                range(0, n_items, batch_size),
                desc=f"  {module}_{config_num} {name}",
            ):
                batch = items[batch_start:batch_start + batch_size]
                prompts = [it["prompt"] for it in batch]
                try:
                    raw_texts, batch_usages = judge_client.generate_batch_raw(
                        prompts=prompts,
                        temperature=_DEFAULT_TEMP,
                        max_tokens=spec["max_tokens"],
                        return_usage=True,
                        max_concurrent=max_concurrent,
                    )
                    usage["calls"]             += len(raw_texts)
                    usage["prompt_tokens"]     += sum(int(u.get("prompt_tokens", 0))     for u in batch_usages)
                    usage["completion_tokens"] += sum(int(u.get("completion_tokens", 0)) for u in batch_usages)
                    for j, raw in enumerate(raw_texts):
                        idx = batch_start + j
                        s = parse_fn(raw)
                        if s is None:
                            failed.append(idx)
                        else:
                            scores[idx] = s
                except Exception as e:
                    logging.warning(
                        f"  {name} batch [{batch_start}:{batch_start+len(batch)}] failed: {e}"
                    )
                    for j in range(len(batch)):
                        failed.append(batch_start + j)

            if failed:
                logging.warning(f"  {name}: retrying {len(failed)} items individually")
                for i in tqdm(failed, desc=f"  retry {name}"):
                    s = _retry_with_temp(
                        prompt=items[i]["prompt"],
                        parse_fn=parse_fn,
                        judge_client=judge_client,
                        max_tokens=spec["max_tokens"],
                        usage=usage,
                    )
                    if s is not None:
                        scores[i] = s

            for i, item in enumerate(items):
                final = scores[i] if scores[i] is not None else -1
                qa_entries[item["qa_idx"]][name] = final

    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "module": module,
            "config_num": config_num,
            "llm": llm,
            "active_subdims": active_subdims,
            "usage_this_run": usage,
            "qa_entries": qa_entries,
        }, f, indent=2, ensure_ascii=False)
    logging.info(f"  saved → {detail_path}")

    summary = build_summary_row(module, config_num, llm, qa_entries, active_subdims, usage)
    return summary, n_new


# ---------------- OpenAI Batch (hybrid) ----------------

# Batch API constants
_BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _build_batch_input_file(
    global_requests: List[Dict[str, Any]],
    judge_model: str,
    path: Path,
    temperature: float = _DEFAULT_TEMP,
) -> None:
    """Write a JSONL file in the format OpenAI's Batch API expects.

    For reasoning models (gpt-5*, o1*, o3*, o4*) the per-request body is rewritten
    to use `max_completion_tokens` (with a 4096 floor) and to drop temperature.
    """
    is_reasoning = _is_reasoning_model(judge_model)
    if is_reasoning:
        logging.info(
            f"  reasoning model detected ({judge_model}): "
            "using max_completion_tokens, dropping temperature"
        )
    with open(path, "w", encoding="utf-8") as f:
        for req in global_requests:
            body: Dict[str, Any] = {
                "model":       judge_model,
                "messages":    [{"role": "user", "content": req["prompt"]}],
                "max_tokens":  req["max_tokens"],
                "temperature": temperature,
            }
            _adjust_for_reasoning_model(body)
            line = {
                "custom_id": req["custom_id"],
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body":      body,
            }
            f.write(json.dumps(line, ensure_ascii=True) + "\n")


def _submit_batch(client, batch_input_path: Path, completion_window: str):
    with open(batch_input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch


def _download_batch_records(client, output_file_id: str) -> List[Dict[str, Any]]:
    resp = client.files.content(output_file_id)
    if hasattr(resp, "text") and isinstance(resp.text, str):
        text = resp.text
    elif hasattr(resp, "content"):
        raw = resp.content
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    else:
        text = str(resp)
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logging.warning(f"  bad batch output line skipped: {e}")
    return records


def _extract_batch_result(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, int]], Optional[str]]:
    """Returns (text, usage_dict, error_msg). text is None on any failure."""
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
        "prompt_tokens":     int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }
    return text, usage_dict, None


def run_hybrid_batch_mode(
    args,
    candidates: List[Tuple[str, int, Path]],
    out_dir: Path,
    active_subdims: List[str],
) -> None:
    """One OpenAI batch per (module, config) folder, submitted in parallel.

    Each folder's detail JSON is written as soon as its batch finishes; sync
    fallback handles parse failures within that folder. Summary CSV is NOT
    written here — derive it from the detail JSONs separately.

    Per-batch sizing assumption: each results.jsonl × #criteria stays under
    OpenAI's per-batch limits (50K requests / 200MB).
    """
    try:
        import openai
    except ImportError as e:
        raise ImportError("openai package required for --engine openai-batch") from e

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Pass --api-key or set OPENAI_API_KEY env var."
        )
    client = openai.OpenAI(api_key=api_key)

    state_path = out_dir / "batch_state.json"
    existing_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for entry in raw.get("batches", []):
                key = (entry["module"], int(entry["config_num"]))
                existing_by_key[key] = entry
            logging.info(f"Loaded {len(existing_by_key)} prior batch entries from {state_path.name}")
        except Exception as e:
            logging.warning(f"Could not load {state_path.name}: {e}")

    # -------- Phase 1: build per-folder jobs --------
    logging.info("== Phase 1: building per-folder batches ==")
    folder_jobs: List[Dict[str, Any]] = []

    for module, cnum, cdir in candidates:
        mc = f"{module}_{cnum}"
        detail_path = out_dir / f"judge_{mc}_{args.llm}.json"
        logging.info(f"── {mc}  dir={cdir.name}")

        if detail_path.exists() and not args.overwrite:
            try:
                with open(detail_path, "r", encoding="utf-8") as f:
                    prior = json.load(f)
                qa_prev = prior.get("qa_entries", [])
                fully_done = qa_prev and all(
                    (not q.get("judge_eligible")) or
                    all(_is_filled(q.get(n)) for n in active_subdims)
                    for q in qa_prev
                )
                if fully_done:
                    logging.info("  already complete — skipping")
                    continue
            except Exception as e:
                logging.warning(f"  could not inspect prior file: {e}")

        qa_entries, flat_items = _load_folder_qa(cdir, detail_path, active_subdims, args.overwrite)
        if qa_entries is None:
            logging.warning(f"  no results.jsonl in {cdir} — skipping")
            continue

        requests: List[Dict[str, Any]] = []
        for subdim, items in flat_items.items():
            for item in items:
                requests.append({
                    "custom_id":  f"req-{len(requests):08d}",
                    "prompt":     item["prompt"],
                    "subdim":     subdim,
                    "qa_idx":     item["qa_idx"],
                    "max_tokens": SUBDIM_SPEC[subdim]["max_tokens"],
                })

        job = {
            "module":        module,
            "config_num":    cnum,
            "detail_path":   detail_path,
            "qa_entries":    qa_entries,
            "requests":      requests,
            "input_path":    out_dir / f"batch_input_{mc}.jsonl",
            "usage":         {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "batch_id":      None,
            "input_file_id": None,
        }

        if not requests:
            # No new prompts (e.g. resumed and only -1s left to fill but overwrite=False);
            # just persist whatever's already in qa_entries.
            _save_folder_detail(job, args, active_subdims)
            logging.info("  no new prompts — wrote detail JSON only")
            continue

        folder_jobs.append(job)
        logging.info(f"  queued {len(requests)} prompts ({mc})")

    if not folder_jobs:
        logging.info("Nothing to submit — done")
        return

    # -------- Phase 2: submit (or resume) per-folder batches --------
    logging.info(f"== Phase 2: submitting/resuming {len(folder_jobs)} batches ==")
    for job in folder_jobs:
        mc = f"{job['module']}_{job['config_num']}"
        key = (job["module"], job["config_num"])
        prior = existing_by_key.get(key)
        if prior and prior.get("batch_id"):
            try:
                batch = client.batches.retrieve(prior["batch_id"])
                if batch.status in ("expired", "cancelled", "failed"):
                    logging.warning(
                        f"  [{mc}] prior batch {prior['batch_id']} is {batch.status} — resubmitting"
                    )
                else:
                    job["batch_id"]      = prior["batch_id"]
                    job["input_file_id"] = prior.get("input_file_id")
                    logging.info(f"  [{mc}] resumed batch {batch.id} (status={batch.status})")
                    continue
            except Exception as e:
                logging.warning(f"  [{mc}] could not retrieve prior batch: {e}")

        _build_batch_input_file(job["requests"], args.judge_model, job["input_path"])
        batch = _submit_batch(client, job["input_path"], args.completion_window)
        job["batch_id"]      = batch.id
        job["input_file_id"] = batch.input_file_id
        logging.info(f"  [{mc}] submitted batch {batch.id} ({len(job['requests'])} requests)")

    _save_batch_state(state_path, folder_jobs, args)

    # -------- Phase 3: poll all batches concurrently, process each as it finishes --------
    logging.info(f"== Phase 3: polling {len(folder_jobs)} batches every {args.poll_interval}s ==")
    pending: Dict[Tuple[str, int], Dict[str, Any]] = {
        (j["module"], j["config_num"]): j for j in folder_jobs
    }
    failed_jobs: List[str] = []

    while pending:
        finished_keys: List[Tuple[str, int]] = []
        for key, job in list(pending.items()):
            mc = f"{job['module']}_{job['config_num']}"
            try:
                batch = client.batches.retrieve(job["batch_id"])
            except Exception as e:
                logging.warning(f"  [{mc}] retrieve failed: {e}")
                continue
            counts    = getattr(batch, "request_counts", None)
            completed = getattr(counts, "completed", "?") if counts else "?"
            total     = getattr(counts, "total",     "?") if counts else "?"
            failed    = getattr(counts, "failed",    "?") if counts else "?"
            logging.info(
                f"  [{mc}] batch={job['batch_id']} status={batch.status} "
                f"progress={completed}/{total} failed={failed}"
            )

            if batch.status in _BATCH_TERMINAL_STATUSES:
                if batch.status == "completed":
                    try:
                        _process_completed_batch(client, job, batch.output_file_id,
                                                 args, api_key, active_subdims)
                    except Exception as e:
                        logging.error(f"  [{mc}] processing failed: {e}", exc_info=True)
                        failed_jobs.append(mc)
                else:
                    logging.error(
                        f"  [{mc}] ended with status={batch.status} — "
                        "leaving for next run (will resubmit on rerun)"
                    )
                    failed_jobs.append(mc)
                finished_keys.append(key)

        for key in finished_keys:
            del pending[key]
        _save_batch_state(state_path, folder_jobs, args)

        if pending:
            time.sleep(args.poll_interval)

    # -------- Phase 4: cleanup state file if everything saved out --------
    all_saved = all(Path(j["detail_path"]).exists() for j in folder_jobs)
    if all_saved and not failed_jobs and state_path.exists():
        try:
            state_path.unlink()
        except Exception:
            pass

    # -------- Phase 5: (re)build judge_summary.csv from all detail JSONs --------
    build_summary_csv_from_details(out_dir)

    if failed_jobs:
        logging.warning(
            f"Done with {len(failed_jobs)} failed folder(s): {failed_jobs}. "
            "Re-run the same command to retry — failed batches will be resubmitted."
        )
    else:
        logging.info(f"\nDone → {out_dir}")


def _process_completed_batch(
    client,
    job: Dict[str, Any],
    output_file_id: str,
    args,
    api_key: str,
    active_subdims: List[str],
) -> None:
    """Download a completed batch's output, parse, sync-retry failures, save detail JSON."""
    mc = f"{job['module']}_{job['config_num']}"
    logging.info(f"  [{mc}] downloading results...")
    records = _download_batch_records(client, output_file_id)
    logging.info(f"  [{mc}] parsed {len(records)} records")

    results_by_cid: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("custom_id")
        if not cid:
            continue
        text, usage_dict, err = _extract_batch_result(rec)
        results_by_cid[cid] = {"text": text, "usage": usage_dict, "error": err}

    sync_retry_queue: List[Dict[str, Any]] = []
    for req in job["requests"]:
        res = results_by_cid.get(req["custom_id"])
        if res is None or res["text"] is None:
            sync_retry_queue.append(req)
            continue
        if res["usage"]:
            job["usage"]["calls"]             += 1
            job["usage"]["prompt_tokens"]     += res["usage"]["prompt_tokens"]
            job["usage"]["completion_tokens"] += res["usage"]["completion_tokens"]
        parse_fn = make_parser(req["subdim"])
        score = parse_fn(res["text"])
        if score is None:
            sync_retry_queue.append(req)
        else:
            job["qa_entries"][req["qa_idx"]][req["subdim"]] = score

    n_filled = len(job["requests"]) - len(sync_retry_queue)
    logging.info(
        f"  [{mc}] filled {n_filled}/{len(job['requests'])} from batch; "
        f"{len(sync_retry_queue)} need sync retry"
    )

    if sync_retry_queue:
        sync_client = UnifiedLLMClient(
            engine="openai", model_name=args.judge_model, api_key=api_key,
        )
        for req in tqdm(sync_retry_queue, desc=f"  sync-retry {mc}"):
            parse_fn = make_parser(req["subdim"])
            score = _retry_with_temp(
                prompt=req["prompt"],
                parse_fn=parse_fn,
                judge_client=sync_client,
                max_tokens=req["max_tokens"],
                usage=job["usage"],
            )
            job["qa_entries"][req["qa_idx"]][req["subdim"]] = (
                score if score is not None else -1
            )

    _save_folder_detail(job, args, active_subdims)


def _save_folder_detail(
    job: Dict[str, Any],
    args,
    active_subdims: List[str],
) -> None:
    """Fill any remaining None scores with -1 and write the folder's detail JSON.
    Does NOT touch summary CSV.
    """
    for entry in job["qa_entries"]:
        for subdim in active_subdims:
            if entry.get(subdim) is None:
                entry[subdim] = -1

    detail_path: Path = job["detail_path"]
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    with open(detail_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump({
            "module":         job["module"],
            "config_num":     job["config_num"],
            "llm":            args.llm,
            "active_subdims": active_subdims,
            "usage_this_run": job["usage"],
            "qa_entries":     job["qa_entries"],
        }, f, indent=2, ensure_ascii=False)
    mc = f"{job['module']}_{job['config_num']}"
    logging.info(f"  [{mc}] saved → {detail_path}")


def _save_batch_state(state_path: Path, folder_jobs: List[Dict[str, Any]], args) -> None:
    """Persist all in-flight batch ids so a subsequent run can resume."""
    payload = {
        "judge_model": args.judge_model,
        "llm":         args.llm,
        "batches": [
            {
                "module":        j["module"],
                "config_num":    j["config_num"],
                "batch_id":      j["batch_id"],
                "input_file_id": j["input_file_id"],
                "n_requests":    len(j["requests"]),
            }
            for j in folder_jobs if j.get("batch_id")
        ],
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------- main ----------------

def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    root = Path(args.root) if args.root else script_dir.parent

    active_subdims: List[str] = list(args.criteria)

    judge_short = args.judge_model.split("/")[-1]
    out_dir = script_dir / f"{args.llm}_judge_{judge_short}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "judge_summary.csv"

    logging.info(f"Root        : {root}")
    logging.info(f"LLM         : {args.llm}")
    logging.info(f"Engine      : {args.engine}")
    logging.info(f"Judge model : {args.judge_model}")
    logging.info(f"Criteria    : {active_subdims}")
    logging.info(f"Output dir  : {out_dir}")

    # --summary-only: skip judging; just (re)build judge_summary.csv from the
    # detail JSONs already present in out_dir (same as the notebook cell).
    if args.summary_only:
        logging.info("== summary-only: rebuilding judge_summary.csv from detail JSONs ==")
        n = build_summary_csv_from_details(out_dir, summary_csv)
        if n == 0:
            logging.warning(f"No judge_*.json detail files found in {out_dir}")
        return

    candidates: List[Tuple[str, int, Path]] = []
    for module in args.modules:
        module_dir = root / module
        if not module_dir.is_dir():
            logging.info(f"[skip] module folder not found: {module}/")
            continue
        cfgs = find_config_dirs(module_dir, args.llm)
        if not cfgs:
            logging.info(f"[skip] no config_*_outputs_{args.llm} under {module}/")
            continue
        for cnum, cdir in cfgs:
            candidates.append((module, cnum, cdir))

    if not candidates:
        logging.error("No candidate folders found.")
        return

    # Branch: OpenAI Batch hybrid mode — one batch per (module, config) folder,
    # all submitted in parallel. Writes per-folder detail JSON and, at the end,
    # rebuilds judge_summary.csv from those detail JSONs (same as the sync path).
    if args.engine == "openai-batch":
        run_hybrid_batch_mode(
            args=args,
            candidates=candidates,
            out_dir=out_dir,
            active_subdims=active_subdims,
        )
        return

    judge_client_holder: Dict[str, Any] = {"client": None}
    def judge_client_factory():
        if judge_client_holder["client"] is None:
            if args.engine == "vllm":
                logging.info(f"Loading judge LLM (vLLM): {args.judge_model}")
                judge_client_holder["client"] = UnifiedLLMClient(
                    engine="vllm",
                    model_path=args.judge_model,
                    tensor_parallel_size=args.tensor_parallel,
                    gpu_memory_utilization=args.gpu_memory,
                    max_model_len=args.max_model_len,
                )
            elif args.engine == "openai":
                logging.info(f"Initializing OpenAI judge client: {args.judge_model}")
                judge_client_holder["client"] = UnifiedLLMClient(
                    engine="openai",
                    model_name=args.judge_model,
                    api_key=args.api_key,
                )
            else:
                raise ValueError(f"unsupported engine: {args.engine}")
        return judge_client_holder["client"]

    for module, cnum, cdir in candidates:
        mc = f"{module}_{cnum}"
        detail_path = out_dir / f"judge_{mc}_{args.llm}.json"
        logging.info(f"\n── {mc}  dir={cdir.name}")

        if detail_path.exists() and not args.overwrite:
            try:
                with open(detail_path, "r", encoding="utf-8") as f:
                    prior = json.load(f)
                qa_prev = prior.get("qa_entries", [])
                fully_done = qa_prev and all(
                    (not q.get("judge_eligible")) or
                    all(_is_filled(q.get(n)) for n in active_subdims)
                    for q in qa_prev
                )
                if fully_done:
                    logging.info("  already complete — skipping (use --overwrite to redo)")
                    continue
            except Exception as e:
                logging.warning(f"  could not inspect prior file: {e}")

        summary, _ = evaluate_folder(
            module=module, config_num=cnum, cfg_dir=cdir,
            llm=args.llm, active_subdims=active_subdims,
            detail_path=detail_path,
            judge_client_factory=judge_client_factory,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            max_concurrent=args.max_concurrent,
        )
        if not summary:
            continue
        avg_parts = "  ".join(
            f"{n}={summary.get(f'avg_{n}', 'n/a')}" for n in active_subdims
        )
        logging.info(
            f"  valid={summary['num_valid_qa']}  failed={summary['num_failed_qa']}  "
            f"avg_total={summary['avg_total']}  | {avg_parts}  "
            f"calls={summary['judge_calls']}  "
            f"in_tok={summary['judge_prompt_tokens']}  "
            f"out_tok={summary['judge_completion_tokens']}"
        )

    # Build judge_summary.csv post-hoc from all detail JSONs (same format/builder
    # as the openai-batch path and the plot_judge_scores_ notebook cell).
    build_summary_csv_from_details(out_dir, summary_csv)

    logging.info(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
