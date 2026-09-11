# Auto-merges per-session experiment results, then LLM-judges them.
#
# Given --llm / --subset / --session-num, scans every module folder under implexconv/ for a
# `config_*_{llm}_{subset}` output dir, concatenates sessions 0..(session-num-1) into
#   evaluation/{llm}_results/opp_{session-num}/{module}/results_{llm}_{subset}_merged.json
# and judges each module (one summary-CSV row per module, keyed by module name).
# Only --subset opposed is supported (judge prompts are opposed-specific).
#
# Usage examples:
#
# vLLM (local GPU) — judge-model is an HF path:
#   python evaluation_llm_judge.py \
#       --llm Qwen3-1.7B --subset opposed --session-num 10 \
#       --engine vllm --judge-model meta-llama/Llama-3.1-8B-Instruct \
#       --tensor-parallel 1 --gpu-memory 0.9 --max-model-len 8192 \
#       --dims 1 2 --batch-size 32
#
# OpenAI Batch API — judge-model is an OpenAI model name:
#   export OPENAI_API_KEY=sk-...            # or point --api-key-env at another env var
#   python evaluation_llm_judge.py \
#       --llm Qwen3-1.7B --subset opposed --session-num 10 \
#       --engine openai-batch --judge-model gpt-4o-mini \
#       --poll-interval 60 --completion-window 24h \
#       --dims 1 2 --batch-size 32
#
# --dims meaning:
#   1 = response_competence  (rc_question_addressing)
#   2 = persona_adaptation   (pa_persona_recognition / pa_generic_distinctness / pa_substantive_integration)

import argparse
import fcntl
import gzip
import importlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_module.llm_client import UnifiedLLMClient, _parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

DATASET_PATH = (
    Path(__file__).parent.parent / "dataset/implexconv/ImplexConv_opposed_processed.json.gz"
)
FALLBACK_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_TOP_K = 3

_RETRY_TEMP_START = 0.3
_RETRY_TEMP_STEP  = 0.025
_RETRY_MAX        = 30
_CLIENT_JSON_RETRY = 30

# ---------------- OpenAI Batch constants ----------------
_BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_REASONING_MAX_TOKENS_FLOOR = 4096


# ---------------- sub-dim meta ----------------
# Static metadata (group, max, max_tokens) + names of symbols expected in the prompt module.
# The actual system_prompt / user_template / json_schema are resolved at runtime from the
# selected prompt module via build_subdim_spec().
SUBDIM_META: Dict[str, Dict[str, Any]] = {
    "rc_question_addressing": {
        "group": 1, "max": 1, "max_tokens": 200,
        "system_attr": "SYSTEM_PROMPT_RC",
        "template_attr": "USER_PROMPT_TEMPLATE_RC1",
        "schema_attr": "JUDGE_JSON_SCHEMA_RC1",
    },
    "pa_persona_recognition": {
        "group": 2, "max": 1, "max_tokens": 250,
        "system_attr": "SYSTEM_PROMPT_PA",
        "template_attr": "USER_PROMPT_TEMPLATE_PA1",
        "schema_attr": "JUDGE_JSON_SCHEMA_PA1",
    },
    "pa_generic_distinctness": {
        "group": 2, "max": 1, "max_tokens": 250,
        "system_attr": "SYSTEM_PROMPT_PA",
        "template_attr": "USER_PROMPT_TEMPLATE_PA2",
        "schema_attr": "JUDGE_JSON_SCHEMA_PA2",
    },
    "pa_substantive_integration": {
        "group": 2, "max": 1, "max_tokens": 250,
        "system_attr": "SYSTEM_PROMPT_PA",
        "template_attr": "USER_PROMPT_TEMPLATE_PA3",
        "schema_attr": "JUDGE_JSON_SCHEMA_PA3",
    },
}

GROUP_TO_SUBDIMS: Dict[int, List[str]] = {1: [], 2: []}
for _name, _meta in SUBDIM_META.items():
    GROUP_TO_SUBDIMS[_meta["group"]].append(_name)

GROUP_NAME = {1: "response_competence", 2: "persona_adaptation"}


ALL_SUBDIMS: List[str] = list(SUBDIM_META.keys())


def build_subdim_spec(prompt_mod, active_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Bind system_prompt/template/schema for `active_names` from the given prompt module."""
    spec: Dict[str, Dict[str, Any]] = {}
    missing: List[Tuple[str, List[str]]] = []
    for name in active_names:
        meta = SUBDIM_META[name]
        attrs = [meta["system_attr"], meta["template_attr"], meta["schema_attr"]]
        my_missing = [a for a in attrs if not hasattr(prompt_mod, a)]
        if my_missing:
            missing.append((name, my_missing))
            continue
        spec[name] = {
            "group":      meta["group"],
            "system":     getattr(prompt_mod, meta["system_attr"]),
            "template":   getattr(prompt_mod, meta["template_attr"]),
            "schema":     getattr(prompt_mod, meta["schema_attr"]),
            "max":        meta["max"],
            "max_tokens": meta["max_tokens"],
        }
    if missing:
        details = "\n".join(f"  - {n}: missing {', '.join(s)}" for n, s in missing)
        raise ImportError(
            f"prompt module '{prompt_mod.__name__}' is missing required symbols:\n{details}"
        )
    return spec


def build_summary_cols() -> List[str]:
    cols = ["model", "llm", "subset", "num_valid_qa", "num_failed_qa"]
    for name in ALL_SUBDIMS:
        cols.append(f"avg_{name}")
        max_s = SUBDIM_META[name]["max"]
        for s in range(max_s + 1):
            cols.append(f"{name}_score_{s}")
    cols.append("avg_total")
    cols += ["judge_calls", "judge_prompt_tokens", "judge_completion_tokens"]
    return cols


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge"
    )
    parser.add_argument("--llm", required=True,
                        help="LLM tag, e.g. Qwen3-1.7B. Selects experiment output folders named "
                             "config_*_{llm}_{subset} and is used as the summary-CSV model column.")
    parser.add_argument("--subset", default="opposed",
                        help="Dataset subset (only 'opposed' supported; judge prompts are opposed-specific)")
    parser.add_argument("--judge-model", required=True,
                        help="Judge model path/name (vllm: HF path, openai: model name)")
    parser.add_argument("--session-num", type=int, required=True,
                        help="Number of sessions to merge & judge: sessions 0..(session-num-1)")
    parser.add_argument("--dims", nargs="+", type=int, choices=[1, 2], default=[1, 2],
                        help="Groups: 1=response_competence 2=persona_adaptation")
    parser.add_argument("--prompt-module", default="prompt",
                        help="Python module name (under evaluation/) that exports the prompt "
                             "templates and JSON schemas. Default: prompt. "
                             "E.g. --prompt-module prompt_1 loads prompt_1.py.")
    parser.add_argument("--engine", choices=["vllm", "openai", "openai-batch", "elice"], default="vllm",
                        help="Inference engine (default: vllm). elice = OpenAI-compatible "
                             "reasoning model via custom base_url (temp=1, max_completion_tokens=4000). "
                             "openai-batch = submits one OpenAI Batch per model (~50%% cheaper, "
                             "up to 24h latency); parse failures are sync-retried via the OpenAI client.")
    # vLLM-only options (ignored when --engine openai)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    # OpenAI-only options (ignored when --engine vllm)
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint base URL (e.g. https://mlapi.run/<id>)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Env var name holding the API key (default: OPENAI_API_KEY)")
    parser.add_argument("--batch-size", type=int, default=32)
    # openai-batch-only options
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="(openai-batch only) seconds between batch status polls (default: 60)")
    parser.add_argument("--completion-window", default="24h",
                        help="(openai-batch only) batch completion window (default: 24h)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these model folder names (exact match). "
                             "If omitted, all model folders under the input dir are evaluated.")
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> Dict:
    _open = gzip.open if Path(dataset_path).suffix == ".gz" else open
    with _open(dataset_path, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    session_map: Dict[int, Dict] = {}
    for session in raw:
        sid = session["metadata"]["session_id"]
        convs_by_id: Dict[int, list] = {}
        for turn in session["conversations"]:
            convs_by_id.setdefault(turn["conv_id"], []).append(turn)
        qa_by_question = {qa["question"]: qa for qa in session["qa"]}
        session_map[sid] = {
            "convs_by_id":    convs_by_id,
            "qa_by_question": qa_by_question,
        }

    logging.info(f"Loaded dataset: {len(session_map)} sessions")
    return session_map


class ConvEmbeddingIndex:
    def __init__(self, model_name: str = FALLBACK_EMB_MODEL):
        self.model_name = model_name
        self._model = None
        self._cache: Dict[int, Tuple[List[int], np.ndarray]] = {}

    def _ensure_model(self):
        if self._model is None:
            logging.info(f"Loading fallback embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def _ensure_session(self, sid: int, convs_by_id: Dict):
        if sid in self._cache:
            return
        self._ensure_model()
        conv_ids = sorted(convs_by_id.keys())
        texts = [
            " ".join(
                t["utterance"]
                for t in sorted(convs_by_id[cid], key=lambda x: x["turn_id"])
            )
            for cid in conv_ids
        ]
        embs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self._cache[sid] = (conv_ids, embs)

    def top_k(self, query: str, sid: int, convs_by_id: Dict,
               k: int = FALLBACK_TOP_K) -> List[int]:
        from sklearn.metrics.pairwise import cosine_similarity
        self._ensure_session(sid, convs_by_id)
        conv_ids, embs = self._cache[sid]
        if not conv_ids:
            return []
        q_emb = self._model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        sims = cosine_similarity(q_emb, embs)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [conv_ids[i] for i in top_idx]


def format_reference_conv(conv_ids: List[int], convs_by_id: Dict) -> str:
    parts = []
    for cid in conv_ids:
        turns = convs_by_id.get(cid, [])
        if not turns:
            continue
        parts.append(f"====conv_id = {cid}====")
        for turn in sorted(turns, key=lambda x: x["turn_id"]):
            speaker = "User" if turn["speaker"] == "user" else "Agent"
            parts.append(f"{speaker}: {turn['utterance']}")
    return "\n".join(parts) if parts else "(no reference conversations available)"


def _extract_int_field(raw: Any, field: str, max_val: int) -> Optional[int]:
    try:
        data = _parse_json_response(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return None
        val = data.get(field)
        if isinstance(val, int) and 0 <= val <= max_val:
            return val
        return None
    except Exception:
        return None


def make_parser(field: str, max_val: int):
    def _parse(raw: Any) -> Optional[int]:
        return _extract_int_field(raw, field, max_val)
    return _parse


def _accumulate_usage(usage: Optional[Dict], u: Optional[Dict]) -> None:
    if usage is None or u is None:
        return
    usage["calls"] = usage.get("calls", 0) + 1
    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + int(u.get("prompt_tokens", 0))
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + int(u.get("completion_tokens", 0))


def _retry_with_temp(
    prompt: str,
    system_prompt: str,
    schema: Dict,
    parse_fn,
    judge_client,
    max_tokens: int,
    usage: Optional[Dict] = None,
) -> Optional[int]:
    temperature = _RETRY_TEMP_START
    for attempt in range(_RETRY_MAX):
        try:
            out = judge_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                guided_json=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                json_retry=_CLIENT_JSON_RETRY,
                return_usage=True,
            )
            u = out.pop('_usage', None) if isinstance(out, dict) else None
            result = out
            _accumulate_usage(usage, u)
            score = parse_fn(result)
            if score is not None:
                return score
        except Exception as e:
            logging.warning(f"    retry attempt {attempt + 1} failed (temp={temperature:.3f}): {e}")
        temperature += _RETRY_TEMP_STEP
    return None


def get_model_name(folder_name: str, session_num: int) -> str:
    suffix = f"_{session_num}"
    return folder_name[: -len(suffix)] if folder_name.endswith(suffix) else folder_name


def find_result_file(model_dir: Path) -> Optional[Path]:
    files = sorted(model_dir.glob("results_*.json"))
    return files[0] if files else None


def get_llm_from_data(data: List[Dict]) -> str:
    for session in data:
        for qa in session.get("qa_results", []):
            model_field = qa.get("qa_tokens", {}).get("model")
            if model_field:
                return model_field.split("/")[-1]
    return "unknown"


# ---------------- auto-merge: gather per-session results per module ----------------

def collect_sessions_in_range(config_dir: Path, llm: str, subset: str, n: int) -> List[Dict]:
    """Full session-result objects from session_*/results_{llm}_{subset}_session_*.json
    whose session range falls within [0, n-1]."""
    out: List[Dict] = []
    for rf in sorted(config_dir.glob(f"session_*/results_{llm}_{subset}_session_*.json")):
        parts = rf.stem.split("_session_")
        if len(parts) != 2:
            continue
        nums = parts[1].split("_")
        if len(nums) != 2:
            continue
        try:
            start, end = int(nums[0]), int(nums[1])
        except ValueError:
            continue
        if start < 0 or end > n - 1:
            continue
        with open(rf, "r", encoding="utf-8") as f:
            out.extend(json.load(f))
    return out


def build_merged_inputs(parent_dir: Path, input_dir: Path, llm: str, subset: str, n: int) -> None:
    """For each module folder under parent_dir, find config_*{llm}_{subset} output folder(s),
    concatenate sessions 0..n-1 (dedup by session_id), and write one merged results file to
    input_dir/{module}/results_{llm}_{subset}_merged.json."""
    tag = f"{llm}_{subset}"
    n_modules = 0
    for module_dir in sorted(p for p in parent_dir.iterdir() if p.is_dir()):
        config_dirs = [
            d for d in sorted(module_dir.iterdir())
            if d.is_dir() and d.name.startswith("config_") and d.name.endswith(tag)
        ]
        if not config_dirs:
            continue

        seen: Set[int] = set()
        merged: List[Dict] = []
        for cdir in config_dirs:
            for s in collect_sessions_in_range(cdir, llm, subset, n):
                sid = s.get("session_id")
                if sid in seen:
                    continue
                seen.add(sid)
                merged.append(s)

        if not merged:
            logging.warning(
                f"[merge] {module_dir.name}: matched {[c.name for c in config_dirs]} "
                f"but no session results in range 0..{n - 1}"
            )
            continue

        missing = sorted(set(range(n)) - seen)
        if missing:
            logging.warning(f"[merge] {module_dir.name}: missing sessions {missing}")

        out_dir = input_dir / module_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"results_{llm}_{subset}_merged.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        logging.info(
            f"[merge] {module_dir.name}: {len(merged)} session(s) "
            f"from {len(config_dirs)} config dir(s) → {out_path}"
        )
        n_modules += 1

    if n_modules == 0:
        raise FileNotFoundError(
            f"No module produced merged inputs: found no '*/config_*{tag}' output folders "
            f"with sessions 0..{n - 1} under {parent_dir}"
        )


def is_valid_qa(generated: Any, gt: Any) -> bool:
    def ok(v):
        return v is not None and str(v).strip() not in ("", "N/A")
    return ok(generated) and ok(gt)


def upsert_to_csv(csv_path: Path, row: Dict, columns: List[str], key_cols: List[str]):
    # Serialize concurrent writers via a sidecar lock file (fcntl.flock).
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, dtype=str).fillna("")
                except Exception:
                    df = pd.DataFrame(columns=columns)
            else:
                df = pd.DataFrame(columns=columns)

            for col in columns:
                if col not in df.columns:
                    df[col] = ""

            df = df[columns]
            if key_cols:
                mask = pd.Series(True, index=df.index)
                for col in key_cols:
                    mask &= df[col].astype(str) == str(row.get(col, ""))
                df = df[~mask]

            new_row = pd.DataFrame([{c: row.get(c, "") for c in columns}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False, encoding="utf-8")
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _is_filled(v: Any) -> bool:
    return isinstance(v, int) and v >= 0


def normalize_session_scores(
    session_scores: List[Dict],
    active_subdims: List[str],
) -> Tuple[int, List[Dict]]:
    attempted_items = 0
    valid_entries: List[Dict] = []

    for s_entry in session_scores:
        s_entry["evaluated_subdims"] = active_subdims
        for q_entry in s_entry.get("qa_scores", []):
            eligible = q_entry.get("judge_eligible")
            if not isinstance(eligible, bool):
                eligible = is_valid_qa(
                    q_entry.get("generated_answer"),
                    q_entry.get("ground_truth_answer"),
                )
                q_entry["judge_eligible"] = eligible

            if not eligible:
                q_entry["total"] = None
                q_entry["valid"] = False
                continue

            attempted_items += 1
            filled = [name for name in ALL_SUBDIMS if _is_filled(q_entry.get(name))]
            if filled:
                q_entry["total"] = sum(int(q_entry[name]) for name in filled)
                q_entry["valid"] = True
                valid_entries.append(q_entry)
            else:
                q_entry["total"] = None
                q_entry["valid"] = False

    return attempted_items, valid_entries


def build_summary_row(
    model_name: str,
    llm: str,
    session_scores: List[Dict],
    active_subdims: List[str],
) -> Tuple[Dict, int, int]:
    attempted_items, valid_entries = normalize_session_scores(session_scores, active_subdims)

    row: Dict[str, Any] = {
        "model": model_name,
        "llm": llm,
        "subset": "opp",
        "num_valid_qa": len(valid_entries),
        "num_failed_qa": attempted_items - len(valid_entries),
    }

    for name in ALL_SUBDIMS:
        max_s = SUBDIM_META[name]["max"]
        scores = [int(q[name]) for q in valid_entries if _is_filled(q.get(name))]
        if scores:
            row[f"avg_{name}"] = round(float(np.mean(scores)), 4)
            for s in range(max_s + 1):
                row[f"{name}_score_{s}"] = sum(1 for v in scores if v == s)
        else:
            row[f"avg_{name}"] = ""
            for s in range(max_s + 1):
                row[f"{name}_score_{s}"] = ""

    totals = [q["total"] for q in valid_entries if q.get("total") is not None]
    if totals:
        row["avg_total"] = round(float(np.mean(totals)), 4)
    else:
        row["avg_total"] = ""

    return row, attempted_items, len(valid_entries)


def build_prompt_for_subdim(
    subdim: str,
    question: str,
    generated: str,
    gt: str,
    reason: str,
    ref_conv_block: str,
    subdim_spec: Dict[str, Dict[str, Any]],
) -> str:
    s = subdim_spec[subdim]
    group = s["group"]
    template = s["template"]
    if group == 1:
        return template.format(query=question, gt_answer=gt, generated_answer=generated)
    if group == 2:
        return template.format(
            reason=reason,
            reference_conv_block=ref_conv_block,
            gt_answer=gt,
            query=question,
            generated_answer=generated,
        )
    raise ValueError(f"Unknown group: {group}")


# ---------------- OpenAI Batch (hybrid) ----------------

def is_reasoning_model(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    m = model_name.lower()
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def adjust_for_reasoning_model(body: Dict[str, Any]) -> Dict[str, Any]:
    if not is_reasoning_model(body.get("model", "")):
        return body
    if "max_tokens" in body:
        mt = body.pop("max_tokens")
        body["max_completion_tokens"] = max(int(mt or 0), _REASONING_MAX_TOKENS_FLOOR)
    elif "max_completion_tokens" in body:
        body["max_completion_tokens"] = max(
            int(body["max_completion_tokens"] or 0), _REASONING_MAX_TOKENS_FLOOR
        )
    body.pop("temperature", None)
    body.pop("top_p", None)
    return body


def build_batch_response_format(subdim_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": subdim_name,
            "strict": True,
            "schema": schema,
        },
    }


def build_batch_input_file(
    requests: List[Dict[str, Any]],
    judge_model: str,
    path: Path,
    subdim_spec: Dict[str, Dict[str, Any]],
    temperature: float = _RETRY_TEMP_START,
) -> None:
    is_reasoning = is_reasoning_model(judge_model)
    if is_reasoning:
        logging.info(
            f"  reasoning model detected ({judge_model}): "
            f"using max_completion_tokens, dropping temperature"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            spec = subdim_spec[req["subdim"]]
            body: Dict[str, Any] = {
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": spec["system"]},
                    {"role": "user", "content": req["prompt"]},
                ],
                "max_tokens": spec["max_tokens"],
                "temperature": temperature,
                "response_format": build_batch_response_format(req["subdim"], spec["schema"]),
            }
            adjust_for_reasoning_model(body)
            line = {
                "custom_id": req["custom_id"],
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


def download_batch_records(client, output_file_id: str) -> List[Dict[str, Any]]:
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
        except json.JSONDecodeError as exc:
            logging.warning(f"  bad batch output line skipped: {exc}")
    return records


def extract_batch_result(
    record: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, int]], Optional[str]]:
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


def save_batch_state(state_path: Path, jobs: List[Dict[str, Any]], judge_model: str) -> None:
    payload = {
        "judge_model": judge_model,
        "batches": [
            {
                "model_name": j["model_name"],
                "llm": j["llm"],
                "batch_id": j["batch_id"],
                "input_file_id": j.get("input_file_id"),
                "n_requests": len(j["requests"]),
            }
            for j in jobs if j.get("batch_id")
        ],
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_batch_state(state_path: Path) -> Dict[str, Dict[str, Any]]:
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logging.warning(f"Could not load {state_path.name}: {exc}")
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in raw.get("batches", []):
        out[entry["model_name"]] = entry
    return out


def prepare_model_judge_data(
    data: List[Dict],
    score_file: Path,
    session_map: Dict[int, Dict],
    emb_index: "ConvEmbeddingIndex",
    active_subdims: List[str],
    subdim_spec: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Build session_scores (with resume) + per-subdim flat_items needing judging.

    Shared by both the sync judge loop and the openai-batch path.
    """
    existing_scores: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if score_file.exists():
        try:
            with open(score_file, "r", encoding="utf-8") as f:
                prior = json.load(f)
            for s_entry in prior:
                sid_prev = s_entry.get("session_id")
                for q_entry in s_entry.get("qa_scores", []):
                    key = (sid_prev, q_entry.get("question", ""))
                    existing_scores[key] = {
                        name: q_entry.get(name) for name in ALL_SUBDIMS
                    }
            logging.info(
                f"  Resuming from {score_file.name} ({len(existing_scores)} prior QA entries)"
            )
        except Exception as exc:
            logging.warning(f"  Could not load existing scores: {exc}")
            existing_scores = {}

    flat_items: Dict[str, List[Dict]] = {name: [] for name in active_subdims}
    session_scores: List[Dict] = []

    for session in data:
        sid = session["session_id"]
        ds_session = session_map.get(sid)
        session_idx = len(session_scores)
        qa_score_list: List[Dict] = []

        for qa in session.get("qa_results", []):
            question  = qa.get("question", "")
            generated = qa.get("generated_answer", "")
            gt        = qa.get("ground_truth_answer", "")
            base = {
                "question":            question,
                "generated_answer":    generated,
                "ground_truth_answer": gt,
            }

            prior = existing_scores.get((sid, question), {})

            if not is_valid_qa(generated, gt) or ds_session is None:
                qa_entry = {
                    **base,
                    "judge_eligible": False,
                    **{name: None for name in ALL_SUBDIMS},
                    "total": None,
                    "valid": False,
                }
                qa_score_list.append(qa_entry)
                continue

            q_entry: Dict = {
                **base,
                "judge_eligible": True,
                **{name: prior.get(name) for name in ALL_SUBDIMS},
                "total": None,
                "valid": False,
            }
            qa_score_list.append(q_entry)
            qa_idx = len(qa_score_list) - 1

            ds_qa = ds_session["qa_by_question"].get(question, {})
            raw_ids = ds_qa.get("retrieved_conv_ids", [])
            ref_conv_ids = [int(x) for x in raw_ids] if raw_ids else []
            reason = ds_qa.get("opposed_implicit_reasoning", "")

            ref_block = None
            if any(SUBDIM_META[n_]["group"] == 2 for n_ in active_subdims):
                dim_ref_ids = list(ref_conv_ids)
                if not dim_ref_ids:
                    dim_ref_ids = emb_index.top_k(question, sid, ds_session["convs_by_id"])
                ref_block = format_reference_conv(dim_ref_ids, ds_session["convs_by_id"])

            for name in active_subdims:
                if _is_filled(q_entry.get(name)):
                    continue
                prompt_text = build_prompt_for_subdim(
                    subdim=name,
                    question=question,
                    generated=generated,
                    gt=gt,
                    reason=reason,
                    ref_conv_block=ref_block or "",
                    subdim_spec=subdim_spec,
                )
                item: Dict = {
                    "session_idx": session_idx,
                    "qa_idx":      qa_idx,
                    "prompt":      prompt_text,
                }
                flat_items[name].append(item)

        session_scores.append({
            "session_id": sid,
            "evaluated_subdims": [],
            "qa_scores": qa_score_list,
        })

    return session_scores, flat_items


def finalize_model_output(
    job: Dict[str, Any],
    summary_csv: Path,
    summary_cols: List[str],
    active_subdims: List[str],
) -> None:
    """Fill remaining None scores with -1, write score_file, upsert summary CSV."""
    for s_entry in job["session_scores"]:
        for q_entry in s_entry.get("qa_scores", []):
            if not q_entry.get("judge_eligible"):
                continue
            for name in active_subdims:
                if q_entry.get(name) is None:
                    q_entry[name] = -1

    summary_row, attempted_items, num_valid = build_summary_row(
        model_name=job["model_name"],
        llm=job["llm"],
        session_scores=job["session_scores"],
        active_subdims=active_subdims,
    )
    usage = job["usage"]
    summary_row["judge_calls"]             = usage["calls"]
    summary_row["judge_prompt_tokens"]     = usage["prompt_tokens"]
    summary_row["judge_completion_tokens"] = usage["completion_tokens"]

    score_file = job["score_file"]
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(job["session_scores"], f, indent=2, ensure_ascii=False)
    logging.info(f"  Saved → {score_file.name}")

    upsert_to_csv(summary_csv, summary_row, summary_cols, key_cols=["model", "llm"])

    avg_parts = "  ".join(
        f"{name}={summary_row.get(f'avg_{name}', 'n/a')}"
        for name in active_subdims
    )
    logging.info(
        f"  attempted={attempted_items}  valid={num_valid}  "
        f"failed={summary_row['num_failed_qa']}  "
        f"avg_total={summary_row['avg_total']}  | {avg_parts}  "
        f"judge_calls={usage['calls']}  "
        f"prompt_tokens={usage['prompt_tokens']}  "
        f"completion_tokens={usage['completion_tokens']}"
    )


def process_completed_batch(
    client,
    job: Dict[str, Any],
    output_file_id: str,
    judge_model: str,
    get_sync_client,
    subdim_spec: Dict[str, Dict[str, Any]],
    active_subdims: List[str],
    summary_csv: Path,
    summary_cols: List[str],
) -> None:
    """Download a completed batch's output, parse, sync-retry failures, save."""
    label = job["model_name"]
    logging.info(f"  [{label}] downloading results...")
    records = download_batch_records(client, output_file_id)
    logging.info(f"  [{label}] parsed {len(records)} records")

    results_by_cid: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("custom_id")
        if not cid:
            continue
        text, usage_dict, err = extract_batch_result(rec)
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
        spec = subdim_spec[req["subdim"]]
        parse_fn = make_parser(req["subdim"], spec["max"])
        score = parse_fn(res["text"])
        if score is None:
            sync_retry_queue.append(req)
        else:
            job["session_scores"][req["session_idx"]]["qa_scores"][req["qa_idx"]][req["subdim"]] = score

    n_filled = len(job["requests"]) - len(sync_retry_queue)
    logging.info(
        f"  [{label}] filled {n_filled}/{len(job['requests'])} from batch; "
        f"{len(sync_retry_queue)} need sync retry"
    )

    if sync_retry_queue:
        sync_client = get_sync_client()
        for req in tqdm(sync_retry_queue, desc=f"sync-retry {label}"):
            spec = subdim_spec[req["subdim"]]
            parse_fn = make_parser(req["subdim"], spec["max"])
            score = _retry_with_temp(
                prompt=req["prompt"],
                system_prompt=spec["system"],
                schema=spec["schema"],
                parse_fn=parse_fn,
                judge_client=sync_client,
                max_tokens=spec["max_tokens"],
                usage=job["usage"],
            )
            if score is not None:
                job["session_scores"][req["session_idx"]]["qa_scores"][req["qa_idx"]][req["subdim"]] = score

    finalize_model_output(job, summary_csv, summary_cols, active_subdims)


def run_openai_batch_mode(
    args,
    candidates: List[Tuple],
    output_dir: Path,
    summary_csv: Path,
    summary_cols: List[str],
    active_subdims: List[str],
    subdim_spec: Dict[str, Dict[str, Any]],
    session_map: Dict[int, Dict],
    emb_index: "ConvEmbeddingIndex",
    api_key: str,
) -> None:
    """One OpenAI batch per model. All submitted in parallel, polled to completion;
    parse failures fall back to sync OpenAI calls.

    Per-batch sizing assumption: each model's (N_qa × N_subdims) stays under
    OpenAI's per-batch limits (50K requests / 200MB).
    """
    try:
        import openai
    except ImportError as exc:
        raise ImportError("openai package required for --engine openai-batch") from exc

    client = openai.OpenAI(api_key=api_key)

    state_path = output_dir / "batch_state.json"
    existing_by_key = load_batch_state(state_path)
    if existing_by_key:
        logging.info(
            f"Loaded {len(existing_by_key)} prior batch entries from {state_path.name}"
        )

    # -------- Phase 1: build per-model jobs --------
    logging.info("== Phase 1: building per-model batches ==")
    jobs: List[Dict[str, Any]] = []
    for model_dir, model_name, result_file, data, llm in candidates:
        logging.info(f"── {model_dir.name}  model={model_name}  llm={llm}")
        score_file = output_dir / f"judge_scores_{model_name}_{llm}.json"
        session_scores, flat_items = prepare_model_judge_data(
            data, score_file, session_map, emb_index, active_subdims, subdim_spec,
        )

        requests: List[Dict[str, Any]] = []
        for subdim, items in flat_items.items():
            for item in items:
                requests.append({
                    "custom_id":   f"req-{len(requests):08d}",
                    "prompt":      item["prompt"],
                    "subdim":      subdim,
                    "session_idx": item["session_idx"],
                    "qa_idx":      item["qa_idx"],
                })

        usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        job = {
            "model_dir":      model_dir,
            "model_name":     model_name,
            "llm":            llm,
            "score_file":     score_file,
            "session_scores": session_scores,
            "requests":       requests,
            "input_path":     output_dir / f"batch_input_{model_name}.jsonl",
            "usage":          usage,
            "batch_id":       None,
            "input_file_id":  None,
        }

        if not requests:
            finalize_model_output(job, summary_csv, summary_cols, active_subdims)
            logging.info("  nothing new to judge — wrote outputs only")
            continue

        jobs.append(job)
        logging.info(f"  queued {len(requests)} prompts")

    if not jobs:
        logging.info("Nothing to submit — done")
        return

    # -------- Phase 2: submit (or resume) per-model batches --------
    logging.info(f"== Phase 2: submitting/resuming {len(jobs)} batches ==")
    for job in jobs:
        label = job["model_name"]
        prior = existing_by_key.get(label)
        if prior and prior.get("batch_id"):
            try:
                batch = client.batches.retrieve(prior["batch_id"])
                if batch.status in ("expired", "cancelled", "failed"):
                    logging.warning(
                        f"  [{label}] prior batch {prior['batch_id']} is {batch.status} — resubmitting"
                    )
                else:
                    job["batch_id"]      = prior["batch_id"]
                    job["input_file_id"] = prior.get("input_file_id")
                    logging.info(
                        f"  [{label}] resumed batch {batch.id} (status={batch.status})"
                    )
                    continue
            except Exception as exc:
                logging.warning(f"  [{label}] could not retrieve prior batch: {exc}")

        build_batch_input_file(
            job["requests"], args.judge_model, job["input_path"], subdim_spec,
        )
        batch = submit_batch(client, job["input_path"], args.completion_window)
        job["batch_id"]      = batch.id
        job["input_file_id"] = batch.input_file_id
        logging.info(
            f"  [{label}] submitted batch {batch.id} ({len(job['requests'])} requests)"
        )

    save_batch_state(state_path, jobs, args.judge_model)

    # -------- Phase 3: poll all batches; process each on completion --------
    logging.info(
        f"== Phase 3: polling {len(jobs)} batches every {args.poll_interval}s =="
    )

    sync_client_holder: Dict[str, Any] = {"client": None}
    def get_sync_client():
        if sync_client_holder["client"] is None:
            sync_client_holder["client"] = UnifiedLLMClient(
                engine="openai",
                model_name=args.judge_model,
                api_key=api_key,
                base_url=args.base_url,
                reasoning_mode=is_reasoning_model(args.judge_model),
            )
        return sync_client_holder["client"]

    pending: Dict[str, Dict[str, Any]] = {j["model_name"]: j for j in jobs}
    failed_labels: List[str] = []

    while pending:
        finished_keys: List[str] = []
        for key, job in list(pending.items()):
            try:
                batch = client.batches.retrieve(job["batch_id"])
            except Exception as exc:
                logging.warning(f"  [{key}] retrieve failed: {exc}")
                continue
            counts    = getattr(batch, "request_counts", None)
            completed = getattr(counts, "completed", "?") if counts else "?"
            total     = getattr(counts, "total",     "?") if counts else "?"
            failed_n  = getattr(counts, "failed",    "?") if counts else "?"
            logging.info(
                f"  [{key}] batch={job['batch_id']} status={batch.status} "
                f"progress={completed}/{total} failed={failed_n}"
            )
            if batch.status in _BATCH_TERMINAL_STATUSES:
                if batch.status == "completed":
                    try:
                        process_completed_batch(
                            client, job, batch.output_file_id, args.judge_model,
                            get_sync_client, subdim_spec, active_subdims,
                            summary_csv, summary_cols,
                        )
                    except Exception as exc:
                        logging.error(
                            f"  [{key}] processing failed: {exc}", exc_info=True
                        )
                        failed_labels.append(key)
                else:
                    logging.error(
                        f"  [{key}] ended with status={batch.status} — leaving for next run"
                    )
                    failed_labels.append(key)
                finished_keys.append(key)

        for key in finished_keys:
            del pending[key]
        save_batch_state(state_path, jobs, args.judge_model)

        if pending:
            time.sleep(args.poll_interval)

    # -------- Phase 4: cleanup --------
    all_saved = all(j["score_file"].exists() for j in jobs)
    if all_saved and not failed_labels and state_path.exists():
        try:
            state_path.unlink()
        except Exception:
            pass

    if failed_labels:
        logging.warning(
            f"Done with {len(failed_labels)} failed model(s): {failed_labels}. "
            f"Re-run to retry."
        )


def main():
    args = parse_args()

    script_dir  = Path(__file__).parent
    if args.subset != "opposed":
        raise SystemExit("Only --subset opposed is supported (judge prompts are opposed-specific).")
    root_name   = f"{args.llm}_results"
    n           = args.session_num
    judge_model = args.judge_model
    batch_size  = args.batch_size
    eval_groups = sorted(set(args.dims))

    active_subdims: List[str] = []
    for g in eval_groups:
        active_subdims.extend(GROUP_TO_SUBDIMS[g])

    # Dynamically import the prompt module and bind templates/schemas.
    try:
        prompt_mod = importlib.import_module(args.prompt_module)
    except ImportError as e:
        raise ImportError(
            f"Could not import prompt module '{args.prompt_module}'. "
            f"Make sure {args.prompt_module}.py exists under evaluation/."
        ) from e
    subdim_spec = build_subdim_spec(prompt_mod, active_subdims)

    summary_cols = build_summary_cols()

    judge_short = judge_model.split("/")[-1]
    eval_base   = root_name.replace("_results", "")
    input_dir   = script_dir / root_name / f"opp_{n}"
    output_dir  = script_dir / f"{eval_base}_judge_{judge_short}_{args.prompt_module}" / f"opp_{n}"

    # Auto-merge per-session experiment results into input_dir/{module}/results_*.json.
    logging.info(f"[merge] llm={args.llm} subset={args.subset} sessions=0..{n - 1}")
    build_merged_inputs(script_dir.parent, input_dir, args.llm, args.subset, n)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "judge_summary.csv"

    logging.info(f"Input  : {input_dir}")
    logging.info(f"Output : {output_dir}")
    logging.info(f"Judge  : {judge_model}")
    logging.info(f"Prompt : {args.prompt_module} ({prompt_mod.__file__})")
    logging.info(f"Groups : {[GROUP_NAME[g] for g in eval_groups]}")
    logging.info(f"Sub-dims: {active_subdims}")

    session_map = load_dataset(DATASET_PATH)

    model_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        logging.error(f"No model subdirectories found in {input_dir}")
        return

    if args.models:
        requested = set(args.models)
        available = {d.name for d in model_dirs}
        missing = requested - available
        for m in sorted(missing):
            logging.warning(f"  --models: '{m}' not found under {input_dir}, skipping")
        model_dirs = [d for d in model_dirs if d.name in requested]
        if not model_dirs:
            logging.error("No matching model folders after --models filter")
            return

    logging.info(f"Found {len(model_dirs)} model folder(s)")

    candidates = []
    for model_dir in model_dirs:
        model_name = get_model_name(model_dir.name, n)
        result_file = find_result_file(model_dir)
        if result_file is None:
            logging.warning(f"  Skipping {model_dir.name}: no results_*.json found")
            continue
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        llm = get_llm_from_data(data)
        candidates.append((model_dir, model_name, result_file, data, llm))

    if not candidates:
        logging.info("No models with result files found.")
        return

    emb_index = ConvEmbeddingIndex()

    # Branch: OpenAI Batch hybrid mode — one batch per model, submitted in parallel.
    if args.engine == "openai-batch":
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in env var '{args.api_key_env}'")
        run_openai_batch_mode(
            args=args,
            candidates=candidates,
            output_dir=output_dir,
            summary_csv=summary_csv,
            summary_cols=summary_cols,
            active_subdims=active_subdims,
            subdim_spec=subdim_spec,
            session_map=session_map,
            emb_index=emb_index,
            api_key=api_key,
        )
        logging.info(f"\nDone. Results in {output_dir}/")
        return

    judge_client = None

    def get_judge_client():
        nonlocal judge_client
        if judge_client is None:
            logging.info(f"Loading judge LLM (engine={args.engine})...")
            if args.engine == "vllm":
                judge_client = UnifiedLLMClient(
                    engine="vllm",
                    model_path=judge_model,
                    tensor_parallel_size=args.tensor_parallel,
                    gpu_memory_utilization=args.gpu_memory,
                    max_model_len=args.max_model_len,
                )
            else:  # openai or elice (both use OpenAI-compatible API)
                api_key = os.getenv(args.api_key_env)
                if not api_key:
                    raise ValueError(
                        f"API key not found in env var '{args.api_key_env}'"
                    )
                judge_client = UnifiedLLMClient(
                    engine="openai",
                    model_name=judge_model,
                    api_key=api_key,
                    base_url=args.base_url,
                    reasoning_mode=(args.engine == "elice"),
                )
        return judge_client

    for model_dir, model_name, result_file, data, llm in candidates:
        logging.info(f"\n── {model_dir.name}  model={model_name}  llm={llm}")

        score_file = output_dir / f"judge_scores_{model_name}_{llm}.json"
        usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

        session_scores, flat_items = prepare_model_judge_data(
            data, score_file, session_map, emb_index, active_subdims, subdim_spec,
        )

        total_items = sum(len(v) for v in flat_items.values())
        if total_items == 0:
            logging.info("  No new items to judge — writing outputs only")
        else:
            counts = "  ".join(f"{name}={len(flat_items[name])}" for name in active_subdims)
            logging.info(f"  Items to evaluate: {counts}")

        for name in active_subdims:
            items = flat_items[name]
            if not items:
                continue
            spec = subdim_spec[name]
            parse_fn = make_parser(name, spec["max"])
            n_items = len(items)
            scores: List[Optional[int]] = [None] * n_items
            failed: List[int] = []

            judge_client = get_judge_client()

            for batch_start in tqdm(
                range(0, n_items, batch_size),
                desc=f"Judging {model_name} {name}",
            ):
                batch_items = items[batch_start:batch_start + batch_size]
                batch_prompts = [it["prompt"] for it in batch_items]

                if args.engine == "vllm":
                    try:
                        raw_texts, batch_usages = judge_client.generate_batch_raw(
                            prompts=batch_prompts,
                            system_prompt=spec["system"],
                            guided_json=spec["schema"],
                            temperature=_RETRY_TEMP_START,
                            max_tokens=spec["max_tokens"],
                            return_usage=True,
                        )
                        p_sum = sum(u["prompt_tokens"] for u in batch_usages)
                        c_sum = sum(u["completion_tokens"] for u in batch_usages)
                        usage["calls"] += len(raw_texts)
                        usage["prompt_tokens"] += p_sum
                        usage["completion_tokens"] += c_sum
                        for j, raw in enumerate(raw_texts):
                            idx = batch_start + j
                            score = parse_fn(raw)
                            if score is None:
                                failed.append(idx)
                            else:
                                scores[idx] = score
                    except Exception as e:
                        logging.warning(
                            f"  {name} batch [{batch_start}:{batch_start + len(batch_items)}] "
                            f"failed: {e} — queuing {len(batch_items)} items for individual retry"
                        )
                        for j in range(len(batch_items)):
                            failed.append(batch_start + j)
                else:  # openai / elice: concurrent single-call via ThreadPoolExecutor
                    def _call_one(prompt_text: str):
                        try:
                            out = judge_client.generate(
                                prompt=prompt_text,
                                system_prompt=spec["system"],
                                guided_json=spec["schema"],
                                temperature=_RETRY_TEMP_START,
                                max_tokens=spec["max_tokens"],
                                json_retry=1,
                            )
                            return ("ok", out)
                        except Exception as e:
                            return ("err", e)

                    with ThreadPoolExecutor(max_workers=len(batch_prompts)) as ex:
                        results = list(ex.map(_call_one, batch_prompts))

                    for j, (status, payload) in enumerate(results):
                        idx = batch_start + j
                        if status == "err":
                            logging.warning(
                                f"  {name} item {idx} failed: {payload} — queuing for retry"
                            )
                            failed.append(idx)
                            continue
                        usage["calls"] += 1
                        score = parse_fn(payload)
                        if score is None:
                            failed.append(idx)
                        else:
                            scores[idx] = score

            if failed:
                logging.warning(
                    f"  {name}: {len(failed)} items for individual retry"
                )
                for i in tqdm(failed, desc=f"Retrying {model_name} {name}"):
                    score = _retry_with_temp(
                        prompt=items[i]["prompt"],
                        system_prompt=spec["system"],
                        schema=spec["schema"],
                        parse_fn=parse_fn,
                        judge_client=judge_client,
                        max_tokens=spec["max_tokens"],
                        usage=usage,
                    )
                    if score is not None:
                        scores[i] = score

            for i, item in enumerate(items):
                final = scores[i] if scores[i] is not None else -1
                session_scores[item["session_idx"]]["qa_scores"][item["qa_idx"]][name] = final

        summary_row, attempted_items, num_valid = build_summary_row(
            model_name=model_name,
            llm=llm,
            session_scores=session_scores,
            active_subdims=active_subdims,
        )
        summary_row["judge_calls"] = usage["calls"]
        summary_row["judge_prompt_tokens"] = usage["prompt_tokens"]
        summary_row["judge_completion_tokens"] = usage["completion_tokens"]

        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(session_scores, f, indent=2, ensure_ascii=False)
        logging.info(f"  Saved → {score_file.name}")

        upsert_to_csv(summary_csv, summary_row, summary_cols, key_cols=["model", "llm"])

        avg_parts = "  ".join(
            f"{name}={summary_row.get(f'avg_{name}', 'n/a')}"
            for name in active_subdims
        )
        logging.info(
            f"  attempted={attempted_items}  valid={num_valid}  "
            f"failed={summary_row['num_failed_qa']}  "
            f"avg_total={summary_row['avg_total']}  | {avg_parts}  "
            f"judge_calls={usage['calls']}  "
            f"prompt_tokens={usage['prompt_tokens']}  "
            f"completion_tokens={usage['completion_tokens']}"
        )

    logging.info(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
