# Usage:
#   python evaluation_gmem_llm_judge.py \
#       --config-num 1 --llm Qwen3-1.7B --session-num 100 \
#       --judge-model meta-llama/Llama-3.1-8B-Instruct \
#       --dims 1 2 --batch-size 32
#
# --dims meaning:
#   1 = response_competence  (rc_question_addressing / rc_specificity)
#   2 = persona_adaptation   (pa_persona_recognition / pa_generic_distinctness / pa_substantive_integration)

import argparse
import fcntl
import importlib
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["HF_TOKEN"] = "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"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_module.llm_client import UnifiedLLMClient, _parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

DATASET_PATH = (
    Path(__file__).parent.parent / "dataset/implexconv/ImplexConv_opposed_processed.json"
)
GMEM_DIR = Path(__file__).parent.parent / "gmem"
FALLBACK_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_TOP_K = 3

_RETRY_TEMP_START = 0.3
_RETRY_TEMP_STEP  = 0.025
_RETRY_MAX        = 30
_CLIENT_JSON_RETRY = 30


# ---------------- sub-dim meta ----------------
SUBDIM_META: Dict[str, Dict[str, Any]] = {
    "rc_question_addressing": {
        "group": 1, "max": 1, "max_tokens": 200,
        "system_attr": "SYSTEM_PROMPT_RC",
        "template_attr": "USER_PROMPT_TEMPLATE_RC1",
        "schema_attr": "JUDGE_JSON_SCHEMA_RC1",
    },
    "rc_specificity": {
        "group": 1, "max": 1, "max_tokens": 200,
        "system_attr": "SYSTEM_PROMPT_RC",
        "template_attr": "USER_PROMPT_TEMPLATE_RC2",
        "schema_attr": "JUDGE_JSON_SCHEMA_RC2",
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
        description="LLM-as-Judge for gmem outputs"
    )
    parser.add_argument("--config-num", type=int, required=True,
                        help="gmem config number (e.g. 1). Matches config_<N>_outputs_<llm>_opposed "
                             "AND config_<N>_*_outputs_<llm>_opposed under gmem/")
    parser.add_argument("--llm", required=True,
                        help="LLM short name as it appears in gmem folder names (e.g. Qwen3-1.7B)")
    parser.add_argument("--session-num", type=int, required=True,
                        help="Session count integer (e.g. 100). Only folders containing "
                             "session_0_<session_num-1>/results_*.json are evaluated.")
    parser.add_argument("--judge-model", required=True,
                        help="Judge model path/name (vllm: HF path, openai: model name)")
    parser.add_argument("--dims", nargs="+", type=int, choices=[1, 2], default=[1, 2],
                        help="Groups: 1=response_competence 2=persona_adaptation")
    parser.add_argument("--prompt-module", default="prompt",
                        help="Python module name (under evaluation/) that exports the prompt "
                             "templates and JSON schemas. Default: prompt.")
    parser.add_argument("--engine", choices=["vllm", "openai", "elice"], default="vllm",
                        help="Inference engine (default: vllm).")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Env var name holding the API key (default: OPENAI_API_KEY)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these stripped model names (e.g. config_1 config_1_q1). "
                             "If omitted, all matching config folders are evaluated.")
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> Dict:
    with open(dataset_path, "r", encoding="utf-8") as f:
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


def strip_model_name(folder_name: str, llm: str) -> str:
    """gmem/config_1_outputs_Qwen3-1.7B_opposed -> config_1
       gmem/config_1_q1_outputs_Qwen3-1.7B_opposed -> config_1_q1"""
    suffix = f"_outputs_{llm}_opposed"
    return folder_name[: -len(suffix)] if folder_name.endswith(suffix) else folder_name


def find_matching_config_dirs(
    gmem_dir: Path, config_num: int, llm: str, session_num: int
) -> List[Path]:
    """Match config_<N>_outputs_<llm>_opposed AND config_<N>_*_outputs_<llm>_opposed,
    then keep only those containing session_0_<session_num-1>/results_*.json."""
    session_folder = f"session_0_{session_num - 1}"
    suffix = f"_outputs_{llm}_opposed"
    exact = f"config_{config_num}{suffix}"
    prefix = f"config_{config_num}_"

    # prefix already contains the trailing '_', so "config_10_..." won't match "config_1_"
    matched: List[Path] = []
    for d in sorted(gmem_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.endswith(suffix):
            continue
        if not (name == exact or name.startswith(prefix)):
            continue
        session_dir = d / session_folder
        if not sorted(session_dir.glob("results_*.json")):
            continue
        matched.append(d)
    return matched


def find_result_file(model_dir: Path, session_num: int) -> Optional[Path]:
    session_dir = model_dir / f"session_0_{session_num - 1}"
    files = sorted(session_dir.glob("results_*.json"))
    return files[0] if files else None


def get_llm_from_data(data: List[Dict]) -> str:
    for session in data:
        for qa in session.get("qa_results", []):
            model_field = qa.get("qa_tokens", {}).get("model")
            if model_field:
                return model_field.split("/")[-1]
    return "unknown"


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


def main():
    args = parse_args()

    script_dir  = Path(__file__).parent
    n           = args.session_num
    judge_model = args.judge_model
    batch_size  = args.batch_size
    eval_groups = sorted(set(args.dims))

    active_subdims: List[str] = []
    for g in eval_groups:
        active_subdims.extend(GROUP_TO_SUBDIMS[g])

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
    output_dir  = script_dir / f"gmem_judge_{judge_short}_{args.prompt_module}" / f"opp_{n}"

    if not GMEM_DIR.exists():
        raise FileNotFoundError(f"gmem directory not found: {GMEM_DIR}")

    model_dirs = find_matching_config_dirs(GMEM_DIR, args.config_num, args.llm, n)
    if not model_dirs:
        logging.error(
            f"No matching config folders found in {GMEM_DIR} "
            f"for config_num={args.config_num}, llm={args.llm}, "
            f"session_folder=session_0_{n - 1}"
        )
        return

    if args.models:
        requested = set(args.models)
        kept: List[Path] = []
        for d in model_dirs:
            stripped = strip_model_name(d.name, args.llm)
            if stripped in requested:
                kept.append(d)
        missing = requested - {strip_model_name(d.name, args.llm) for d in model_dirs}
        for m in sorted(missing):
            logging.warning(f"  --models: '{m}' not found among matched folders, skipping")
        model_dirs = kept
        if not model_dirs:
            logging.error("No matching model folders after --models filter")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "judge_summary.csv"

    logging.info(f"gmem    : {GMEM_DIR}")
    logging.info(f"Output  : {output_dir}")
    logging.info(f"Judge   : {judge_model}")
    logging.info(f"Prompt  : {args.prompt_module} ({prompt_mod.__file__})")
    logging.info(f"Groups  : {[GROUP_NAME[g] for g in eval_groups]}")
    logging.info(f"Sub-dims: {active_subdims}")
    logging.info(f"Matched {len(model_dirs)} config folder(s):")
    for d in model_dirs:
        logging.info(f"  - {d.name}")

    session_map = load_dataset(DATASET_PATH)

    candidates = []
    for model_dir in model_dirs:
        model_name = strip_model_name(model_dir.name, args.llm)
        result_file = find_result_file(model_dir, n)
        if result_file is None:
            logging.warning(f"  Skipping {model_dir.name}: no results_*.json in session_0_{n - 1}")
            continue
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        llm = get_llm_from_data(data)
        candidates.append((model_dir, model_name, result_file, data, llm))

    if not candidates:
        logging.info("No models with result files found.")
        return

    emb_index = ConvEmbeddingIndex()
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
            else:
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
                logging.info(f"  Resuming from {score_file.name} ({len(existing_scores)} prior QA entries)")
            except Exception as e:
                logging.warning(f"  Could not load existing scores: {e}")
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
                else:
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
