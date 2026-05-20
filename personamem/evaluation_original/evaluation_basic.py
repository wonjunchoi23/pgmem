"""
evaluation_basic.py — Evaluation for new-format experiment results (exact match)

Input structure:
    evaluation/{root}/{model}/
        ├── results_*.json
        └── retrieval_logs/
            └── session_*_retrieval_log.jsonl

Output structure:
    evaluation/{root_without_results}_eval/
        ├── qa_score.csv
        └── token_memory_stats.csv

Example:
    python evaluation_basic.py --root 128k_qwen3_1.7b_results
    python evaluation_basic.py --root 128k_gemma3_4b_results

Arguments:
    --root : root folder name under evaluation/ (e.g. 32k_qwen3_1.7b_results)
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

VALID_OPTIONS = {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation script for personalized dialogue experiments"
    )
    parser.add_argument("--root", required=True,
                        help="Root folder name under evaluation/ (e.g. 32k_qwen3_1.7b_results)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def is_valid_text(val) -> bool:
    return val is not None and str(val).strip() not in ("", "N/A")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def get_llm_from_data(data: List[Dict]) -> str:
    for session in data:
        model = session.get("config_metadata", {}).get("model")
        if model:
            return model.split("/")[-1]
    for session in data:
        for qa in session.get("qa_results", []):
            model = qa.get("qa_tokens", {}).get("model")
            if model:
                return model.split("/")[-1]
    return "unknown"


def find_result_file(model_dir: Path) -> Optional[Path]:
    files = sorted(model_dir.glob("results_*.json"))
    return files[0] if files else None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def get_evaluated_models(csv_path: Path, required_columns: Optional[List[str]] = None) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if "model" not in df.columns:
            return set()
        if required_columns:
            if any(col not in df.columns for col in required_columns):
                return set()
            complete_mask = (df[required_columns].astype(str).apply(lambda col: col.str.strip() != "")).all(axis=1)
            return set(df.loc[complete_mask, "model"].tolist())
        return set(df["model"].tolist())
    except Exception:
        return set()


def upsert_to_csv(csv_path: Path, row: Dict, columns: List[str]):
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

    if "model" in df.columns and row.get("model") is not None:
        df = df[df["model"].astype(str) != str(row["model"])]

    new_row = pd.DataFrame([{c: row.get(c, "") for c in columns}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Turn counting from retrieval logs
# ---------------------------------------------------------------------------

def count_turns_from_retrieval_logs(retrieval_log_dir: Path) -> Tuple[float, int]:
    if not retrieval_log_dir.exists():
        return 0.0, 0
    log_files = sorted(retrieval_log_dir.glob("session_*_retrieval_log.jsonl"))
    if not log_files:
        return 0.0, 0

    counts = []
    for lf in log_files:
        with open(lf, "r", encoding="utf-8") as f:
            counts.append(sum(1 for line in f if line.strip()))
    if not counts:
        return 0.0, 0
    return float(np.mean(counts)), int(sum(counts))


# ---------------------------------------------------------------------------
# QA evaluation — exact match
# ---------------------------------------------------------------------------

def evaluate_qa_exact_match(data: List[Dict], all_types: List[str]) -> Dict:
    """Case-insensitive exact match against a/b/c/d options."""
    total_qa = valid_qa = overall_correct = invalid_count = 0
    per_type: Dict[str, Dict] = {t: {"total": 0, "correct": 0, "invalid": 0} for t in all_types}

    for session in data:
        for qa in session.get("qa_results", []):
            qt = qa.get("question_type", "")
            gen_raw = qa.get("generated_answer")
            gt_raw = qa.get("ground_truth_answer")

            total_qa += 1

            if not is_valid_text(gen_raw) or not is_valid_text(gt_raw):
                continue

            gen = str(gen_raw).strip().lower()
            gt = str(gt_raw).strip().lower()
            valid_qa += 1

            is_invalid = gen not in VALID_OPTIONS
            is_correct = gen == gt

            if is_invalid:
                invalid_count += 1
            if is_correct:
                overall_correct += 1

            if qt in per_type:
                per_type[qt]["total"] += 1
                if is_invalid:
                    per_type[qt]["invalid"] += 1
                if is_correct:
                    per_type[qt]["correct"] += 1

    return {
        "total_qa": total_qa,
        "valid_qa": valid_qa,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_correct / valid_qa if valid_qa > 0 else 0.0,
        "invalid_count": invalid_count,
        "per_type": {
            t: {
                **v,
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
            }
            for t, v in per_type.items()
        },
    }


# ---------------------------------------------------------------------------
# Token statistics
# ---------------------------------------------------------------------------

def evaluate_token_stats(data: List[Dict]) -> Dict:
    accum: Dict[str, List[float]] = {
        "total_input": [],
        "total_output": [],
        "total_llm_calls": [],
    }

    for session in data:
        tok = session.get("token_statistics", {})
        if not tok:
            continue
        for k in accum:
            if k in tok:
                accum[k].append(float(tok[k]))

    return {k: (float(np.mean(v)) if v else 0.0) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Retrieval / memory statistics
# ---------------------------------------------------------------------------

def evaluate_memory_stats(retrieval_log_dir: Path) -> Optional[Dict]:
    if not retrieval_log_dir.exists():
        return None
    log_files = sorted(retrieval_log_dir.glob("session_*_retrieval_log.jsonl"))
    if not log_files:
        return None

    memory_type: Optional[List[str]] = None
    per_position: List[List[float]] = []
    sum_per_entry: List[float] = []

    for lf in log_files:
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                nr = entry.get("num_retrieved", [])
                mt = entry.get("memory_type", [])
                if isinstance(nr, (int, float)): nr = [nr]
                if isinstance(mt, str):          mt = [mt]

                if memory_type is None and mt:
                    memory_type = mt
                    per_position = [[] for _ in mt]

                sum_per_entry.append(float(sum(nr)))
                for i, count in enumerate(nr):
                    if i < len(per_position):
                        per_position[i].append(float(count))

    if not sum_per_entry:
        return None

    return {
        "memory_type":         memory_type or [],
        "list_retrieved_avg":  [float(np.mean(p)) if p else 0.0 for p in per_position],
        "total_retrieved_avg": float(np.mean(sum_per_entry)),
    }


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def collect_all_question_types(model_dirs: List[Path]) -> List[str]:
    """First pass: scan all result files to collect every question_type."""
    types: set = set()
    for model_dir in model_dirs:
        result_file = find_result_file(model_dir)
        if result_file is None:
            continue
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for session in data:
                for qa in session.get("qa_results", []):
                    qt = qa.get("question_type", "")
                    if qt:
                        types.add(qt)
        except Exception:
            pass
    return sorted(types)


def build_qa_columns(all_types: List[str]) -> List[str]:
    base = ["model", "llm", "total_qa", "valid_qa", "overall_accuracy", "invalid_count"]
    for t in all_types:
        base += [f"{t}_total", f"{t}_correct", f"{t}_accuracy", f"{t}_invalid"]
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    root_name = args.root
    eval_base = root_name.replace("_results", "")

    input_dir = script_dir / root_name
    eval_dir  = script_dir / f"{eval_base}_eval"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    eval_dir.mkdir(parents=True, exist_ok=True)

    qa_csv  = eval_dir / "qa_score.csv"
    tok_csv = eval_dir / "token_memory_stats.csv"

    TOK_COLS = [
        "model", "llm",
        "avg_turn", "total_turn",
        "avg_total_input", "avg_total_output", "avg_llm_calls",
        "memory_type", "list_retrieved_avg", "total_retrieved_avg",
    ]

    model_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        print(f"No subdirectories found in {input_dir}")
        return

    print(f"Found {len(model_dirs)} model folder(s) in {input_dir}")

    print("Scanning question types...")
    all_types = collect_all_question_types(model_dirs)
    print(f"Question types ({len(all_types)}): {all_types}")

    QA_COLS = build_qa_columns(all_types)
    print(f"Output → {eval_dir}")

    for model_dir in tqdm(model_dirs, desc="Evaluating models"):
        model_name = model_dir.name
        print(f"\n── {model_name}")

        already_qa  = model_name in get_evaluated_models(qa_csv, QA_COLS)
        already_tok = model_name in get_evaluated_models(tok_csv, TOK_COLS)

        if already_qa and already_tok:
            print("   Already evaluated — skipping.")
            continue

        result_file = find_result_file(model_dir)
        if result_file is None:
            print("   Skipping: no results_*.json found")
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        llm = get_llm_from_data(data)
        print(f"   llm={llm}  sessions={len(data)}")

        retrieval_log_dir = model_dir / "retrieval_logs"
        avg_turn, total_turn = count_turns_from_retrieval_logs(retrieval_log_dir)

        # ── QA evaluation ────────────────────────────────────────────────
        if not already_qa:
            print("   [QA] Exact match evaluation...")
            qa_res = evaluate_qa_exact_match(data, all_types)

            row: Dict = {
                "model":            model_name,
                "llm":              llm,
                "total_qa":         qa_res["total_qa"],
                "valid_qa":         qa_res["valid_qa"],
                "overall_accuracy": round(qa_res["overall_accuracy"], 6),
                "invalid_count":    qa_res["invalid_count"],
            }
            for t in all_types:
                pt = qa_res["per_type"][t]
                row[f"{t}_total"]    = pt["total"]
                row[f"{t}_correct"]  = pt["correct"]
                row[f"{t}_accuracy"] = round(pt["accuracy"], 6)
                row[f"{t}_invalid"]  = pt["invalid"]

            upsert_to_csv(qa_csv, row, QA_COLS)

            print(f"   → overall_accuracy={qa_res['overall_accuracy']:.4f}  "
                  f"valid_qa={qa_res['valid_qa']}  "
                  f"invalid_count={qa_res['invalid_count']}")
            for t in all_types:
                pt = qa_res["per_type"][t]
                print(f"     {t}: accuracy={pt['accuracy']:.4f} "
                      f"({pt['correct']}/{pt['total']}) invalid={pt['invalid']}")

        # ── Token + memory statistics ─────────────────────────────────────
        if not already_tok:
            print("   [Token stats] Computing...")
            tok = evaluate_token_stats(data)

            print("   [Memory stats] Computing...")
            ms = evaluate_memory_stats(retrieval_log_dir)

            if ms is not None:
                mem_type_str  = json.dumps(ms["memory_type"])
                list_avg_str  = json.dumps([round(v, 4) for v in ms["list_retrieved_avg"]])
                total_ret_avg = round(ms["total_retrieved_avg"], 4)
            else:
                mem_type_str = list_avg_str = ""
                total_ret_avg = ""

            upsert_to_csv(tok_csv, {
                "model":               model_name,
                "llm":                 llm,
                "avg_turn":            round(avg_turn, 2),
                "total_turn":          total_turn,
                "avg_total_input":     round(tok["total_input"], 2),
                "avg_total_output":    round(tok["total_output"], 2),
                "avg_llm_calls":       round(tok["total_llm_calls"], 2),
                "memory_type":         mem_type_str,
                "list_retrieved_avg":  list_avg_str,
                "total_retrieved_avg": total_ret_avg,
            }, TOK_COLS)

            print(f"   → avg_total_input={tok['total_input']:.1f}  "
                  f"avg_llm_calls={tok['total_llm_calls']:.1f}  "
                  f"avg_turn={avg_turn:.2f} total_turn={total_turn}  "
                  f"total_retrieved_avg={total_ret_avg}")

    print(f"\nDone. Results written to {eval_dir}/")


if __name__ == "__main__":
    main()
