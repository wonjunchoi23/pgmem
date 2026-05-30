"""
Example:
    python evaluation_basic.py --llm Qwen3-1.7B --module pgmem --benchmark 32k
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

VALID_OPTIONS = {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-session results for one module and evaluate (exact match)."
    )
    parser.add_argument("--llm", required=True,
                        help="LLM tag, e.g. Qwen3-1.7B")
    parser.add_argument("--module", required=True,
                        help="Memory-module folder under personamem/, e.g. pgmem")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark size tag, e.g. 32k")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def is_valid_text(val) -> bool:
    return val is not None and str(val).strip() not in ("", "N/A")


# ---------------------------------------------------------------------------
# Merge: gather per-session result files and merge their qa_results
# ---------------------------------------------------------------------------

def find_output_dirs(module_dir: Path, llm: str, benchmark: str) -> List[Path]:
    """Experiment output folders ending with `{llm}_{benchmark}` (e.g. config_0_outputs_Qwen3-1.7B_32k)."""
    tag = f"{llm}_{benchmark}"
    matches = [
        d for d in module_dir.iterdir()
        if d.is_dir() and (d.name == tag or d.name.endswith(f"_{tag}"))
    ]
    if not matches:
        raise FileNotFoundError(
            f"No experiment output folder ending with '{tag}' found in {module_dir}"
        )
    return sorted(matches)


def merge_qa_results(output_dirs: List[Path], llm: str, benchmark: str) -> List[Dict]:
    """Collect qa_results from every `*/results_{llm}_{benchmark}_session_*.json` file."""
    pattern = f"*/results_{llm}_{benchmark}_session_*.json"
    merged: List[Dict] = []
    n_files = 0
    for out_dir in output_dirs:
        for result_file in sorted(out_dir.glob(pattern)):
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for session in data:
                merged.extend(session.get("qa_results", []))
            n_files += 1
    print(f"   Merged {n_files} session file(s) → {len(merged)} qa_results")
    return merged


def find_result_file(module_dir: Path) -> Optional[Path]:
    files = sorted(module_dir.glob("results_*.json"))
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
# Column helpers
# ---------------------------------------------------------------------------

def collect_all_question_types(module_dirs: List[Path]) -> List[str]:
    """First pass: scan all merged result files to collect every question_type."""
    types: set = set()
    for module_dir in module_dirs:
        result_file = find_result_file(module_dir)
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

    script_dir     = Path(__file__).parent          # personamem/evaluation
    personamem_dir = script_dir.parent              # personamem
    module_dir     = personamem_dir / args.module

    if not module_dir.exists():
        raise FileNotFoundError(f"Module folder not found: {module_dir}")

    tag = f"{args.llm}_{args.benchmark}"

    # ── Step 1: merge qa_results for this module ──────────────────────────
    print(f"[merge] module={args.module}  llm={args.llm}  benchmark={args.benchmark}")
    output_dirs = find_output_dirs(module_dir, args.llm, args.benchmark)
    merged_qa = merge_qa_results(output_dirs, args.llm, args.benchmark)
    if not merged_qa:
        raise RuntimeError(f"No qa_results found under {module_dir} for tag '{tag}'.")

    results_root   = script_dir / f"{tag}_results"
    module_out_dir = results_root / args.module
    module_out_dir.mkdir(parents=True, exist_ok=True)
    merged_path    = module_out_dir / f"results_{tag}_merged.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump([{"qa_results": merged_qa}], f, ensure_ascii=False, indent=2)
    print(f"[merge] wrote {merged_path}")

    # ── Step 2: evaluate every module folder under {tag}_results ──────────
    eval_dir = script_dir / f"{tag}_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    qa_csv = eval_dir / "qa_score.csv"

    module_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    print(f"[eval] {len(module_dirs)} module folder(s) under {results_root}")

    all_types = collect_all_question_types(module_dirs)
    QA_COLS = build_qa_columns(all_types)
    print(f"[eval] question types ({len(all_types)}): {all_types}")
    print(f"[eval] output → {eval_dir}")

    for mdir in tqdm(module_dirs, desc="Evaluating modules"):
        model_name = mdir.name

        # Always re-evaluate the module we just merged; skip others already done.
        if model_name != args.module and model_name in get_evaluated_models(qa_csv, QA_COLS):
            print(f"\n── {model_name}: already evaluated — skipping.")
            continue

        result_file = find_result_file(mdir)
        if result_file is None:
            print(f"\n── {model_name}: no results_*.json — skipping.")
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_res = evaluate_qa_exact_match(data, all_types)

        row: Dict = {
            "model":            model_name,
            "llm":              args.llm,
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

        print(f"\n── {model_name}: overall_accuracy={qa_res['overall_accuracy']:.4f}  "
              f"valid_qa={qa_res['valid_qa']}  invalid_count={qa_res['invalid_count']}")
        for t in all_types:
            pt = qa_res["per_type"][t]
            print(f"     {t}: accuracy={pt['accuracy']:.4f} "
                  f"({pt['correct']}/{pt['total']}) invalid={pt['invalid']}")

    print(f"\nDone. → {qa_csv}")


if __name__ == "__main__":
    main()
