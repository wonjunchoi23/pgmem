"""
repair_judge_outputs.py — Rebuild judge score metadata and summary CSV.

Usage:
    python repair_judge_outputs.py evaluation/qwen3_1.7b_judge_Qwen3-8B/opp_200
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation_llm_judge import (
    SUMMARY_COLS,
    build_summary_row,
    parse_score_file_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair judge_scores JSON files and rebuild judge_summary.csv",
    )
    parser.add_argument(
        "output_dir",
        help="Judge output directory (e.g. evaluation/qwen3_1.7b_judge_Qwen3-8B/opp_200)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path(__file__).parent.parent / output_dir).resolve()

    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Judge output directory not found: {output_dir}")

    summary_csv = output_dir / "judge_summary.csv"
    score_files = sorted(output_dir.glob("judge_scores_*.json"))
    if not score_files:
        raise FileNotFoundError(f"No judge_scores_*.json found in {output_dir}")

    rows = []
    for score_file in score_files:
        model_name, llm = parse_score_file_name(score_file, summary_csv=summary_csv)
        with open(score_file, "r", encoding="utf-8") as f:
            session_scores = json.load(f)

        row, active_dims, attempted_items, num_valid = build_summary_row(
            model_name=model_name,
            llm=llm,
            session_scores=session_scores,
        )

        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(session_scores, f, indent=2, ensure_ascii=False)

        rows.append({col: row.get(col, "") for col in SUMMARY_COLS})
        print(
            f"{score_file.name}: dims={active_dims or []} "
            f"valid={num_valid} failed={attempted_items - num_valid}"
        )

    df = pd.DataFrame(rows, columns=SUMMARY_COLS)
    if not df.empty:
        df = df.sort_values(by=["model", "llm"], kind="stable")
    df.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"Rebuilt {summary_csv}")


if __name__ == "__main__":
    main()
