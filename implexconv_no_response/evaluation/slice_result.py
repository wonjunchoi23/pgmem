"""
slice_result.py — Slice all model results inside a subset folder to a session range [n, m].

Usage:
    # Auto-derive output path (recommended)
    python slice_result.py --input qwen3_1.7b_results/opp_100 --n 0 --m 49
    python slice_result.py --input gemma3_4b_results/opp_500 --n 0 --m 299


    # Explicit output path
    python slice_result.py --input qwen3_1.7b_results/sup_500 --n 0 --m 199 \
                           --output qwen3_1.7b_results/sup_200

Auto-derive rule (when --output is omitted):
    new_num = m - n + 1
    Replace _{old_num} suffix in the input folder name with _{new_num}.
    e.g.  sup_500  (--n 0 --m 199)  →  sup_200
    Each model subfolder is also renamed accordingly:  lb_500 → lb_200

Copies per model subfolder:
  1. results_*.json     → filtered to sessions n..m, filename updated to session_{n}_{m}.json
  2. retrieval_logs/    → only session_{n..m}_retrieval_log.jsonl files
  3. memory_snapshots/  → only session_{n..m}/ subfolders (copied as-is)
  4. prompt_log/        → only session_{n..m}/ subfolders (copied as-is)
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice all model results in a subset folder to session range [n, m]."
    )
    parser.add_argument("--input", required=True,
                        help="Input subset folder (e.g. qwen3_1.7b_results/sup_500)")
    parser.add_argument("--n", type=int, required=True,
                        help="Start session index (inclusive)")
    parser.add_argument("--m", type=int, required=True,
                        help="End session index (inclusive)")
    parser.add_argument("--output", default=None,
                        help="Output subset folder. Auto-derived from --input if omitted.")
    return parser.parse_args()


def resolve_path(p: str) -> Path:
    """Resolve path relative to this script's directory if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path


def infer_old_num(input_path: Path) -> int:
    """Infer session_num from a folder name ending in _{num}.

    e.g. sup_500 → 500,  lb_200 → 200
    Raises ValueError if no trailing integer is found.
    """
    m = re.search(r"_(\d+)$", input_path.name)
    if not m:
        raise ValueError(
            f"Cannot infer session_num from folder name '{input_path.name}'. "
            "Please specify --output explicitly."
        )
    return int(m.group(1))


def rename_num_suffix(name: str, old_num: int, new_num: int) -> str:
    """Replace _{old_num} suffix at the end of name with _{new_num}."""
    suffix_old = f"_{old_num}"
    suffix_new = f"_{new_num}"
    if name.endswith(suffix_old):
        return name[: -len(suffix_old)] + suffix_new
    return name


# ---------------------------------------------------------------------------
# Slice functions (operate on a single model-level directory)
# ---------------------------------------------------------------------------

def slice_results_json(input_dir: Path, output_dir: Path, n: int, m: int):
    json_files = list(input_dir.glob("results_*.json"))
    if not json_files:
        print("  [WARN] No results_*.json found, skipping.")
        return

    for src in json_files:
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  [WARN] {src.name} is not a list, skipping.")
            continue

        sliced = [item for item in data if n <= item.get("session_id", -1) <= m]

        # Rename: replace session_{old_start}_{old_end} → session_{n}_{m}
        new_name = re.sub(r"session_\d+_\d+", f"session_{n}_{m}", src.name)
        dst = output_dir / new_name

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(sliced, f, ensure_ascii=False, indent=2)

        print(f"  [JSON] {src.name} → {dst.name}  ({len(sliced)} sessions)")


def slice_retrieval_logs(input_dir: Path, output_dir: Path, n: int, m: int):
    logs_src = input_dir / "retrieval_logs"
    if not logs_src.exists():
        print("  [WARN] retrieval_logs/ not found, skipping.")
        return

    logs_dst = output_dir / "retrieval_logs"
    logs_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for i in range(n, m + 1):
        fname = f"session_{i}_retrieval_log.jsonl"
        src = logs_src / fname
        if src.exists():
            shutil.copy2(src, logs_dst / fname)
            copied += 1
        else:
            print(f"  [WARN] retrieval_logs/{fname} not found, skipping.")

    print(f"  [LOGS] Copied {copied} retrieval log file(s).")


def slice_session_subdirs(input_dir: Path, output_dir: Path, n: int, m: int, subdir_name: str):
    """Copy session_{n..m}/ subfolders from input_dir/<subdir_name> as-is."""
    src_root = input_dir / subdir_name
    if not src_root.exists():
        print(f"  [WARN] {subdir_name}/ not found, skipping.")
        return

    dst_root = output_dir / subdir_name
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for i in range(n, m + 1):
        src = src_root / f"session_{i}"
        if src.is_dir():
            shutil.copytree(src, dst_root / f"session_{i}", dirs_exist_ok=True)
            copied += 1
        else:
            print(f"  [WARN] {subdir_name}/session_{i}/ not found, skipping.")

    print(f"  [{subdir_name.upper()}] Copied {copied} session folder(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_dir = resolve_path(args.input)
    n, m = args.n, args.m
    new_num = m - n + 1

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Resolve output subset dir
    if args.output:
        output_dir = resolve_path(args.output)
    else:
        old_num = infer_old_num(input_dir)
        new_name = rename_num_suffix(input_dir.name, old_num, new_num)
        output_dir = input_dir.parent / new_name
        print(f"[AUTO] Output path derived: {output_dir}")

    # Discover model subdirectories
    model_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        print(f"No subdirectories found in {input_dir}")
        return

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Range : session {n} ~ {m}  ({new_num} sessions)")
    print(f"Models: {[d.name for d in model_dirs]}")
    print()

    old_num = infer_old_num(input_dir)

    for model_dir in model_dirs:
        new_model_name = rename_num_suffix(model_dir.name, old_num, new_num)
        output_model_dir = output_dir / new_model_name

        if output_model_dir.exists():
            print(f"── {model_dir.name} → {new_model_name}  [SKIP: already exists]")
            continue

        print(f"── {model_dir.name} → {new_model_name}")
        slice_results_json(model_dir, output_model_dir, n, m)
        slice_retrieval_logs(model_dir, output_model_dir, n, m)
        slice_session_subdirs(model_dir, output_model_dir, n, m, "memory_snapshots")
        slice_session_subdirs(model_dir, output_model_dir, n, m, "prompt_log")

    print("\nDone.")


if __name__ == "__main__":
    main()
