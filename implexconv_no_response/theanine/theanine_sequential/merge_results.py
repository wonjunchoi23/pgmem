"""
Merge Results Script

Takes an output folder (e.g. config_0_outputs_Qwen3-1.7B_supportive) as argument,
finds all session_*/ subdirectories, merges their results JSON files, and creates
a new session_{min}_{max}/ folder containing:
  - results_{model}_{subset}_session_{min}_{max}.json
  - retrieval_logs/ (copied from all session dirs)
  - memory_snapshots/ (copied from all session dirs)

Usage:
    python merge_results.py config_0_outputs_Qwen3-1.7B_supportive
    python merge_results.py /absolute/path/to/config_0_outputs_Qwen3-1.7B_supportive
    python merge_results.py config_0_outputs_Qwen3-1.7B_supportive --dry-run
"""
import json
import sys
import argparse
import shutil
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict


# =============================================================================
# HELPERS
# =============================================================================

def _parse_session_range(dirname: str) -> Tuple[int, int]:
    """Parse 'session_<start>_<end>' → (start, end)."""
    parts = dirname.split("_")
    if len(parts) != 3 or parts[0] != "session":
        raise ValueError(f"Unexpected directory name: {dirname!r}")
    return int(parts[1]), int(parts[2])


def _find_session_dirs(output_dir: Path) -> List[Tuple[Path, int, int]]:
    """Return sorted list of (path, start, end) for all session_*/ dirs."""
    result = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("session_"):
            try:
                start, end = _parse_session_range(item.name)
                result.append((item, start, end))
            except ValueError as e:
                print(f"  [skip] {item.name}: {e}")
    result.sort(key=lambda x: x[1])
    return result


def _find_results_files(session_dir: Path) -> List[Path]:
    """
    Find all result files directly under session_dir whose names start with 'results_'.
    """
    return sorted(
        [
            p for p in session_dir.iterdir()
            if p.is_file() and p.name.startswith("results_") and p.suffix == ".json"
        ],
        key=lambda p: p.name
    )


def _result_group_key(filename: str) -> str:
    """
    Normalize:
      results_xxx_session_0_99.json
    -> results_xxx.json

    This lets files from different session ranges be grouped together.
    """
    return re.sub(r"_session_\d+_\d+(?=\.json$)", "", filename)


def _merged_result_filename(group_key: str, global_min: int, global_max: int) -> str:
    """
    Convert:
      results_xxx.json
    -> results_xxx_session_{global_min}_{global_max}.json
    """
    stem = group_key[:-5] if group_key.endswith(".json") else group_key
    return f"{stem}_session_{global_min}_{global_max}.json"


def _load_result_file(path: Path) -> List[Dict]:
    """
    Load one results_*.json file. Expected: list of dicts.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  [warn] Unexpected format in {path}")
            return []
        return data
    except Exception as e:
        print(f"  [warn] Could not load {path}: {e}")
        return []


def _validate(all_results: List[Dict]) -> List[str]:
    """Check for duplicate or missing session_ids. Returns list of warning strings."""
    warnings = []

    ids = [r["session_id"] for r in all_results if isinstance(r, dict) and "session_id" in r]
    if not ids:
        return warnings

    seen, dupes = set(), set()
    for sid in ids:
        if sid in seen:
            dupes.add(sid)
        seen.add(sid)
    if dupes:
        warnings.append(f"Duplicate session_ids: {sorted(dupes)}")

    sorted_ids = sorted(ids)
    gaps = []
    for i in range(len(sorted_ids) - 1):
        if sorted_ids[i + 1] != sorted_ids[i] + 1:
            gaps.append((sorted_ids[i], sorted_ids[i + 1]))
    if gaps:
        for a, b in gaps:
            warnings.append(
                f"Gap between session_id {a} and {b} (missing: {list(range(a+1, b))})"
            )

    return warnings


def _merge_directory_contents(src_dir: Path, dst_dir: Path) -> int:
    """
    Recursively merge src_dir into dst_dir.
    - folders with same name are merged
    - files with same path are overwritten
    Returns number of copied files
    """
    copied = 0
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.rglob("*"):
        rel = item.relative_to(src_dir)
        target = dst_dir / rel

        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied += 1

    return copied


def _merge_named_subdirs(session_dirs: List[Tuple[Path, int, int]], dest_dir: Path) -> Dict[str, int]:
    """
    Merge only 'retrieval_logs' subdirectory under each session dir into dest_dir/retrieval_logs/.
    """
    merged_counts: Dict[str, int] = defaultdict(int)

    for session_dir, start, end in session_dirs:
        for item in session_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name != "retrieval_logs":
                continue

            dst_subdir = dest_dir / item.name
            copied = _merge_directory_contents(item, dst_subdir)
            merged_counts[item.name] += copied

    return dict(merged_counts)


# =============================================================================
# CORE
# =============================================================================

def merge_results(output_dir: Path, dry_run: bool = False) -> bool:
    print(f"\n{'='*60}")
    print(f"Output dir : {output_dir}")
    print(f"{'='*60}\n")

    if not output_dir.exists():
        print(f"[error] Directory not found: {output_dir}")
        return False

    session_dirs = _find_session_dirs(output_dir)
    if not session_dirs:
        print(f"[error] No session_*/ directories found in {output_dir}")
        return False

    print(f"Found {len(session_dirs)} session directory/ies:")
    for _, s, e in session_dirs:
        print(f"  session_{s}_{e}/")
    print()

    global_min = min(s for _, s, _ in session_dirs)
    global_max = max(e for _, _, e in session_dirs)

    merged_dirname = f"session_{global_min}_{global_max}"
    merged_dir = output_dir / merged_dirname

    # -------------------------------------------------------------------------
    # 1) Merge every results_*.json file, grouped by normalized filename
    # -------------------------------------------------------------------------
    grouped_results: DefaultDict[str, List[Dict]] = defaultdict(list)

    for session_dir, start, end in session_dirs:
        result_files = _find_results_files(session_dir)
        if not result_files:
            print(f"  session_{start}_{end}: no results_*.json files found")
            continue

        print(f"  session_{start}_{end}:")
        for rf in result_files:
            chunk = _load_result_file(rf)
            group_key = _result_group_key(rf.name)
            grouped_results[group_key].extend(chunk)
            print(f"    - {rf.name}: {len(chunk)} rows loaded -> group '{group_key}'")

    if not grouped_results:
        print("\n[error] No results_*.json files loaded.")
        return False

    # sort + validate per results group
    merged_result_outputs: Dict[str, List[Dict]] = {}
    for group_key, rows in grouped_results.items():
        rows.sort(
            key=lambda r: r.get("session_id", -1) if isinstance(r, dict) else -1
        )
        merged_result_outputs[group_key] = rows

        warnings_list = _validate(rows)
        if warnings_list:
            print(f"\nWarnings for {group_key}:")
            for w in warnings_list:
                print(f"  ! {w}")
        else:
            print(f"\n  [ok] {group_key}: no gaps or duplicates detected.")

    print(f"\nOutput folder        : {merged_dir}/")
    print("Merged result files  :")
    for group_key in merged_result_outputs:
        out_name = _merged_result_filename(group_key, global_min, global_max)
        print(f"  - {out_name}")
    print("Merged subdirs       : retrieval_logs/ only")

    if dry_run:
        print("\n[dry-run] No files written.")
        if merged_dir.exists():
            print("  [!] Folder already exists and would be overwritten.")
        return True

    # recreate merged directory
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2) Write merged results files
    # -------------------------------------------------------------------------
    for group_key, rows in merged_result_outputs.items():
        out_name = _merged_result_filename(group_key, global_min, global_max)
        out_path = merged_dir / out_name
        try:
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2, ensure_ascii=True)
            print(f"[ok] Written merged results: {out_path}")
        except Exception as e:
            print(f"[error] Failed to write {out_path}: {e}")
            return False

    # -------------------------------------------------------------------------
    # 3) Merge retrieval_logs/ only
    # -------------------------------------------------------------------------
    try:
        merged_counts = _merge_named_subdirs(session_dirs, merged_dir)
        if merged_counts:
            print("[ok] Merged subdirectories:")
            for dirname, count in sorted(merged_counts.items()):
                print(f"  - {dirname}/ : {count} file(s) copied")
        else:
            print("[ok] No mergeable subdirectories found.")
    except Exception as e:
        print(f"[warn] Failed while merging subdirectories: {e}")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge all session result JSONs and same-named subdirs."
    )
    parser.add_argument(
        "output_dir",
        help="Output folder to merge (relative to CWD or absolute path)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing output",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    ok = merge_results(output_dir, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()