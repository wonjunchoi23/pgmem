"""
Shared merge-results script for top-level experiment modules.

Usage:
    python merge_results.py gmem config_0_outputs_Qwen3-1.7B_opposed
    python merge_results.py memorybank config_0_outputs_Qwen3-1.7B_opposed
    python merge_results.py ubllm config_0_outputs_Qwen3-1.7B_opposed --dry-run
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent
VALID_MODULES = (
    "amem",
    "gmem",
    "ldagent",
    "memorybank",
    "ubllm",
    "theanine",
)
MERGEABLE_SUBDIRS = ("retrieval_logs", "memory_snapshots")
PROMPT_LOG_SUBDIR = "prompt_log"
MERGEABLE_RESULT_PREFIXES = ("results_", "memory_build_stats_")


def _parse_session_range(dirname: str) -> Tuple[int, int]:
    """Parse 'session_<start>_<end>' -> (start, end)."""
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
            except ValueError as exc:
                print(f"  [skip] {item.name}: {exc}")
    result.sort(key=lambda x: x[1])
    return result


def _find_results_files(session_dir: Path) -> List[Path]:
    """Find all mergeable result files directly under session_dir."""
    return sorted(
        [
            path
            for path in session_dir.iterdir()
            if path.is_file()
            and path.suffix == ".json"
            and path.name.startswith(MERGEABLE_RESULT_PREFIXES)
        ],
        key=lambda path: path.name,
    )


def _result_group_key(filename: str) -> str:
    """Normalize results_xxx_session_0_99.json -> results_xxx.json."""
    return re.sub(r"_session_\d+_\d+(?=\.json$)", "", filename)


def _merged_result_filename(group_key: str, global_min: int, global_max: int) -> str:
    """Convert results_xxx.json -> results_xxx_session_{min}_{max}.json."""
    stem = group_key[:-5] if group_key.endswith(".json") else group_key
    return f"{stem}_session_{global_min}_{global_max}.json"


def _load_result_file(path: Path) -> List[Dict]:
    """Load one results_*.json file. Expected: list of dicts."""
    try:
        with open(path) as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            print(f"  [warn] Unexpected format in {path}")
            return []
        return data
    except Exception as exc:
        print(f"  [warn] Could not load {path}: {exc}")
        return []


def _validate(all_results: List[Dict]) -> List[str]:
    """Check for duplicate or missing session_ids."""
    warnings = []

    ids = [row["session_id"] for row in all_results if isinstance(row, dict) and "session_id" in row]
    if not ids:
        return warnings

    seen, dupes = set(), set()
    for session_id in ids:
        if session_id in seen:
            dupes.add(session_id)
        seen.add(session_id)
    if dupes:
        warnings.append(f"Duplicate session_ids: {sorted(dupes)}")

    sorted_ids = sorted(ids)
    gaps = []
    for idx in range(len(sorted_ids) - 1):
        if sorted_ids[idx + 1] != sorted_ids[idx] + 1:
            gaps.append((sorted_ids[idx], sorted_ids[idx + 1]))
    if gaps:
        for start, end in gaps:
            warnings.append(
                f"Gap between session_id {start} and {end} (missing: {list(range(start + 1, end))})"
            )

    return warnings


def _merge_directory_contents(src_dir: Path, dst_dir: Path) -> int:
    """
    Recursively merge src_dir into dst_dir.
    - folders with same name are merged
    - files with same path are overwritten
    Returns number of copied files.
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


def _merge_prompt_log(src_prompt_log: Path, dst_prompt_log: Path) -> int:
    """
    Merge prompt_log/, but only call_*/ folders whose name ends with '_qa'.
    Layout: prompt_log/session_N/call_X_<name>/...
    """
    copied = 0
    for session_subdir in src_prompt_log.iterdir():
        if not session_subdir.is_dir():
            continue
        for call_dir in session_subdir.iterdir():
            if not call_dir.is_dir() or not call_dir.name.endswith("_qa"):
                continue
            dst_call_dir = dst_prompt_log / session_subdir.name / call_dir.name
            copied += _merge_directory_contents(call_dir, dst_call_dir)
    return copied


def _merge_named_subdirs(session_dirs: List[Tuple[Path, int, int]], dest_dir: Path) -> Dict[str, int]:
    """Merge named output subdirectories under each session dir into the merged output."""
    merged_counts: Dict[str, int] = defaultdict(int)

    for session_dir, _, _ in session_dirs:
        for item in session_dir.iterdir():
            if not item.is_dir():
                continue

            if item.name in MERGEABLE_SUBDIRS:
                dst_subdir = dest_dir / item.name
                copied = _merge_directory_contents(item, dst_subdir)
                merged_counts[item.name] += copied
            elif item.name == PROMPT_LOG_SUBDIR:
                dst_subdir = dest_dir / item.name
                copied = _merge_prompt_log(item, dst_subdir)
                merged_counts[item.name] += copied

    return dict(merged_counts)


def _resolve_output_dir(module: str, output_dir_name: str) -> Path:
    rel_path = Path(output_dir_name)
    if rel_path.is_absolute():
        raise ValueError("output_dir_name must be a folder name, not an absolute path.")
    if len(rel_path.parts) != 1 or rel_path.name != output_dir_name:
        raise ValueError("output_dir_name must be a single folder name without path separators.")
    return (ROOT_DIR / module / rel_path).resolve()


def merge_results(output_dir: Path, dry_run: bool = False) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Output dir : {output_dir}")
    print(f"{'=' * 60}\n")

    if not output_dir.exists():
        print(f"[error] Directory not found: {output_dir}")
        return False

    session_dirs = _find_session_dirs(output_dir)
    if not session_dirs:
        print(f"[error] No session_*/ directories found in {output_dir}")
        return False

    print(f"Found {len(session_dirs)} session directory/ies:")
    for _, start, end in session_dirs:
        print(f"  session_{start}_{end}/")
    print()

    global_min = min(start for _, start, _ in session_dirs)
    global_max = max(end for _, _, end in session_dirs)

    merged_dirname = f"session_{global_min}_{global_max}"
    merged_dir = output_dir / merged_dirname

    grouped_results: DefaultDict[str, List[Dict]] = defaultdict(list)

    for session_dir, start, end in session_dirs:
        result_files = _find_results_files(session_dir)
        if not result_files:
            print(f"  session_{start}_{end}: no results_*.json files found")
            continue

        print(f"  session_{start}_{end}:")
        for result_file in result_files:
            chunk = _load_result_file(result_file)
            group_key = _result_group_key(result_file.name)
            grouped_results[group_key].extend(chunk)
            print(f"    - {result_file.name}: {len(chunk)} rows loaded -> group '{group_key}'")

    if not grouped_results:
        print("\n[error] No results_*.json files loaded.")
        return False

    merged_result_outputs: Dict[str, List[Dict]] = {}
    for group_key, rows in grouped_results.items():
        rows.sort(key=lambda row: row.get("session_id", -1) if isinstance(row, dict) else -1)
        merged_result_outputs[group_key] = rows

        warnings_list = _validate(rows)
        if warnings_list:
            print(f"\nWarnings for {group_key}:")
            for warning in warnings_list:
                print(f"  ! {warning}")
        else:
            print(f"\n  [ok] {group_key}: no gaps or duplicates detected.")

    print(f"\nOutput folder        : {merged_dir}/")
    print("Merged result files  :")
    for group_key in merged_result_outputs:
        out_name = _merged_result_filename(group_key, global_min, global_max)
        print(f"  - {out_name}")
    merged_subdir_names = ", ".join(
        f"{dirname}/" for dirname in (*MERGEABLE_SUBDIRS, f"{PROMPT_LOG_SUBDIR} (only call_*_qa)")
    )
    print(f"Merged subdirs       : {merged_subdir_names}")

    if dry_run:
        print("\n[dry-run] No files written.")
        if merged_dir.exists():
            print("  [!] Folder already exists and would be overwritten.")
        return True

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    for group_key, rows in merged_result_outputs.items():
        out_name = _merged_result_filename(group_key, global_min, global_max)
        out_path = merged_dir / out_name
        try:
            with open(out_path, "w") as handle:
                json.dump(rows, handle, indent=2, ensure_ascii=True)
            print(f"[ok] Written merged results: {out_path}")
        except Exception as exc:
            print(f"[error] Failed to write {out_path}: {exc}")
            return False

    try:
        merged_counts = _merge_named_subdirs(session_dirs, merged_dir)
        if merged_counts:
            print("[ok] Merged subdirectories:")
            for dirname, count in sorted(merged_counts.items()):
                print(f"  - {dirname}/ : {count} file(s) copied")
        else:
            print("[ok] No mergeable subdirectories found.")
    except Exception as exc:
        print(f"[warn] Failed while merging subdirectories: {exc}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge session-range results for a top-level experiment module."
    )
    parser.add_argument("module", choices=VALID_MODULES, help="Top-level module directory name")
    parser.add_argument(
        "output_dir_name",
        help="Output directory name inside the module directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing output",
    )
    args = parser.parse_args()

    try:
        output_dir = _resolve_output_dir(args.module, args.output_dir_name)
    except ValueError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    ok = merge_results(output_dir, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
