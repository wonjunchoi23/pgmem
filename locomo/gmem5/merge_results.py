"""
Merge Results Script
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple


def _parse_sample_range(dirname: str) -> Tuple[int, int]:
    parts = dirname.split("_")
    if len(parts) != 3 or parts[0] != "sample":
        raise ValueError(f"Unexpected directory name: {dirname!r}")
    return int(parts[1]), int(parts[2])


def _find_sample_dirs(output_dir: Path) -> List[Tuple[Path, int, int]]:
    result = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("sample_"):
            try:
                start, end = _parse_sample_range(item.name)
                result.append((item, start, end))
            except ValueError as exc:
                print(f"  [skip] {item.name}: {exc}")
    result.sort(key=lambda item: item[1])
    return result


def _find_results_files(sample_dir: Path) -> List[Path]:
    return sorted(
        [
            path for path in sample_dir.iterdir()
            if path.is_file() and path.name.startswith("results_") and path.suffix == ".json"
        ],
        key=lambda path: path.name,
    )


def _result_group_key(filename: str) -> str:
    return re.sub(r"_sample_\d+_\d+(?=\.json$)", "", filename)


def _merged_result_filename(group_key: str, global_min: int, global_max: int) -> str:
    stem = group_key[:-5] if group_key.endswith(".json") else group_key
    return f"{stem}_sample_{global_min}_{global_max}.json"


def _load_result_file(path: Path) -> List[Dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  [warn] Unexpected format in {path}")
            return []
        return data
    except Exception as exc:
        print(f"  [warn] Could not load {path}: {exc}")
        return []


def _validate(all_results: List[Dict]) -> List[str]:
    warnings = []
    ids = [row["sample_id"] for row in all_results if isinstance(row, dict) and "sample_id" in row]
    if not ids:
        return warnings

    seen, dupes = set(), set()
    for sample_id in ids:
        if sample_id in seen:
            dupes.add(sample_id)
        seen.add(sample_id)
    if dupes:
        warnings.append(f"Duplicate sample_ids: {sorted(dupes)}")
    return warnings


def _merge_directory_contents(src_dir: Path, dst_dir: Path) -> int:
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


def _merge_named_subdirs(sample_dirs: List[Tuple[Path, int, int]], dest_dir: Path) -> Dict[str, int]:
    merged_counts: Dict[str, int] = defaultdict(int)

    for sample_dir, _, _ in sample_dirs:
        for item in sample_dir.iterdir():
            if not item.is_dir() or item.name != "retrieval_logs":
                continue
            dst_subdir = dest_dir / item.name
            copied = _merge_directory_contents(item, dst_subdir)
            merged_counts[item.name] += copied

    return dict(merged_counts)


def merge_results(output_dir: Path, dry_run: bool = False) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Output dir : {output_dir}")
    print(f"{'=' * 60}\n")

    if not output_dir.exists():
        print(f"[error] Directory not found: {output_dir}")
        return False

    sample_dirs = _find_sample_dirs(output_dir)
    if not sample_dirs:
        print(f"[error] No sample_*/ directories found in {output_dir}")
        return False

    print(f"Found {len(sample_dirs)} sample directory/ies:")
    for _, start, end in sample_dirs:
        print(f"  sample_{start}_{end}/")
    print()

    global_min = min(start for _, start, _ in sample_dirs)
    global_max = max(end for _, _, end in sample_dirs)
    merged_dir = output_dir / f"sample_{global_min}_{global_max}"

    grouped_results: DefaultDict[str, List[Dict]] = defaultdict(list)
    for sample_dir, start, end in sample_dirs:
        result_files = _find_results_files(sample_dir)
        if not result_files:
            print(f"  sample_{start}_{end}: no results_*.json files found")
            continue
        print(f"  sample_{start}_{end}:")
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
        rows.sort(key=lambda row: row.get("sample_id", "") if isinstance(row, dict) else "")
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
    print("Merged subdirs       : retrieval_logs/ only")

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
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2, ensure_ascii=True)
            print(f"[ok] Written merged results: {out_path}")
        except Exception as exc:
            print(f"[error] Failed to write {out_path}: {exc}")
            return False

    merged_counts = _merge_named_subdirs(sample_dirs, merged_dir)
    for subdir_name, copied in merged_counts.items():
        print(f"[ok] Merged {subdir_name}/: {copied} files copied")

    print("\nDone.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge sample-based result folders.")
    parser.add_argument("output_dir", type=str, help="Path to config_*_outputs_<model>/")
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; do not write files.")
    args = parser.parse_args()

    ok = merge_results(Path(args.output_dir), dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
