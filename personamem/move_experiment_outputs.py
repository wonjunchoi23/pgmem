#!/usr/bin/env python3
"""Move result files and retrieval logs from experiment run folders.

Example:
    python move_experiment_outputs.py \
        gmem6/config_0_outputs_Qwen3-1.7B_32k \
        collected_outputs \
        qwen3_1_7b_32k

This scans every direct child directory under SOURCE_DIR. For each child, it
moves files whose names start with "results_" and the "retrieval_logs"
directory into DEST_DIR/RUN_NAME/<child-directory-name>/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move results_* files and retrieval_logs folders from every direct "
            "child folder of a source experiment directory."
        )
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Experiment output directory containing run/session subfolders.",
    )
    parser.add_argument(
        "dest_dir",
        type=Path,
        help="Directory where collected outputs should be moved.",
    )
    parser.add_argument(
        "run_name",
        help="Name for the collected output folder created under dest_dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved without changing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination file or folder if it already exists.",
    )
    return parser.parse_args()


def ensure_directory(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def move_path(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}. Use --overwrite to replace it."
            )
        if dry_run:
            print(f"WOULD REMOVE: {dst}")
        elif dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    print(f"{'WOULD MOVE' if dry_run else 'MOVE'}: {src} -> {dst}")
    if not dry_run:
        shutil.move(str(src), str(dst))


def collect_outputs(
    source_dir: Path,
    dest_dir: Path,
    run_name: str,
    overwrite: bool,
    dry_run: bool,
) -> int:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")

    target_root = dest_dir / run_name
    moved_count = 0

    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue

        outputs: list[Path] = sorted(
            path for path in child.iterdir() if path.is_file() and path.name.startswith("results_")
        )
        retrieval_logs = child / "retrieval_logs"
        if retrieval_logs.is_dir():
            outputs.append(retrieval_logs)

        if not outputs:
            print(f"SKIP: no results_* files or retrieval_logs in {child}")
            continue

        child_target = target_root / child.name
        ensure_directory(child_target, dry_run)

        for output in outputs:
            move_path(output, child_target / output.name, overwrite, dry_run)
            moved_count += 1

    return moved_count


def main() -> None:
    args = parse_args()
    moved_count = collect_outputs(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"Done. {'Would move' if args.dry_run else 'Moved'} {moved_count} item(s).")


if __name__ == "__main__":
    main()
