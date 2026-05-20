"""
evaluation_basic.py — token + memory statistics for PrefEval runs.

For each memory module under exp_prefeval/{module}/, scans for
config_{N}_outputs_{llm} folders and aggregates stats.json + retrieval_log.jsonl
into a single CSV at evaluation/{llm}_basic/basic_summary.csv.

Usage:
    python evaluation_basic.py --llm Qwen3-1.7B
    python evaluation_basic.py --llm Qwen3-1.7B --modules amem dense
    python evaluation_basic.py --llm Qwen3-1.7B --overwrite
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_MODULES = ["amem", "ldagent", "dense", "memorybank", "theanine", "gmem"]

CONFIG_FOLDER_RE = re.compile(r"^config_(\d+)_outputs_(.+)$")

CSV_COLS = [
    "module_config", "module", "config_num", "llm",
    "num_qa", "num_checkpoints",
    "total_input", "total_output", "total_llm_calls",
    "qa_input", "qa_output", "qa_llm_calls",
    "mem_input", "mem_output", "mem_llm_calls",
    "avg_input_per_qa", "avg_output_per_qa",
    "num_retrieval_logs",
    "avg_num_retrieved", "avg_top_retrieval_score",
    "memory_type", "list_retrieved_avg",
    "module_specific",
]


def parse_args():
    p = argparse.ArgumentParser(description="PrefEval basic stats aggregator")
    p.add_argument("--llm", required=True,
                   help="LLM tag in folder names, e.g. Qwen3-1.7B")
    p.add_argument("--modules", nargs="+", default=DEFAULT_MODULES,
                   help=f"Memory modules to scan (default: {DEFAULT_MODULES})")
    p.add_argument("--root", default=None,
                   help="exp_prefeval root (default: parent of this script)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-evaluate even if module_config row already exists in CSV")
    return p.parse_args()


def find_config_dirs(module_dir: Path, llm: str) -> List[tuple]:
    """Return [(config_num, path), ...] matching config_{N}_outputs_{llm}."""
    out = []
    if not module_dir.is_dir():
        return out
    for sub in sorted(module_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = CONFIG_FOLDER_RE.match(sub.name)
        if not m:
            continue
        if m.group(2) != llm:
            continue
        out.append((int(m.group(1)), sub))
    return out


def load_stats(stats_path: Path) -> Dict[str, Any]:
    if not stats_path.exists():
        return {}
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def split_call_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Sum input/output/llm_calls split into qa vs non-qa (other module-side calls)."""
    qa_in = qa_out = qa_calls = 0
    mem_in = mem_out = mem_calls = 0
    module_specific: Dict[str, Any] = {}

    for key, val in stats.items():
        if key == "checkpoints_completed":
            continue
        if not isinstance(val, dict):
            continue
        is_call = key.startswith("call_")
        is_qa = is_call and key.endswith("_qa")
        if is_qa:
            qa_in    += int(val.get("input", 0) or 0)
            qa_out   += int(val.get("output", 0) or 0)
            qa_calls += int(val.get("llm_calls", 0) or 0)
        elif is_call:
            mem_in    += int(val.get("input", 0) or 0)
            mem_out   += int(val.get("output", 0) or 0)
            mem_calls += int(val.get("llm_calls", 0) or 0)
        else:
            # non-call dict-valued field (e.g. evolution, forgetting)
            module_specific[key] = val

    return {
        "qa_input":     qa_in,
        "qa_output":    qa_out,
        "qa_llm_calls": qa_calls,
        "mem_input":    mem_in,
        "mem_output":   mem_out,
        "mem_llm_calls": mem_calls,
        "total_input":  qa_in + mem_in,
        "total_output": qa_out + mem_out,
        "total_llm_calls": qa_calls + mem_calls,
        "module_specific": module_specific,
    }


def count_results(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    n = 0
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def aggregate_retrieval_log(log_path: Path) -> Dict[str, Any]:
    """Average retrieval-side stats from retrieval_log.jsonl."""
    if not log_path.exists():
        return {
            "num_retrieval_logs": 0,
            "avg_num_retrieved": 0.0,
            "avg_top_retrieval_score": 0.0,
            "memory_type": [],
            "list_retrieved_avg": [],
        }

    n_logs = 0
    n_retrieved: List[int] = []
    top_scores: List[float] = []
    memory_type: Optional[List[str]] = None
    per_position: List[List[float]] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_logs += 1

            items = entry.get("retrieved_items", []) or []
            n_retrieved.append(len(items))

            scores = entry.get("retrieval_scores", []) or []
            if scores:
                try:
                    top_scores.append(float(scores[0]))
                except (TypeError, ValueError):
                    pass

            nr = entry.get("num_retrieved")
            mt = entry.get("memory_type")
            if isinstance(nr, (int, float)):
                nr = [nr]
            if isinstance(mt, str):
                mt = [mt]
            if memory_type is None and isinstance(mt, list) and mt:
                memory_type = mt
                per_position = [[] for _ in mt]
            if isinstance(nr, list) and per_position:
                for i, c in enumerate(nr):
                    if i < len(per_position):
                        try:
                            per_position[i].append(float(c))
                        except (TypeError, ValueError):
                            pass

    return {
        "num_retrieval_logs":      n_logs,
        "avg_num_retrieved":       float(np.mean(n_retrieved)) if n_retrieved else 0.0,
        "avg_top_retrieval_score": float(np.mean(top_scores))  if top_scores  else 0.0,
        "memory_type":             memory_type or [],
        "list_retrieved_avg":      [round(float(np.mean(p)), 4) if p else 0.0
                                    for p in per_position],
    }


def already_done(csv_path: Path, module_config: str) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception:
        return False
    return "module_config" in df.columns and (df["module_config"] == module_config).any()


def upsert_row(csv_path: Path, row: Dict[str, Any]):
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
        except Exception:
            df = pd.DataFrame(columns=CSV_COLS)
    else:
        df = pd.DataFrame(columns=CSV_COLS)

    for c in CSV_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[CSV_COLS]

    df = df[df["module_config"] != row["module_config"]]
    new_row = pd.DataFrame([{c: row.get(c, "") for c in CSV_COLS}])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values(["module", "config_num"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def evaluate_one(module: str, config_num: int, cfg_dir: Path, llm: str) -> Dict[str, Any]:
    stats = load_stats(cfg_dir / "stats.json")
    call_breakdown = split_call_stats(stats)
    n_ckpt = len(stats.get("checkpoints_completed", [])) if isinstance(stats, dict) else 0

    num_qa = count_results(cfg_dir / "results.jsonl")
    retr = aggregate_retrieval_log(cfg_dir / "retrieval_log.jsonl")

    avg_in_per_qa  = call_breakdown["qa_input"]  / num_qa if num_qa else 0.0
    avg_out_per_qa = call_breakdown["qa_output"] / num_qa if num_qa else 0.0

    return {
        "module_config":    f"{module}_{config_num}",
        "module":           module,
        "config_num":       config_num,
        "llm":              llm,
        "num_qa":           num_qa,
        "num_checkpoints":  n_ckpt,
        "total_input":      call_breakdown["total_input"],
        "total_output":     call_breakdown["total_output"],
        "total_llm_calls":  call_breakdown["total_llm_calls"],
        "qa_input":         call_breakdown["qa_input"],
        "qa_output":        call_breakdown["qa_output"],
        "qa_llm_calls":     call_breakdown["qa_llm_calls"],
        "mem_input":        call_breakdown["mem_input"],
        "mem_output":       call_breakdown["mem_output"],
        "mem_llm_calls":    call_breakdown["mem_llm_calls"],
        "avg_input_per_qa":  round(avg_in_per_qa,  2),
        "avg_output_per_qa": round(avg_out_per_qa, 2),
        "num_retrieval_logs":      retr["num_retrieval_logs"],
        "avg_num_retrieved":       round(retr["avg_num_retrieved"], 4),
        "avg_top_retrieval_score": round(retr["avg_top_retrieval_score"], 6),
        "memory_type":       json.dumps(retr["memory_type"], ensure_ascii=False),
        "list_retrieved_avg": json.dumps(retr["list_retrieved_avg"]),
        "module_specific":   json.dumps(call_breakdown["module_specific"], ensure_ascii=False),
    }


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    root = Path(args.root) if args.root else script_dir.parent
    out_dir = script_dir / f"{args.llm}_basic"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "basic_summary.csv"

    print(f"Root  : {root}")
    print(f"LLM   : {args.llm}")
    print(f"Output: {csv_path}")

    found_any = False
    for module in args.modules:
        module_dir = root / module
        if not module_dir.is_dir():
            print(f"[skip] module folder not found: {module}/")
            continue
        cfg_dirs = find_config_dirs(module_dir, args.llm)
        if not cfg_dirs:
            print(f"[skip] no config_*_outputs_{args.llm} under {module}/")
            continue

        for config_num, cfg_dir in cfg_dirs:
            mc = f"{module}_{config_num}"
            if not args.overwrite and already_done(csv_path, mc):
                print(f"[skip] {mc}: already in CSV")
                continue

            print(f"[eval] {mc}  ({cfg_dir})")
            row = evaluate_one(module, config_num, cfg_dir, args.llm)
            upsert_row(csv_path, row)
            print(f"   total_input={row['total_input']:,}  qa_input={row['qa_input']:,}  "
                  f"num_qa={row['num_qa']}  avg_retrieved={row['avg_num_retrieved']}")
            found_any = True

    if not found_any:
        print("Nothing evaluated.")
    else:
        print(f"\nDone → {csv_path}")


if __name__ == "__main__":
    main()
