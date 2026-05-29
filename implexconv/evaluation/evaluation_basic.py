"""
evaluation_basic.py — Evaluation for new-format experiment results

Input structure:
    evaluation/{root}/{subset}_{session_num}/{model}/
        ├── results_{LLM}_{subset_full}_session_{s}_{e}.json
        └── retrieval_logs/
            ├── session_0_retrieval_log.jsonl
            └── ...

Output structure:
    evaluation/{root_without_results}_eval/{subset}_{session_num}/
        ├── qa_opp_{session_num}_score.csv      (if subset==opp)
        ├── qa_sup_{session_num}_score.csv      (if subset==sup)
        ├── token_memory_stats_{subset}_{session_num}.csv
        └── qa_log_{subset}_{session_num}/      (opp only)
            └── qa_top_bottom_{subset}_{session_num}_{model}_{llm}.json

Example:
    python evaluation_basic.py \
        --root qwen3_1.7b_results \
        --subset opp \
        --session_num 500 \
        --k 5 \
        --emb-model sentence-transformers/all-MiniLM-L6-v2

Arguments:
    --root        : root folder name under evaluation/  (e.g. qwen3_1.7b_results)
    --subset      : opp or sup
    --session_num : session count integer  (e.g. 200)
    --k           : top/bottom k for QA log (default: 5)
    --emb-model   : embedding model (default: sentence-transformers/all-MiniLM-L6-v2)

token_statistics format (new):
    {
        "call_N_qa": {"input": int, "output": int, "llm_calls": int, ...},
        "total_input": int,
        "total_output": int,
        "total_llm_calls": int,
    }
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nltk.translate.meteor_score import single_meteor_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore")

BATCH_SIZE = 128
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 5
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation script for personalized dialogue experiments"
    )
    parser.add_argument("--root", required=True,
                        help="Root folder name under evaluation/  (e.g. qwen3_1.7b_results)")
    parser.add_argument("--subset", required=True, choices=["opp", "sup"])
    parser.add_argument("--session_num", type=int, required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--emb-model", default=DEFAULT_EMB_MODEL)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def is_valid_text(val) -> bool:
    return val is not None and str(val).strip() not in ("", "N/A")


def sanitize_text(val) -> str:
    s = val if isinstance(val, str) else str(val or "")
    try:
        s = s.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    except (UnicodeEncodeError, UnicodeDecodeError):
        s = s.encode("ascii", errors="replace").decode("ascii")
    return s.replace("\x00", "")


def tokenize_text(val) -> List[str]:
    return TOKEN_PATTERN.findall(sanitize_text(val).lower())


def lcs_length(tokens1: List[str], tokens2: List[str]) -> int:
    if not tokens1 or not tokens2:
        return 0

    if len(tokens1) < len(tokens2):
        short_tokens, long_tokens = tokens1, tokens2
    else:
        short_tokens, long_tokens = tokens2, tokens1

    prev = [0] * (len(short_tokens) + 1)
    for token in long_tokens:
        curr = [0]
        for idx, short_token in enumerate(short_tokens, start=1):
            if token == short_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def compute_rouge_l_score(generated: str, reference: str) -> float:
    gen_tokens = tokenize_text(generated)
    ref_tokens = tokenize_text(reference)
    if not gen_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(gen_tokens, ref_tokens)
    precision = lcs / len(gen_tokens)
    recall = lcs / len(ref_tokens)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def compute_meteor_score(generated: str, reference: str) -> float:
    gen_tokens = tokenize_text(generated)
    ref_tokens = tokenize_text(reference)
    if not gen_tokens or not ref_tokens:
        return 0.0
    return float(single_meteor_score(ref_tokens, gen_tokens))


# ---------------------------------------------------------------------------
# Embedding & similarity
# ---------------------------------------------------------------------------

def compute_similarity_scores(texts1: List[str], texts2: List[str],
                               model: SentenceTransformer) -> List[float]:
    if not texts1:
        return []
    emb1 = model.encode([sanitize_text(t) for t in texts1],
                         batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
    emb2 = model.encode([sanitize_text(t) for t in texts2],
                         batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
    return cosine_similarity(emb1, emb2).diagonal().tolist()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def get_llm_from_data(data: List[Dict]) -> str:
    """Extract LLM name from config_metadata or qa_tokens.model."""
    # prefer config_metadata (available on every session)
    for session in data:
        model = session.get("config_metadata", {}).get("model")
        if model:
            return model.split("/")[-1]
    # fallback: qa_tokens.model
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
    """Count average and total turns from retrieval log JSONL files."""
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
# QA evaluation
# ---------------------------------------------------------------------------

def _normalize_yn(val: str) -> str:
    v = val.strip().lower()
    return v if v in ("yes", "no") else "unknown"


def evaluate_qa_opp(data: List[Dict], emb_model: SentenceTransformer, k: int) -> Dict:
    """Embedding similarity between generated_answer and ground_truth_answer (opp)."""
    per_session_total, per_session_valid = [], []
    all_sims, all_rouge_l, all_meteor, all_entries = [], [], [], []

    for session in data:
        session_id = session.get("session_id")
        qa_results = session.get("qa_results", [])
        valid = [qa for qa in qa_results
                 if is_valid_text(qa.get("generated_answer"))
                 and is_valid_text(qa.get("ground_truth_answer"))]

        per_session_total.append(len(qa_results))
        per_session_valid.append(len(valid))

        if not valid:
            continue

        sims = compute_similarity_scores(
            [str(qa["generated_answer"]) for qa in valid],
            [str(qa["ground_truth_answer"]) for qa in valid],
            emb_model,
        )
        for qa, sim in zip(valid, sims):
            rouge_l = compute_rouge_l_score(
                str(qa["generated_answer"]), str(qa["ground_truth_answer"])
            )
            meteor = compute_meteor_score(
                str(qa["generated_answer"]), str(qa["ground_truth_answer"])
            )
            all_sims.append(sim)
            all_rouge_l.append(rouge_l)
            all_meteor.append(meteor)
            all_entries.append({
                "session_id": session_id,
                "question": qa.get("question", ""),
                "generated_answer": qa.get("generated_answer", ""),
                "ground_truth_answer": qa.get("ground_truth_answer", ""),
                "similarity": sim,
                "rouge_l": rouge_l,
                "meteor": meteor,
            })

    sorted_desc = sorted(all_entries, key=lambda x: x["similarity"], reverse=True)
    sorted_asc  = sorted(all_entries, key=lambda x: x["similarity"])
    return {
        "total_qa":     int(sum(per_session_total)),
        "valid_qa":     int(sum(per_session_valid)),
        "valid_avg_qa": float(np.mean(per_session_valid)) if per_session_valid else 0.0,
        "avg_emb_sim":  float(np.mean(all_sims))          if all_sims          else 0.0,
        "avg_rouge_l":  float(np.mean(all_rouge_l))       if all_rouge_l       else 0.0,
        "avg_meteor":   float(np.mean(all_meteor))        if all_meteor        else 0.0,
        "top_k":    sorted_desc[:k],
        "bottom_k": sorted_asc[:k],
    }


def evaluate_qa_sup(data: List[Dict]) -> Dict:
    """yes/no classification accuracy for supportive subset."""
    per_session_total, per_session_valid = [], []
    num_yes = num_no = num_unknown = correct = 0

    for session in data:
        qa_results = session.get("qa_results", [])
        valid = [qa for qa in qa_results
                 if is_valid_text(qa.get("generated_answer"))
                 and is_valid_text(qa.get("ground_truth_answer"))
                 and _normalize_yn(str(qa["ground_truth_answer"])) != "unknown"]

        per_session_total.append(len(qa_results))
        per_session_valid.append(len(valid))

        for qa in valid:
            gen_norm = _normalize_yn(str(qa["generated_answer"]))
            gt_norm  = _normalize_yn(str(qa["ground_truth_answer"]))
            if gen_norm == "yes":   num_yes += 1
            elif gen_norm == "no":  num_no += 1
            else:                   num_unknown += 1
            if gen_norm == gt_norm: correct += 1

    total_valid = sum(per_session_valid)
    return {
        "total_qa":     int(sum(per_session_total)),
        "valid_qa":     int(total_valid),
        "valid_avg_qa": float(np.mean(per_session_valid)) if per_session_valid else 0.0,
        "accuracy":     correct / total_valid if total_valid > 0 else 0.0,
        "num_yes":      num_yes,
        "num_no":       num_no,
        "num_unknown":  num_unknown,
    }


# ---------------------------------------------------------------------------
# Token statistics  (new format)
# ---------------------------------------------------------------------------

def evaluate_token_stats(data: List[Dict]) -> Dict:
    """Average token stats across sessions.

    New token_statistics format:
        {
            "call_N_qa": {"input": int, "output": int, "llm_calls": int, ...},
            "total_input": int,
            "total_output": int,
            "total_llm_calls": int,
        }
    """
    accum: Dict[str, List[float]] = {
        "total_input":      [],
        "total_output":     [],
        "total_llm_calls":  [],
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
    """Compute element-wise average of num_retrieved across all log entries."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    root_name  = args.root
    subset     = args.subset
    n          = args.session_num
    k          = args.k

    eval_base = root_name.replace("_results", "")

    input_dir = script_dir / root_name / f"{subset}_{n}"
    eval_dir  = script_dir / f"{eval_base}_eval" / f"{subset}_{n}"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    eval_dir.mkdir(parents=True, exist_ok=True)

    qa_csv     = eval_dir / (f"qa_opp_{n}_score.csv" if subset == "opp" else f"qa_sup_{n}_score.csv")
    tok_csv    = eval_dir / f"token_memory_stats_{subset}_{n}.csv"
    qa_log_dir = eval_dir / f"qa_log_{subset}_{n}"

    QA_OPP_COLS = [
        "model", "subset", "llm",
        "total_qa", "valid_qa", "valid_avg_qa", "avg_emb_sim", "avg_rouge_l", "avg_meteor",
        "avg_turn", "total_turn",
    ]
    QA_SUP_COLS = [
        "model", "subset", "llm",
        "total_qa", "valid_qa", "valid_avg_qa", "accuracy",
        "num_yes", "num_no", "num_unknown",
        "avg_turn", "total_turn",
    ]
    TOK_COLS = [
        "model", "subset", "llm",
        "avg_turn", "total_turn",
        "avg_total_input", "avg_total_output", "avg_llm_calls",
        "memory_type", "list_retrieved_avg", "total_retrieved_avg",
    ]

    model_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        print(f"No subdirectories found in {input_dir}")
        return

    print(f"Found {len(model_dirs)} model folder(s) in {input_dir}")
    print(f"Output → {eval_dir}")

    emb_model = None
    if subset == "opp":
        print(f"Loading embedding model: {args.emb_model}")
        emb_model = SentenceTransformer(args.emb_model)

    for model_dir in tqdm(model_dirs, desc="Evaluating models"):
        model_name = model_dir.name
        print(f"\n── {model_name}")

        already_qa  = model_name in get_evaluated_models(qa_csv, QA_OPP_COLS if subset == "opp" else QA_SUP_COLS)
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
            if subset == "opp":
                print("   [QA] Embedding similarity (opp)...")
                qa_res = evaluate_qa_opp(data, emb_model, k)

                upsert_to_csv(qa_csv, {
                    "model":        model_name,
                    "subset":       subset,
                    "llm":          llm,
                    "total_qa":     qa_res["total_qa"],
                    "valid_qa":     qa_res["valid_qa"],
                    "valid_avg_qa": round(qa_res["valid_avg_qa"], 4),
                    "avg_emb_sim":  round(qa_res["avg_emb_sim"], 6),
                    "avg_rouge_l":  round(qa_res["avg_rouge_l"], 6),
                    "avg_meteor":   round(qa_res["avg_meteor"], 6),
                    "avg_turn":     round(avg_turn, 2),
                    "total_turn":   total_turn,
                }, QA_OPP_COLS)

                print(f"   → avg_emb_sim={qa_res['avg_emb_sim']:.4f}  "
                      f"avg_rouge_l={qa_res['avg_rouge_l']:.4f}  "
                      f"avg_meteor={qa_res['avg_meteor']:.4f}  "
                      f"valid_qa={qa_res['valid_qa']}  "
                      f"valid_avg_qa={qa_res['valid_avg_qa']:.2f}  "
                      f"avg_turn={avg_turn:.2f} total_turn={total_turn}")

                qa_log_dir.mkdir(exist_ok=True)
                log_path = qa_log_dir / f"qa_top_bottom_{subset}_{n}_{model_name}_{llm}.json"
                with open(log_path, "w", encoding="utf-8") as f_out:
                    json.dump({"top_k": qa_res["top_k"], "bottom_k": qa_res["bottom_k"]},
                              f_out, indent=2, ensure_ascii=False)
                print(f"   → QA log → {log_path.name}")

            else:  # sup
                print("   [QA] Yes/No accuracy (sup)...")
                qa_res = evaluate_qa_sup(data)

                upsert_to_csv(qa_csv, {
                    "model":        model_name,
                    "subset":       subset,
                    "llm":          llm,
                    "total_qa":     qa_res["total_qa"],
                    "valid_qa":     qa_res["valid_qa"],
                    "valid_avg_qa": round(qa_res["valid_avg_qa"], 4),
                    "accuracy":     round(qa_res["accuracy"], 6),
                    "num_yes":      qa_res["num_yes"],
                    "num_no":       qa_res["num_no"],
                    "num_unknown":  qa_res["num_unknown"],
                    "avg_turn":     round(avg_turn, 2),
                    "total_turn":   total_turn,
                }, QA_SUP_COLS)

                print(f"   → accuracy={qa_res['accuracy']:.4f}  "
                      f"valid_qa={qa_res['valid_qa']}  "
                      f"valid_avg_qa={qa_res['valid_avg_qa']:.2f}  "
                      f"yes={qa_res['num_yes']} no={qa_res['num_no']} "
                      f"unknown={qa_res['num_unknown']}  "
                      f"avg_turn={avg_turn:.2f} total_turn={total_turn}")

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
                "subset":              subset,
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
