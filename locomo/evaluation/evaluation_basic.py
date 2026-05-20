"""
evaluation_basic.py

Usage:
    python evaluation_basic.py --llm qwen3_1.7b [--eval_dir /path/to/evaluation]
    python evaluation_basic.py --llm final_qwen3_1.7b
    python evaluation_basic.py --llm final_gemma3_4b

Outputs:
    evaluation/dataset_statistic.csv               -- per-sample dataset stats (sessions, turns, speakers, QA counts)
    {eval_dir}/{llm}_basic_eval/answer_stats.csv   -- F1 + BLEU-1 + METEOR + ROUGE-L + embedding similarity per model/category
    {eval_dir}/{llm}_basic_eval/sub_stats.csv      -- token usage + retrieval stats per model
"""

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import wordpunct_tokenize
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

CATEGORIES = [1, 2, 3, 4, 5]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
ANSWER_SCORE_KEYS = ("f1", "bleu_1", "meteor", "rouge_l", "sim")
DATASET_STAT_COLUMNS = [
    "sample_id",
    "speaker_a",
    "speaker_b",
    "total_sessions",
    "total_turns",
    "turns_speaker_a",
    "turns_speaker_b",
    *[f"cat{cat}_total_qa" for cat in CATEGORIES],
]
ANSWER_STAT_COLUMNS = [
    "model",
    "llm",
    "F1_tot",
    "B1_tot",
    "M_tot",
    "R_tot",
    "S_tot",
    *[f"F1_cat{cat}" for cat in CATEGORIES],
    *[f"B1_cat{cat}" for cat in CATEGORIES],
    *[f"M_cat{cat}" for cat in CATEGORIES],
    *[f"R_cat{cat}" for cat in CATEGORIES],
    *[f"S_cat{cat}" for cat in CATEGORIES],
]
SUB_STAT_COLUMNS = [
    "model",
    "llm",
    "total_turn",
    "total_qa",
    "avg_total_input",
    "avg_total_output",
    "avg_llm_calls",
    "avg_qa_input",
    "avg_qa_output",
    "memory_type",
    "list_retrieved_avg",
    "total_retrieved_avg",
    "memory_snapshot_kb",
]
TOKEN_STAT_KEYS = [
    "total_input",
    "total_output",
    "total_llm_calls",
    "num_total_api_calls",
]


# ── helpers ──────────────────────────────────────────────────────────────────


class _NoWordNet:
    def synsets(self, _word):
        return []


def load_results(model_dir: Path) -> list | None:
    """Load the results_*.json from a model directory."""
    candidates = sorted(model_dir.glob("results_*.json"))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def load_retrieval_logs(model_dir: Path) -> dict:
    """
    Returns {sample_id: [log_entries]} from retrieval_logs/*.jsonl.
    sample_id is the conv-XX part.
    """
    logs = {}
    log_dir = model_dir / "retrieval_logs"
    if not log_dir.exists():
        return logs
    for path in sorted(log_dir.glob("*.jsonl")):
        m = re.search(r"sample_(conv-\w+)_retrieval_log", path.name)
        sid = m.group(1) if m else path.stem
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        logs[sid] = entries
    return logs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def tokenize_text(text: str) -> list[str]:
    return wordpunct_tokenize(str(text).lower())


def compute_f1(hyp: str, ref: str) -> float:
    """Token-level unigram F1 between generated and reference answers."""
    hyp_tok = tokenize_text(hyp)
    ref_tok = tokenize_text(ref)
    if not hyp_tok or not ref_tok:
        return 0.0

    overlap = Counter(hyp_tok) & Counter(ref_tok)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(hyp_tok)
    recall = num_same / len(ref_tok)
    return float(2 * precision * recall / (precision + recall))


def compute_bleu_1(hyp: str, ref: str) -> float:
    """Sentence-level BLEU-1 with unigram precision and brevity penalty."""
    hyp_tok = tokenize_text(hyp)
    ref_tok = tokenize_text(ref)
    if not hyp_tok or not ref_tok:
        return 0.0
    smoothing = SmoothingFunction().method1
    return float(
        sentence_bleu(
            [ref_tok],
            hyp_tok,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smoothing,
        )
    )


def compute_meteor(hyp: str, ref: str) -> float:
    hyp_tok = tokenize_text(hyp)
    ref_tok = tokenize_text(ref)
    if not hyp_tok or not ref_tok:
        return 0.0
    return meteor_score([ref_tok], hyp_tok, wordnet=_NoWordNet())


def compute_rouge_l(hyp: str, ref: str) -> float:
    hyp_tok = tokenize_text(hyp)
    ref_tok = tokenize_text(ref)
    if not hyp_tok or not ref_tok:
        return 0.0
    lcs = longest_common_subsequence_length(ref_tok, hyp_tok)
    precision = lcs / len(hyp_tok)
    recall = lcs / len(ref_tok)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def mean_or_nan(lst: list) -> float:
    return float(np.mean(lst)) if lst else float("nan")


def longest_common_subsequence_length(seq_a: list[str], seq_b: list[str]) -> int:
    prev = [0] * (len(seq_b) + 1)
    for token_a in seq_a:
        curr = [0]
        for idx, token_b in enumerate(seq_b, start=1):
            if token_a == token_b:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def load_existing_csv(
    output_path: Path,
    columns: list[str],
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame | None:
    if not output_path.exists():
        return None

    try:
        df = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    if rename_map:
        df = df.rename(columns=rename_map)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]


def init_answer_scores() -> dict:
    scores = {"all": {metric: [] for metric in ANSWER_SCORE_KEYS}}
    for cat in CATEGORIES:
        scores[cat] = {metric: [] for metric in ANSWER_SCORE_KEYS}
    return scores


def collect_answer_pairs(results: list) -> tuple[list[str], list[str], list[int]]:
    gen_texts: list[str] = []
    ref_texts: list[str] = []
    categories: list[int] = []
    for sample in results:
        for qa in sample.get("qa_results", []):
            gen_texts.append(str(qa.get("generated_answer", "") or ""))
            ref_texts.append(str(qa.get("ground_truth_answer", "") or ""))
            categories.append(qa["category"])
    return gen_texts, ref_texts, categories


def make_answer_stats_row(model: str, llm_name: str, scores: dict) -> dict:
    row: dict[str, float | str] = {"model": model, "llm": llm_name}
    row["F1_tot"] = mean_or_nan(scores["all"]["f1"])
    row["B1_tot"] = mean_or_nan(scores["all"]["bleu_1"])
    row["M_tot"] = mean_or_nan(scores["all"]["meteor"])
    row["R_tot"] = mean_or_nan(scores["all"]["rouge_l"])
    row["S_tot"] = mean_or_nan(scores["all"]["sim"])
    for cat in CATEGORIES:
        row[f"F1_cat{cat}"] = mean_or_nan(scores[cat]["f1"])
        row[f"B1_cat{cat}"] = mean_or_nan(scores[cat]["bleu_1"])
        row[f"M_cat{cat}"] = mean_or_nan(scores[cat]["meteor"])
        row[f"R_cat{cat}"] = mean_or_nan(scores[cat]["rouge_l"])
        row[f"S_cat{cat}"] = mean_or_nan(scores[cat]["sim"])
    return row


def serialize_retrieved_avg(value) -> str:
    if isinstance(value, list):
        return str([round(v, 4) for v in value])
    if value is None:
        return ""
    return str(round(value, 4))


def get_total_turn_count(dataset: list) -> int:
    total_turns = 0
    for sample in dataset:
        conv = sample["conversation"]
        for key, turns in conv.items():
            if re.match(r"^session_\d+$", key):
                total_turns += len(turns)
    return total_turns


def get_memory_snapshot_size_kb(model_dir: Path) -> float | None:
    snap_dir = model_dir / "memory_snapshots"
    if not snap_dir.exists():
        return None
    total_bytes = sum(
        f.stat().st_size
        for f in snap_dir.rglob("*")
        if f.is_file() and f.suffix in {".json", ".jsonl"}
    )
    return round(total_bytes / 1024, 2)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_results_root(eval_dir: Path, llm_name: str) -> Path:
    exact_path = eval_dir / f"{llm_name}_results"
    if exact_path.exists():
        return exact_path

    normalized_target = normalize_name(llm_name)
    matches = []
    for candidate in sorted(eval_dir.iterdir(), key=lambda item: item.name):
        if not candidate.is_dir() or not candidate.name.endswith("_results"):
            continue
        candidate_llm = candidate.name[: -len("_results")]
        if normalize_name(candidate_llm) == normalized_target:
            matches.append(candidate)

    if len(matches) == 1:
        print(f"Resolved results folder by normalized LLM name: {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        match_list = ", ".join(str(path.name) for path in matches)
        raise FileNotFoundError(
            f"Multiple matching results folders found for llm={llm_name!r}: {match_list}"
        )

    available = ", ".join(
        sorted(
            candidate.name
            for candidate in eval_dir.iterdir()
            if candidate.is_dir() and candidate.name.endswith("_results")
        )
    )
    raise FileNotFoundError(
        f"Results folder not found for llm={llm_name!r}. "
        f"Tried exact path {exact_path}. "
        f"Available *_results folders: {available or '(none)'}"
    )


def load_embedder(model_name: str) -> SentenceTransformer | None:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        print(
            "  [warn] Could not load embedding model locally. "
            f"Similarity columns will be NaN. ({exc})"
        )
        return None


def extract_numeric_stat(stats: dict, *keys: str) -> float | None:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_sample_total_llm_calls(sample: dict) -> float | None:
    token_stats = sample.get("token_statistics", {})
    return extract_numeric_stat(token_stats, "total_llm_calls", "num_total_api_calls")


def extract_sample_qa_totals(sample: dict) -> tuple[float | None, float | None]:
    qa_results = sample.get("qa_results", [])
    qa_input = 0.0
    qa_output = 0.0
    found_qa_tokens = False

    for qa in qa_results:
        qa_tokens = qa.get("qa_tokens")
        if not isinstance(qa_tokens, dict):
            continue
        input_value = qa_tokens.get("input")
        output_value = qa_tokens.get("output")
        if isinstance(input_value, (int, float)):
            qa_input += float(input_value)
            found_qa_tokens = True
        if isinstance(output_value, (int, float)):
            qa_output += float(output_value)
            found_qa_tokens = True

    if found_qa_tokens:
        return qa_input, qa_output

    token_stats = sample.get("token_statistics", {})
    qa_call_keys = sorted(
        key
        for key, value in token_stats.items()
        if re.match(r"^call_\d+_qa$", key) and isinstance(value, dict)
    )
    for qa_call_key in qa_call_keys:
        qa_call_stats = token_stats[qa_call_key]
        input_value = extract_numeric_stat(qa_call_stats, "input")
        output_value = extract_numeric_stat(qa_call_stats, "output")
        if input_value is not None or output_value is not None:
            return input_value, output_value

    return (
        extract_numeric_stat(token_stats, "total_qa_input"),
        extract_numeric_stat(token_stats, "total_qa_output"),
    )


# ── dataset_statistic.csv ────────────────────────────────────────────────────


def build_dataset_statistic(dataset: list, output_path: Path):
    """Compute per-sample dataset stats and write dataset_statistic.csv."""
    if output_path.exists():
        print(f"  [dataset_statistic] skip existing file: {output_path}")
        return

    rows = []
    for sample in dataset:
        sid = sample["sample_id"]
        conv = sample["conversation"]
        qa_list = sample["qa"]

        session_keys = [key for key in conv if re.match(r"^session_\d+$", key)]
        total_sessions = len(session_keys)

        total_turns = 0
        speaker_counts: dict[str, int] = defaultdict(int)
        for session_key in session_keys:
            turns = conv[session_key]
            total_turns += len(turns)
            for turn in turns:
                speaker_counts[turn.get("speaker", "unknown")] += 1

        spk_a = conv.get("speaker_a", "")
        spk_b = conv.get("speaker_b", "")

        row = {
            "sample_id": sid,
            "speaker_a": spk_a,
            "speaker_b": spk_b,
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "turns_speaker_a": speaker_counts.get(spk_a, 0),
            "turns_speaker_b": speaker_counts.get(spk_b, 0),
        }
        cat_total = defaultdict(int)
        for qa in qa_list:
            cat_total[qa["category"]] += 1
        for cat in CATEGORIES:
            row[f"cat{cat}_total_qa"] = cat_total.get(cat, 0)

        rows.append(row)

    df = pd.DataFrame(rows, columns=DATASET_STAT_COLUMNS).set_index("sample_id")
    df.to_csv(output_path)
    print(f"  [dataset_statistic] -> {output_path}")


# ── answer_stats.csv ─────────────────────────────────────────────────────────


def evaluate_answers(results: list, embedder: SentenceTransformer | None) -> dict:
    """
    Returns nested dict:
    {cat_or_'all': {'f1': [...], 'bleu_1': [...], 'meteor': [...], 'rouge_l': [...], 'sim': [...]}}
    """
    scores = init_answer_scores()
    gen_texts, ref_texts, cats = collect_answer_pairs(results)
    if not gen_texts:
        return scores

    if embedder is not None:
        gen_embs = embedder.encode(gen_texts, batch_size=64, show_progress_bar=False)
        ref_embs = embedder.encode(ref_texts, batch_size=64, show_progress_bar=False)
    else:
        gen_embs = [None] * len(gen_texts)
        ref_embs = [None] * len(ref_texts)

    for gen, ref, gen_emb, ref_emb, cat in zip(
        gen_texts, ref_texts, gen_embs, ref_embs, cats
    ):
        metric_values = {
            "f1": compute_f1(gen, ref),
            "bleu_1": compute_bleu_1(gen, ref),
            "meteor": compute_meteor(gen, ref),
            "rouge_l": compute_rouge_l(gen, ref),
        }
        if gen_emb is not None and ref_emb is not None:
            metric_values["sim"] = cosine_sim(gen_emb, ref_emb)
        for metric, value in metric_values.items():
            scores["all"][metric].append(value)
        if cat in scores:
            for metric, value in metric_values.items():
                scores[cat][metric].append(value)

    return scores


def has_complete_answer_metrics(existing_df: pd.DataFrame | None, model: str) -> bool:
    """Return True only if an existing row has all required answer metric columns filled.

    This prevents old answer_stats.csv files from being skipped after adding new
    metrics such as F1 and BLEU-1.
    """
    if existing_df is None or existing_df.empty:
        return False
    model_rows = existing_df[existing_df["model"] == model]
    if model_rows.empty:
        return False
    metric_cols = [col for col in ANSWER_STAT_COLUMNS if col not in {"model", "llm"}]
    return not model_rows.iloc[0][metric_cols].isna().any()


def build_answer_stats(
    results_root: Path,
    models: list[str],
    output_path: Path,
    llm_name: str,
    embedder: SentenceTransformer,
):
    existing_df = load_existing_csv(output_path, ANSWER_STAT_COLUMNS)

    new_rows = []
    updated_models = set()
    for model in models:
        if has_complete_answer_metrics(existing_df, model):
            print(f"  [answer_stats] skip {model} (already evaluated)")
            continue

        model_dir = results_root / model
        results = load_results(model_dir)
        if results is None:
            print(f"  [answer_stats] skip {model} (no results file)")
            continue

        print(f"  [answer_stats] evaluating {model} ...")
        scores = evaluate_answers(results, embedder)
        new_rows.append(make_answer_stats_row(model, llm_name, scores))
        updated_models.add(model)

    if not new_rows:
        print("  [answer_stats] nothing new to add.")
        return

    new_df = pd.DataFrame(new_rows)[ANSWER_STAT_COLUMNS]
    if existing_df is not None:
        existing_df = existing_df[~existing_df["model"].isin(updated_models)]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(output_path, index=False)
    print(f"  [answer_stats] -> {output_path}")


# ── sub_stats.csv ─────────────────────────────────────────────────────────────


def parse_retrieval_stats(logs: dict) -> dict:
    """
    Returns {
        'memory_type': str or list,
        'list_retrieved_avg': list[float] or float,
        'total_retrieved_avg': float,
    }
    Handles both scalar (only_llm style) and list (amem style) num_retrieved.
    """
    if not logs:
        return {"memory_type": None, "list_retrieved_avg": None, "total_retrieved_avg": None}

    first_entries = next(iter(logs.values()))
    sample_entry = first_entries[0] if first_entries else {}
    memory_type = sample_entry.get("memory_type", None)

    all_retrieved = []
    for entries in logs.values():
        for entry in entries:
            num_retrieved = entry.get("num_retrieved")
            if num_retrieved is not None:
                all_retrieved.append(num_retrieved)

    if not all_retrieved:
        return {"memory_type": memory_type, "list_retrieved_avg": None, "total_retrieved_avg": None}

    first = all_retrieved[0]
    if isinstance(first, list):
        max_len = max(len(item) for item in all_retrieved)
        arr = np.zeros((len(all_retrieved), max_len))
        for idx, item in enumerate(all_retrieved):
            arr[idx, : len(item)] = item
        list_retrieved_avg = arr.mean(axis=0).tolist()
        total_retrieved_avg = float(arr.sum(axis=1).mean())
    else:
        values = [float(item) for item in all_retrieved]
        list_retrieved_avg = float(np.mean(values))
        total_retrieved_avg = float(np.mean(values))

    return {
        "memory_type": memory_type,
        "list_retrieved_avg": list_retrieved_avg,
        "total_retrieved_avg": total_retrieved_avg,
    }


def build_sub_stats(
    results_root: Path,
    models: list[str],
    output_path: Path,
    llm_name: str,
    dataset: list,
):
    total_turn_all = get_total_turn_count(dataset)
    existing_df = load_existing_csv(
        output_path,
        SUB_STAT_COLUMNS,
        rename_map={"avg_api_calls": "avg_llm_calls"},
    )
    existing_models = set(existing_df["model"].tolist()) if existing_df is not None else set()

    new_rows = []
    for model in models:
        if model in existing_models:
            print(f"  [sub_stats] skip {model} (already evaluated)")
            continue

        model_dir = results_root / model
        results = load_results(model_dir)
        if results is None:
            print(f"  [sub_stats] skip {model} (no results file)")
            continue

        print(f"  [sub_stats] processing {model} ...")
        token_accum: dict[str, list] = defaultdict(list)
        total_qa = 0

        for sample in results:
            token_stats = sample.get("token_statistics", {})
            for key in TOKEN_STAT_KEYS:
                if key in token_stats:
                    token_accum[key].append(token_stats[key])
            total_llm_calls = extract_sample_total_llm_calls(sample)
            if total_llm_calls is not None:
                token_accum["resolved_total_llm_calls"].append(total_llm_calls)
            qa_input, qa_output = extract_sample_qa_totals(sample)
            if qa_input is not None:
                token_accum["resolved_total_qa_input"].append(qa_input)
            if qa_output is not None:
                token_accum["resolved_total_qa_output"].append(qa_output)
            total_qa += len(sample.get("qa_results", []))

        ret_stats = parse_retrieval_stats(load_retrieval_logs(model_dir))
        new_rows.append(
            {
                "model": model,
                "llm": llm_name,
                "total_turn": total_turn_all,
                "total_qa": total_qa,
                "avg_total_input": mean_or_nan(token_accum.get("total_input", [])),
                "avg_total_output": mean_or_nan(token_accum.get("total_output", [])),
                "avg_llm_calls": mean_or_nan(
                    token_accum.get("resolved_total_llm_calls", [])
                ),
                "avg_qa_input": mean_or_nan(
                    token_accum.get("resolved_total_qa_input", [])
                ),
                "avg_qa_output": mean_or_nan(
                    token_accum.get("resolved_total_qa_output", [])
                ),
                "memory_type": str(ret_stats["memory_type"]) if ret_stats["memory_type"] is not None else "",
                "list_retrieved_avg": serialize_retrieved_avg(ret_stats["list_retrieved_avg"]),
                "total_retrieved_avg": (
                    round(ret_stats["total_retrieved_avg"], 4)
                    if ret_stats["total_retrieved_avg"] is not None
                    else ""
                ),
                "memory_snapshot_kb": get_memory_snapshot_size_kb(model_dir) or "",
            }
        )

    if not new_rows:
        print("  [sub_stats] nothing new to add.")
        return

    new_df = pd.DataFrame(new_rows)[SUB_STAT_COLUMNS]
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(output_path, index=False)
    print(f"  [sub_stats] -> {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True, help="LLM name, e.g. qwen3_1.7b")
    parser.add_argument(
        "--eval_dir",
        default=None,
        help="Path to the evaluation/ directory (defaults to the directory of this script)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    eval_dir = Path(args.eval_dir) if args.eval_dir else script_dir

    results_root = resolve_results_root(eval_dir, args.llm)
    output_dir = eval_dir / f"{args.llm}_basic_eval"

    output_dir.mkdir(exist_ok=True)
    print(f"Results root : {results_root}")
    print(f"Output dir   : {output_dir}")

    models = sorted(d.name for d in results_root.iterdir() if d.is_dir())
    print(f"Models found : {models}\n")

    dataset_path = eval_dir.parent / "dataset" / "locomo10.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    print("[1/3] Building dataset_statistic.csv ...")
    build_dataset_statistic(dataset, script_dir / "dataset_statistic.csv")

    print("\n[2/3] Building answer_stats.csv ...")
    print("  Loading embedding model ...")
    embedder = load_embedder(EMBED_MODEL_NAME)
    build_answer_stats(
        results_root,
        models,
        output_dir / "answer_stats.csv",
        args.llm,
        embedder,
    )

    print("\n[3/3] Building sub_stats.csv ...")
    build_sub_stats(
        results_root,
        models,
        output_dir / "sub_stats.csv",
        args.llm,
        dataset,
    )


if __name__ == "__main__":
    main()
