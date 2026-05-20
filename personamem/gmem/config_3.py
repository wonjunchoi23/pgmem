"""Configuration for GraphMem v6 on PersonaMem."""

import os
import sys
from pathlib import Path


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

os.environ["HF_TOKEN"] = "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"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# =============================================================================
# PATH SETTINGS
# =============================================================================

CONFIG_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CONFIG_DIR.parent

LLM_MODULE_DIR = PROJECT_ROOT / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))

DATASET_DIR = PROJECT_ROOT / "dataset"

DATASET_QUESTIONS_32K  = DATASET_DIR / "questions_32k.csv"
DATASET_CONTEXTS_32K   = DATASET_DIR / "shared_contexts_32k.jsonl"
DATASET_QUESTIONS_128K = DATASET_DIR / "questions_128k.csv"
DATASET_CONTEXTS_128K  = DATASET_DIR / "shared_contexts_128k.jsonl"
DATASET_QUESTIONS_1M   = DATASET_DIR / "questions_1M.csv"
DATASET_CONTEXTS_1M    = DATASET_DIR / "shared_contexts_1M.jsonl"

BENCHMARK_SIZES = ["32k", "128k", "1M"]

BASE_OUTPUT_DIR = CONFIG_DIR
LOG_DIR = CONFIG_DIR / "logs"
LOG_TO_FILE = True
LOG_LEVEL = "INFO"


# =============================================================================
# LLM SETTINGS
# =============================================================================

LLM_ENGINE = "vllm"

DEFAULT_VLLM_CONFIG = {
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.5,
    "download_dir": None,
}

TOGETHER_CONFIG = {
    "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "api_key": os.getenv("TOGETHER_API_KEY"),
}

OPENAI_CONFIG = {
    "model_name": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

TEMPERATURE = 0.7
MAX_TOKENS = 1000

# Per-call output caps for the 9 internal calls. Each is set conservatively
# (typically 3-5× the empirically expected output size).
MAX_TOKENS_STATE              = 1000  # ②   1 state object
MAX_TOKENS_STATE_NEW_REL      = 800   # ②b  ≤ C(N,2)+N×prev judgments (N=STATE_MAX_COUNT=1)
MAX_TOKENS_EPISODE            = 1000  # ③   1 episode object
MAX_TOKENS_EPISODE_NEW_REL    = 2000  # ③b  prev_episode + chunk_states judgments
MAX_TOKENS_TRAIT              = 1200  # ④   0 or 1 trait object
MAX_TOKENS_TRAIT_EVIDENCE_5A  = 2500  # ⑤a  recent_states + recent_episodes + prev_trait judgments
MAX_TOKENS_TRAIT_EXTRA_REL_5B = 1500  # ⑤b  ≤ TOPK_STATE+TOPK_EPISODE judgments
MAX_TOKENS_STATE_STATE_5C     = 1000  # ⑤c  ≤ 2 × STATE_STATE_EXTRA_REL_TOPK judgments (pair-level sem_topK ∪ lex_topK)
MAX_TOKENS_STATE_EPISODE_5D   = 800   # ⑤d  ≤ 2 × STATE_EPISODE_EXTRA_REL_TOPK judgments (pair-level sem_topK ∪ lex_topK)

JSON_RETRY = 5
JUDGMENT_RETRY = 3       # retry count when judgments array is empty but expected_judgment_count > 0


# =============================================================================
# BATCH SETTINGS
# =============================================================================

BATCH_SIZE = 4
QA_BATCH_SIZE = 64


# =============================================================================
# GRAPHMEM V6 SETTINGS
# =============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Extraction
STATE_EXTRACTION_H = 1           # per-turn extraction
STATE_MAX_COUNT = 1              # at most one state per ② call
STATE_REF_CONTEXT_TURNS = 3      # prior (user, assistant) pairs shown as read-only reference in ②
CHUNK_SIZE_CONV = 1
TRAIT_EXTRACTION_CHUNKS = 2
TRAIT_MAX_COUNT = 1
MAX_KEYWORDS = 7
MAX_DOMAIN_LABELS = 7
MIN_DOMAIN_LABELS = 5

# Seed retrieval top-k
K_CONTEXT = 24
K_EPISODE = 8
K_EPISODE_FINAL = 5      # final episode count after evidence-based scoring of pooled episodes
K_STATE = 20
K_TRAIT = 8
K_APS = 4

# Seed retrieval weights (context uses fixed pair; states / episodes / traits share scope-dependent pair)
W_SEM_C = 0.65
W_OV_C = 0.35

W_SEM_NARROW = 0.60
W_OV_NARROW = 0.40
W_SEM_BROAD = 0.85
W_OV_BROAD = 0.15

# Pair similarity weights — dead (kept for backward-compat with older configs);
# ⑤b/⑤c/⑤d compute semantic and lexical rankings independently and union them.
W_PAIR_SEM = 0.7
W_PAIR_LEX = 0.3

# Final set
K_T_FINAL = 6            # number of traits kept in the final set
K_SF = 20
TRAIT_VALIDATION_TAU = 0.7
W_SR = 0.5               # support-ratio weight in unified final scoring; seed-score weight is (1 - W_SR)

# Graph traversal
SIGN_PROP_HOP_CAP = 10

# Seed turn-neighbor expansion: after seed retrieval, pull state/episode nodes
# within ± SEED_TURN_NEIGHBOR_DELTA turns of each seed_s/seed_m (same conv_id)
# into the pool as plain members (not expansion drivers). Recall safety net
# for sparse-extraction cases.
SEED_TURN_NEIGHBOR_ENABLED = True
SEED_TURN_NEIGHBOR_DELTA = 1

# APS and shift control
APS_EXCLUDE_SHIFT_SOURCE = True
ENABLE_SHIFT_CHAIN_PRUNING = True
STRICT_HIGH_DEFAULT_LOW = True

# Additional relation extraction (⑤b/⑤c/⑤d)
ENABLE_EXTRA_RELATION_EXTRACTION = True
EXTRA_REL_ONLY_IF_UNCONNECTED = True
TRAIT_EXTRA_REL_TOPK_STATE = 7
TRAIT_EXTRA_REL_TOPK_EPISODE = 3
STATE_STATE_EXTRA_REL_TOPK = 5
STATE_EPISODE_EXTRA_REL_TOPK = 3
STATE_NEW_REL_PREV_WINDOW = 5    # global recent prev-state window for ②b

# Context cache
CONTEXT_CACHE_SIZE = 10

# Sub-block chunking: split each block into CHUNK_FACTOR virtual chunks.
# Episode extracts once per virtual chunk; trait every TRAIT_EXTRACTION_CHUNKS chunks.
# 32k: 1 chunk/block  → episode/block,    trait/2 blocks
# 128k/1M: 2 chunks/block → episode/½ block, trait/1 block
CHUNK_FACTOR = {"32k": 2, "128k": 2, "1M": 2}

# Time model
# PersonaMem: 1 block = 1 virtual day.
# TIME_PER_CONV_ID_HOURS is set at runtime to 24 // CHUNK_FACTOR[benchmark_size].
CONV_IDS_PER_DAY = 1
TIME_PER_CONV_ID_HOURS = 24   # overridden at runtime for 128k/1M
TIME_PER_TURN_MINUTES = 10
MINUTES_PER_TURN = 10

# QA / Response serialization
INCLUDE_RECENT_CONVERSATION_FOR_QA = False
QA_CONTEXT_PAIRS = 5     # number of most-recent (user, agent) pairs included in QA prompt when above is True

# Experiment
ENABLE_CHECKPOINTING = True
CHECKPOINT_INTERVAL = 1
TIMING_CONV_ID = 0
SAVE_MEMORY_SNAPSHOTS = True
ENABLE_LLM_CALL_LOGGING = True
# When ENABLE_LLM_CALL_LOGGING is True, only contexts with context_index < this
# value have their per-call prompts/outputs persisted under prompt_log/.
LLM_CALL_LOG_FIRST_N_SESSIONS = 10


# =============================================================================
# PATH HELPERS
# =============================================================================

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_dataset_paths(benchmark_size: str):
    """Return (questions_path, contexts_path) for the given benchmark size."""
    if benchmark_size not in BENCHMARK_SIZES:
        raise ValueError(f"benchmark_size must be one of {BENCHMARK_SIZES}, got '{benchmark_size}'")
    return (
        DATASET_DIR / f"questions_{benchmark_size}.csv",
        DATASET_DIR / f"shared_contexts_{benchmark_size}.jsonl",
    )


def get_output_dir(model_path: str, benchmark_size: str, config_name: str = "config_0") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}_{benchmark_size}"


def get_session_dir(
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    return get_output_dir(model_path, benchmark_size, config_name) / f"session_{start_session}_{end_session}"


def _suffix_tag(suffix: str | None) -> str:
    # Returns "__<suffix>" or "". QA-only runs use this to keep results /
    # checkpoint / retrieval_logs / prompt_log separate from the full run,
    # while sharing memory_snapshots with the original Phase1.
    return f"__{suffix}" if suffix else ""


def get_results_file(
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
    results_suffix: str | None = None,
) -> Path:
    model_name = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, benchmark_size, start_session, end_session, config_name)
    tag = _suffix_tag(results_suffix)
    return session_dir / f"results_{model_name}{tag}_{benchmark_size}_session_{start_session}_{end_session}.json"


def get_checkpoint_file(
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
    results_suffix: str | None = None,
) -> Path:
    model_name = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, benchmark_size, start_session, end_session, config_name)
    tag = _suffix_tag(results_suffix)
    return session_dir / f"checkpoint_{model_name}{tag}_{benchmark_size}_session_{start_session}_{end_session}.json"


def get_retrieval_log_dir(
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
    results_suffix: str | None = None,
) -> Path:
    tag = _suffix_tag(results_suffix)
    return get_session_dir(model_path, benchmark_size, start_session, end_session, config_name) / f"retrieval_logs{tag}"


def get_memory_snapshots_dir(
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    return get_session_dir(model_path, benchmark_size, start_session, end_session, config_name) / "memory_snapshots"


def get_prompt_log_dir(
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
    results_suffix: str | None = None,
) -> Path:
    tag = _suffix_tag(results_suffix)
    return get_session_dir(model_path, benchmark_size, start_session, end_session, config_name) / f"prompt_log{tag}"


def get_merged_results_file(
    model_path: str,
    benchmark_size: str,
    config_name: str = "config_0",
) -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, benchmark_size, config_name) / f"results_{model_name}_{benchmark_size}_merged.json"


def ensure_directories(
    model_path: str,
    benchmark_size: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
    results_suffix: str | None = None,
) -> None:
    get_session_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, benchmark_size, start_session, end_session, config_name, results_suffix).mkdir(parents=True, exist_ok=True)
    get_memory_snapshots_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, benchmark_size, start_session, end_session, config_name, results_suffix).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_llm_config(engine: str = None) -> dict:
    engine = engine or LLM_ENGINE
    if engine == "vllm":
        return {"engine": "vllm", **DEFAULT_VLLM_CONFIG}
    if engine == "together":
        return {"engine": "together", **TOGETHER_CONFIG}
    if engine == "openai":
        return {"engine": "openai", **OPENAI_CONFIG}
    raise ValueError(f"Unknown LLM engine: {engine}")
