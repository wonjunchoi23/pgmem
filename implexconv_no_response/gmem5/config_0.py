"""
Configuration for GraphMem v5 on ImplexConv.
"""

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

DATASET_DIR = PROJECT_ROOT / "dataset" / "implexconv"
DATASET_OPPOSED = DATASET_DIR / "ImplexConv_opposed_processed.json"
DATASET_SUPPORTIVE = DATASET_DIR / "ImplexConv_supportive_processed.json"

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
# (typically 3-5× the empirically expected output size) so that retries with
# slightly larger outputs do not get truncated, while still being well below
# the previous shared cap of 5000.
MAX_TOKENS_STATE              = 1000  # ②   1 state object
MAX_TOKENS_STATE_NEW_REL      = 800   # ②b  ≤ C(N,2)+N×prev judgments (N=STATE_MAX_COUNT=1)
MAX_TOKENS_MEMORY             = 1000  # ③   1 memory object
MAX_TOKENS_MEMORY_NEW_REL     = 2000  # ③b  prev_memory + chunk_states judgments
MAX_TOKENS_TRAIT              = 1200  # ④   0 or 1 trait object
MAX_TOKENS_TRAIT_EVIDENCE_5A  = 2500  # ⑤a  recent_states + recent_memories + prev_trait judgments
MAX_TOKENS_TRAIT_EXTRA_REL_5B = 1500  # ⑤b  ≤ TOPK_STATE+TOPK_MEMORY judgments
MAX_TOKENS_STATE_STATE_5C     = 1000  # ⑤c  ≤ STATE_STATE_EXTRA_REL_TOPK judgments
MAX_TOKENS_STATE_MEMORY_5D    = 800   # ⑤d  ≤ STATE_MEMORY_EXTRA_REL_TOPK judgments

JSON_RETRY = 5
JUDGMENT_RETRY = 3       # retry count when judgments array is empty but expected_judgment_count > 0


# =============================================================================
# BATCH SETTINGS
# =============================================================================

BATCH_SIZE = 4
QA_BATCH_SIZE = 128


# =============================================================================
# GRAPHMEM V3 SETTINGS
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
MAX_KEYWORDS = 5
MAX_DOMAIN_LABELS = 5
MIN_DOMAIN_LABELS = 3

# Seed retrieval top-k
K_CONTEXT = 20
K_MEMORY = 6
K_MEMORY_FINAL = 6       # final memory count after evidence-based scoring of pooled memories
K_STATE = 17
K_TRAIT = 6
K_APS = 6

# Seed retrieval weights (context uses fixed pair; states / memories / traits share scope-dependent pair)
W_SEM_C = 0.65
W_OV_C = 0.35

W_SEM_NARROW = 0.60
W_OV_NARROW = 0.40
W_SEM_BROAD = 0.85
W_OV_BROAD = 0.15

# Pair similarity (used in ⑤b reranking and ⑤c/⑤d reservoir maintenance)
W_PAIR_SEM = 0.7
W_PAIR_LEX = 0.3

# Final set
K_T_FINAL = 4            # number of traits kept in the final set
K_SF = 15
TRAIT_VALIDATION_TAU = 0.7
W_SR = 0.5               # support-ratio weight in unified final scoring; seed-score weight is (1 - W_SR)

# Graph traversal
SIGN_PROP_HOP_CAP = 10

# APS and shift control
APS_EXCLUDE_SHIFT_SOURCE = True
ENABLE_SHIFT_CHAIN_PRUNING = True
STRICT_HIGH_DEFAULT_LOW = True

# Additional relation extraction (⑤b/⑤c/⑤d)
ENABLE_EXTRA_RELATION_EXTRACTION = True
EXTRA_REL_ONLY_IF_UNCONNECTED = True
TRAIT_EXTRA_REL_TOPK_STATE = 7
TRAIT_EXTRA_REL_TOPK_MEMORY = 2
STATE_STATE_EXTRA_REL_TOPK = 5
STATE_MEMORY_EXTRA_REL_TOPK = 2
STATE_NEW_REL_PREV_WINDOW = 3    # global recent prev-state window for ②b

# Context cache
CONTEXT_CACHE_SIZE = 10

# Time model
TIME_PER_CONV_ID_HOURS = 12
TIME_PER_TURN_MINUTES = 10
CONV_IDS_PER_DAY = 2
MINUTES_PER_TURN = 10

# QA / Response serialization
INCLUDE_RECENT_CONVERSATION_FOR_QA = True
QA_CONTEXT_PAIRS = 5     # number of most-recent (user, agent) pairs included in QA prompt

# Experiment
ENABLE_CHECKPOINTING = True
CHECKPOINT_INTERVAL = 1
TIMING_CONV_ID = 0
SAVE_MEMORY_SNAPSHOTS = True
ENABLE_LLM_CALL_LOGGING = True
# When ENABLE_LLM_CALL_LOGGING is True, only sessions with session_id < this
# value have their per-call prompts/outputs persisted under prompt_log/.
LLM_CALL_LOG_FIRST_N_SESSIONS = 10


# =============================================================================
# PATH HELPERS
# =============================================================================

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_dataset_path(subset: str) -> Path:
    if subset == "opposed":
        return DATASET_OPPOSED
    if subset == "supportive":
        return DATASET_SUPPORTIVE
    raise ValueError(f"Unknown subset '{subset}'. Expected 'opposed' or 'supportive'.")


def get_output_dir(model_path: str, subset: str, config_name: str = "config_0") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}_{subset}"


def get_session_dir(
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    return get_output_dir(model_path, subset, config_name) / f"session_{start_session}_{end_session}"


def get_results_file(
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    model_name = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"results_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_checkpoint_file(
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    model_name = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"checkpoint_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_retrieval_log_dir(
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "retrieval_logs"


def get_memory_snapshots_dir(
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "memory_snapshots"


def get_prompt_log_dir(
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "prompt_log"


def get_merged_results_file(
    model_path: str,
    subset: str,
    config_name: str = "config_0",
) -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, subset, config_name) / f"results_{model_name}_{subset}_merged.json"


def ensure_directories(
    model_path: str,
    subset: str,
    start_session: int,
    end_session: int,
    config_name: str = "config_0",
) -> None:
    get_session_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_memory_snapshots_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
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
