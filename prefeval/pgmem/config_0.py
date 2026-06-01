import os
import sys
from pathlib import Path

# =============================================================================
# PATH SETTINGS
# =============================================================================

CONFIG_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CONFIG_DIR.parent

LLM_MODULE_DIR = PROJECT_ROOT.parent / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))

DATASET_DIR  = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DIR / "implicit_persona.json"

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
MAX_TOKENS = 1500

# Per-call output caps for internal graph LLM calls (PrefEval-agnostic, carried over).
MAX_TOKENS_STATE              = 1000
MAX_TOKENS_STATE_NEW_REL      = 800
MAX_TOKENS_EPISODE            = 1000
MAX_TOKENS_EPISODE_NEW_REL    = 2000
MAX_TOKENS_TRAIT              = 1200
MAX_TOKENS_TRAIT_EVIDENCE_5A  = 2500
MAX_TOKENS_TRAIT_EXTRA_REL_5B = 1500
MAX_TOKENS_STATE_STATE_5C     = 1000
MAX_TOKENS_STATE_EPISODE_5D   = 800

JSON_RETRY = 10
JUDGMENT_RETRY = 3


# =============================================================================
# BATCH SETTINGS
# =============================================================================

# At checkpoint k, batch all q_0..q_k together in chunks of this size.
QA_BATCH_SIZE = 64


# =============================================================================
# PGMEM V6 SETTINGS
# =============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Extraction
STATE_EXTRACTION_H = 1
STATE_MAX_COUNT = 1
STATE_REF_CONTEXT_TURNS = 3
CHUNK_SIZE_CONV = 1
TRAIT_EXTRACTION_CHUNKS = 2
TRAIT_MAX_COUNT = 1
MAX_KEYWORDS = 7
MAX_DOMAIN_LABELS = 7
MIN_DOMAIN_LABELS = 5

# Seed retrieval top-k (aligned with implexconv config_0)
K_CONTEXT = 20
K_EPISODE = 6
K_EPISODE_FINAL = 4
K_STATE = 17
K_TRAIT = 6
K_APS = 6

# Seed retrieval weights
W_SEM_C = 0.65
W_OV_C = 0.35

W_SEM_NARROW = 0.60
W_OV_NARROW = 0.40
W_SEM_BROAD = 0.85
W_OV_BROAD = 0.15

# Final set
K_T_FINAL = 5
K_SF = 18
TRAIT_VALIDATION_TAU = 0.7
W_SR = 0.5

# Graph traversal
SIGN_PROP_HOP_CAP = 10

# Seed turn-neighbor expansion
SEED_TURN_NEIGHBOR_ENABLED = False
SEED_TURN_NEIGHBOR_DELTA = 1

# APS and shift control
APS_EXCLUDE_SHIFT_SOURCE = True
ENABLE_SHIFT_CHAIN_PRUNING = True
STRICT_HIGH_DEFAULT_LOW = True

# Additional relation extraction (⑤b/⑤c/⑤d)
ENABLE_EXTRA_RELATION_EXTRACTION = True
EXTRA_REL_ONLY_IF_UNCONNECTED = True
TRAIT_EXTRA_REL_TOPK_STATE = 10
TRAIT_EXTRA_REL_TOPK_EPISODE = 5
STATE_STATE_EXTRA_REL_TOPK = 8
STATE_EPISODE_EXTRA_REL_TOPK = 4
STATE_NEW_REL_PREV_WINDOW = 5

# Context cache
CONTEXT_CACHE_SIZE = 10

# Sub-block chunking — fixed scalar for PrefEval (PersonaMem used a per-benchmark dict).
# 1 = episode extracted once per session; trait every TRAIT_EXTRACTION_CHUNKS sessions.
CHUNK_FACTOR = 1

# Time model — PrefEval spec §3 (matches ImplexConv).
# CONV_IDS_PER_DAY=2 ⇒ TIME_PER_CONV_ID_HOURS = 24/2 = 12.
CONV_IDS_PER_DAY = 2
TIME_PER_CONV_ID_HOURS = 12
TIME_PER_TURN_MINUTES = 10

# QA prompt cache inclusion (kept False to match PersonaMem; chained PrefEval
# sessions are unrelated, so the most-recent conversation is noise).
INCLUDE_RECENT_CONVERSATION_FOR_QA = False
QA_CONTEXT_PAIRS = 5

# Experiment
SAVE_MEMORY_SNAPSHOTS = True
ENABLE_LLM_CALL_LOGGING = True


# =============================================================================
# PATH HELPERS
# =============================================================================

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_output_dir(model_path: str, config_name: str = "config_8") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}"


def get_results_file(model_path: str, config_name: str = "config_8") -> Path:
    return get_output_dir(model_path, config_name) / "results.jsonl"


def get_retrieval_log_file(model_path: str, config_name: str = "config_8") -> Path:
    return get_output_dir(model_path, config_name) / "retrieval_log.jsonl"


def get_stats_file(model_path: str, config_name: str = "config_8") -> Path:
    return get_output_dir(model_path, config_name) / "stats.json"


def get_meta_file(model_path: str, config_name: str = "config_8") -> Path:
    return get_output_dir(model_path, config_name) / "meta.json"


def get_memory_snapshots_dir(model_path: str, config_name: str = "config_8") -> Path:
    return get_output_dir(model_path, config_name) / "memory_snapshots"


def get_prompt_log_dir(model_path: str, config_name: str = "config_8") -> Path:
    return get_output_dir(model_path, config_name) / "prompt_log"


def get_run_log_dir(model_path: str, config_name: str = "config_8") -> Path:
    return get_output_dir(model_path, config_name) / "logs"


def ensure_directories(model_path: str, config_name: str = "config_8") -> None:
    get_output_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
    get_memory_snapshots_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
    get_run_log_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LLM CONFIG HELPER
# =============================================================================

def get_llm_config(engine: str = None) -> dict:
    engine = engine or LLM_ENGINE
    if engine == "vllm":
        return {"engine": "vllm", **DEFAULT_VLLM_CONFIG}
    if engine == "together":
        return {"engine": "together", **TOGETHER_CONFIG}
    if engine == "openai":
        return {"engine": "openai", **OPENAI_CONFIG}
    raise ValueError(f"Unknown LLM engine: {engine}")
