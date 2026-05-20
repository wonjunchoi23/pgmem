"""
Configuration for MemoryBank Experiment on ImplexConv
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
PROJECT_ROOT = CONFIG_DIR.parent.parent

LLM_MODULE_DIR = PROJECT_ROOT / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))

DATASET_DIR = PROJECT_ROOT / "dataset" / "implexconv"
DATASET_OPPOSED    = DATASET_DIR / "ImplexConv_opposed_processed.json"
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
    "model_name": "gpt-3.5-turbo",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

TEMPERATURE = 0.7
MAX_TOKENS = 750
JSON_RETRY = 3


# =============================================================================
# MEMORY MODULE SETTINGS  (MemoryBank specific)
# =============================================================================

# Sentence embedding model for FAISS retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of memories to retrieve per query (after forgetting re-rank)
# Aligned with original MemoryBank paper (VECTOR_SEARCH_TOP_K = 6).
RETRIEVE_K = 6

# Ebbinghaus forgetting curve: retention = exp(-(conv_gap / CONVS_PER_DAY) / (FORGETTING_DIVISOR * S))
# Original paper uses divisor=5.
FORGETTING_DIVISOR = 5

# How many conv_ids equal one "day".
# Used for: (1) forgetting curve day conversion, (2) daily summary trigger interval.
# e.g. CONVS_PER_DAY=2 → daily summary is generated every 2 conv_ids (batched together).
CONVS_PER_DAY = 2

# Run global summary synthesis every N conv_ids.
# Must be a multiple of CONVS_PER_DAY (e.g. 10 = 5 days worth of conv_ids).
# Phase 1 end always triggers a final global synthesis unconditionally.
GLOBAL_SUMMARY_INTERVAL = 10

# Virtual time model — must match global_readme.md spec across all modules.
# 1 virtual day = CONVS_PER_DAY consecutive conv_ids.
# 1 turn = MINUTES_PER_TURN virtual minutes (local turn_id within a conv).
# MINUTES_PER_TURN is declared for consistency; not directly used in MemoryBank logic.
MINUTES_PER_TURN = 10

# Number of conv_ids to look back for history (1 "day" = CONVS_PER_DAY conv_ids).
# All turn pairs within the last HISTORY_CONV_WINDOW conv_ids are included,
# both in response generation and in QA answering.
HISTORY_CONV_WINDOW = 2

# Summarization LLM settings (separate from response generation)
SUMMARIZE_TEMPERATURE = 0.7
SUMMARIZE_MAX_TOKENS = 400


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

ENABLE_CHECKPOINTING = True
TIMING_CONV_ID = 0          # Unused in QA-only variant (no Phase 1 LLM call)
SAVE_MEMORY_SNAPSHOTS = True
ENABLE_LLM_CALL_LOGGING = True


# =============================================================================
# PATH HELPERS
# =============================================================================

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_output_dir(model_path: str, subset: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}_{subset}"


def get_session_dir(model_path: str, subset: str, start_session: int, end_session: int,
                    config_name: str = "config") -> Path:
    return get_output_dir(model_path, subset, config_name) / f"session_{start_session}_{end_session}"


def get_results_file(model_path: str, subset: str, start_session: int, end_session: int,
                     config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"results_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_checkpoint_file(model_path: str, subset: str, start_session: int, end_session: int,
                        config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"checkpoint_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_retrieval_log_dir(model_path: str, subset: str, start_session: int, end_session: int,
                          config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "retrieval_logs"


def get_memory_snapshots_dir(model_path: str, subset: str, start_session: int, end_session: int,
                              config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "memory_snapshots"


def get_prompt_log_dir(model_path: str, subset: str, start_session: int, end_session: int,
                       config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "prompt_log"


def get_merged_results_file(model_path: str, subset: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, subset, config_name) / f"results_{model_name}_{subset}_merged.json"


def ensure_directories(model_path: str, subset: str, start_session: int, end_session: int,
                       config_name: str = "config"):
    get_session_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_memory_snapshots_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LLM CONFIG HELPER
# =============================================================================

def get_llm_config(engine: str = None) -> dict:
    if engine is None:
        engine = LLM_ENGINE
    if engine == "vllm":
        return {"engine": "vllm", **DEFAULT_VLLM_CONFIG}
    elif engine == "together":
        return {"engine": "together", **TOGETHER_CONFIG}
    elif engine == "openai":
        return {"engine": "openai", **OPENAI_CONFIG}
    raise ValueError(f"Unknown LLM engine: {engine}")
