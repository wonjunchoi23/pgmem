"""
Configuration for Theanine Batch Experiment on ImplexConv

Theanine-specific hyperparameters (from original Theanine paper):
  - LINKING_TOP_J     = 3   (j: top-j similar past nodes for relation extraction)
  - RETRIEVE_TOP_K    = 5   (k: top-k nodes retrieved for QA timeline retrieval)
  - TIMELINE_SAMPLE_N = 1   (n: number of timeline paths sampled per retrieved node)
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

CONFIG_DIR   = Path(__file__).parent.absolute()
PROJECT_ROOT = CONFIG_DIR.parent

LLM_MODULE_DIR = PROJECT_ROOT / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))

DATASET_DIR        = PROJECT_ROOT / "dataset" / "implexconv"
DATASET_OPPOSED    = DATASET_DIR / "ImplexConv_opposed_processed.json"
DATASET_SUPPORTIVE = DATASET_DIR / "ImplexConv_supportive_processed.json"

BASE_OUTPUT_DIR = CONFIG_DIR
LOG_DIR         = CONFIG_DIR / "logs"
LOG_TO_FILE     = True
LOG_LEVEL       = "INFO"


# =============================================================================
# LLM SETTINGS
# =============================================================================

LLM_ENGINE = "vllm"

DEFAULT_VLLM_CONFIG = {
    "model_path":              "meta-llama/Llama-3.1-8B-Instruct",
    "tensor_parallel_size":    1,
    "gpu_memory_utilization":  0.5,
    "download_dir":            None,
}

TOGETHER_CONFIG = {
    "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "api_key":    os.getenv("TOGETHER_API_KEY"),
}

OPENAI_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "api_key":    os.getenv("OPENAI_API_KEY"),
}

TEMPERATURE          = 0.7
MAX_TOKENS           = 2048  # relation extraction, refinement, and QA generation
SUMMARIZE_MAX_TOKENS = 1500  # conv summarization (long dialogues need more room)
JSON_RETRY           = 3


# =============================================================================
# BATCH SETTINGS
# =============================================================================

# Number of sessions to process in parallel within one GPU.
BATCH_SIZE = 4

# Maximum number of prompts to send to vLLM in one generate_batch_raw() call.
SUMMARIZE_BATCH_SIZE = 32
RELATION_BATCH_SIZE = 64
REFINE_BATCH_SIZE = 64
QA_BATCH_SIZE = 64


# =============================================================================
# MEMORY MODULE SETTINGS  (Theanine-specific)
# =============================================================================

# Sentence embedding model for memory retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# j: top-j similar past nodes retrieved during relation-extraction linking
# (original Theanine Retriever top_k=3 used for both linking and retrieval)
LINKING_TOP_J = 3

# k: top-k memory nodes retrieved for QA timeline retrieval
RETRIEVE_TOP_K = 5

# n: max number of unique timeline paths sampled per retrieved node
TIMELINE_SAMPLE_N = 1


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

ENABLE_CHECKPOINTING  = True
TIMING_CONV_ID        = 0       # Phase 1 timing: only conv_id == 0
SAVE_MEMORY_SNAPSHOTS = True

# =============================================================================
# TIME MODEL  (theanine-specific)
# =============================================================================

# Virtual time model for ImplexConv sessions:
#   - CONV_IDS_PER_DAY consecutive conv_ids = 1 day
#   - local turn_id (within a conv) × MINUTES_PER_TURN = elapsed minutes within the day
#
# Consequences:
#   - current_dialogue resets at each day boundary (every CONV_IDS_PER_DAY conv_ids)
#   - FINALIZE_EVERY_N_CONVS is always kept equal to CONV_IDS_PER_DAY
#     so memory finalization also happens once per day
CONV_IDS_PER_DAY   = 2    # how many conv_ids constitute one virtual day
MINUTES_PER_TURN   = 10   # minutes per local turn_id within a conv

# Always equal to CONV_IDS_PER_DAY — kept as a named constant for clarity.
FINALIZE_EVERY_N_CONVS = CONV_IDS_PER_DAY

# =============================================================================
# LLM CALL LOGGING
# =============================================================================

ENABLE_LLM_CALL_LOGGING = True


# =============================================================================
# PATH HELPERS
# =============================================================================

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_output_dir(model_path: str, subset: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}_{subset}"


def get_session_dir(model_path: str, subset: str,
                    start_session: int, end_session: int,
                    config_name: str = "config") -> Path:
    return get_output_dir(model_path, subset, config_name) / f"session_{start_session}_{end_session}"


def get_results_file(model_path: str, subset: str,
                     start_session: int, end_session: int,
                     config_name: str = "config") -> Path:
    model_name  = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"results_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_checkpoint_file(model_path: str, subset: str,
                        start_session: int, end_session: int,
                        config_name: str = "config") -> Path:
    model_name  = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"checkpoint_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_retrieval_log_dir(model_path: str, subset: str,
                          start_session: int, end_session: int,
                          config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "retrieval_logs"


def get_memory_snapshots_dir(model_path: str, subset: str,
                              start_session: int, end_session: int,
                              config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "memory_snapshots"


def get_prompt_log_dir(model_path: str, subset: str,
                       start_session: int, end_session: int,
                       config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "prompt_log"


def get_merged_results_file(model_path: str, subset: str,
                             config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, subset, config_name) / f"results_{model_name}_{subset}_merged.json"


def ensure_directories(model_path: str, subset: str,
                        start_session: int, end_session: int,
                        config_name: str = "config"):
    get_session_dir(
        model_path, subset, start_session, end_session, config_name
    ).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(
        model_path, subset, start_session, end_session, config_name
    ).mkdir(parents=True, exist_ok=True)
    get_memory_snapshots_dir(
        model_path, subset, start_session, end_session, config_name
    ).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(
        model_path, subset, start_session, end_session, config_name
    ).mkdir(parents=True, exist_ok=True)
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
