"""
Configuration for Dense Retrieval Experiment on PrefEval (implicit-persona).

Adapted from exp_implexconv_no_response/dense/config_0.py.
Differences vs ImplexConv config:
- Single dataset path (no opposed/supportive split).
- MAX_TOKENS bumped from 750 to 1500 (PrefEval answers are advisory).
- JSON_RETRY raised from 5 to 10.
- Output dir keyed by model only (no subset, no session range).
- Path helpers take no subset / session-range args.
- BATCH_SIZE (cross-session) removed; chain count is always 1.
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

DATASET_DIR  = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DIR / "implicit_persona.json"

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

TEMPERATURE = 0.7
MAX_TOKENS  = 1500
JSON_RETRY  = 10


# =============================================================================
# BATCH SETTINGS
# =============================================================================

# Maximum number of QA prompts to send to vLLM in one generate_batch_raw() call.
QA_BATCH_SIZE = 64

# Encoding batch size for SentenceTransformer.encode (internal). Optional knob.
ENCODE_BATCH_SIZE = 32


# =============================================================================
# MEMORY MODULE SETTINGS  (Dense retrieval specific)
# =============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVE_K      = 15


# =============================================================================
# TIME MODEL  (kept for cross-module consistency; not used in dense retrieval)
# =============================================================================

CONV_IDS_PER_DAY = 2
MINUTES_PER_TURN = 10


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

SAVE_MEMORY_SNAPSHOTS = True


# =============================================================================
# LLM CALL LOGGING
# =============================================================================

ENABLE_LLM_CALL_LOGGING = True


# =============================================================================
# PATH HELPERS
# =============================================================================

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_output_dir(model_path: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}"


def get_results_file(model_path: str, config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / "results.jsonl"


def get_retrieval_log_file(model_path: str, config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / "retrieval_log.jsonl"


def get_stats_file(model_path: str, config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / "stats.json"


def get_meta_file(model_path: str, config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / "meta.json"


def get_memory_snapshots_dir(model_path: str, config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / "memory_snapshots"


def get_prompt_log_dir(model_path: str, config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / "prompt_log"


def get_run_log_dir(model_path: str, config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / "logs"


def ensure_directories(model_path: str, config_name: str = "config"):
    get_output_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
    get_memory_snapshots_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
    get_run_log_dir(model_path, config_name).mkdir(parents=True, exist_ok=True)
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
