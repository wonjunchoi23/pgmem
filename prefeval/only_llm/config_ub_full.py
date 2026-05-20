"""
Configuration for OnlyLLM (Upper Bound) Experiment on PrefEval (implicit-persona).

OnlyLLM has no memory module; the LLM sees the raw dialogue history as context.
This `ub` config feeds the full chain history to the model, trimmed only to fit
within `--max-model-len` by `_select_suffix_with_budget`.

Differences vs amem/config_0.py:
- No EMBEDDING_MODEL / RETRIEVE_K / EVOLUTION_THRESHOLD (no memory module).
- Adds OnlyLLM-specific context controls: HISTORY_ALL_GIVEN, MAX_CONTEXT_TURNS,
  OUTPUT_TOKEN_RESERVE, CONTEXT_SAFETY_MARGIN.
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
    "model_name": "gpt-3.5-turbo",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

TEMPERATURE = 0.7
MAX_TOKENS  = 1500
JSON_RETRY  = 10


# =============================================================================
# CONTEXT SETTINGS  (OnlyLLM specific)
# =============================================================================

# all_given: keep the longest suffix of turns that fits within the token budget.
# window:    keep only the last MAX_CONTEXT_TURNS turns regardless of model len.
HISTORY_ALL_GIVEN     = True
MAX_CONTEXT_TURNS     = 20      # fallback / sanity cap, unused when HISTORY_ALL_GIVEN=True
OUTPUT_TOKEN_RESERVE  = 600
CONTEXT_SAFETY_MARGIN = 200


# =============================================================================
# BATCH SETTINGS
# =============================================================================

QA_BATCH_SIZE = 64


# =============================================================================
# TIME MODEL
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
