"""
Configuration for LD-Agent Experiment on PrefEval (implicit-persona).

Adapted from exp_implexconv_no_response/ldagent/config_0.py.
Differences vs ImplexConv config:
- Single dataset path (no opposed/supportive split).
- MAX_TOKENS bumped from 750 to 1500 (PrefEval answers are advisory).
- JSON_RETRY raised from 3 to 10.
- Output dir keyed by model only (no subset, no session range).
- Path helpers take no subset / session-range args.
- BATCH_SIZE (cross-session) removed; chain count is always 1.
"""

import os
import sys
from pathlib import Path

# =============================================================================
# 1. ENVIRONMENT & PATHS
# =============================================================================

os.environ["HF_TOKEN"] = "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"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

CONFIG_DIR   = Path(__file__).parent.absolute()
PROJECT_ROOT = CONFIG_DIR.parent

LLM_MODULE_DIR = PROJECT_ROOT / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))


# =============================================================================
# 2. DATASET
# =============================================================================

DATASET_DIR  = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DIR / "implicit_persona.json"

BASE_OUTPUT_DIR = CONFIG_DIR
LOG_DIR         = CONFIG_DIR / "logs"


# =============================================================================
# 3. DIALOGUE ROLES
# =============================================================================

USR_NAME   = "User"
AGENT_NAME = "Agent"


# =============================================================================
# 4. MEMORY  (EventMemory)
# =============================================================================

RELEVANCE_MEMORY_NUMBER = 3
RETRIEVE_K              = 3
DIST_THRESHOLD          = 1.5
ORI_MEM_QUERY           = False


# =============================================================================
# 4a. VIRTUAL TIME MODEL
# =============================================================================

CONV_IDS_PER_DAY       = 2
MINUTES_PER_TURN       = 10
FINALIZE_EVERY_N_CONVS = CONV_IDS_PER_DAY
DECAY_TEMP             = 1e-4


# =============================================================================
# 5. PERSONA  (Personas)
# =============================================================================

MAX_USER_PERSONAS  = 10
MAX_AGENT_PERSONAS = 10


# =============================================================================
# 6. GENERATION  (Generator)
# =============================================================================

MAX_TOKENS  = 1500
TEMPERATURE = 0.7
JSON_RETRY  = 10


# =============================================================================
# 7. EXPERIMENT SETTINGS
# =============================================================================

QA_BATCH_SIZE = 64

SAVE_MEMORY_SNAPSHOTS   = True
ENABLE_LLM_CALL_LOGGING = True


# =============================================================================
# 8. LLM ENGINE
# =============================================================================

LLM_ENGINE = "vllm"

DEFAULT_VLLM_CONFIG = {
    "model_path":             "meta-llama/Llama-3.1-8B-Instruct",
    "tensor_parallel_size":   1,
    "gpu_memory_utilization": 0.5,
    "download_dir":           None,
}

OPENAI_CONFIG = {
    "model_name": "gpt-4o-mini",
    "api_key":    os.getenv("OPENAI_API_KEY"),
}

TOGETHER_CONFIG = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    "api_key":    os.getenv("TOGETHER_API_KEY"),
}


# =============================================================================
# 9. LOGGING
# =============================================================================

LOG_LEVEL   = "INFO"
LOG_TO_FILE = True


# =============================================================================
# 10. PATH HELPERS
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
