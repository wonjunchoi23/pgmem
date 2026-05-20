"""
Configuration for OnlyLLM Batch Experiment (QA-Only Variant)

This module unifies the former lbllm (Lower Bound) and ubllm (Unbounded) baselines
into a single experiment controlled by HISTORY_ALL_GIVEN:

  HISTORY_ALL_GIVEN = True   →  ubllm behaviour:
      All accumulated turns are provided as context, trimmed from the oldest end
      to fit within the model's token budget when the context limit is exceeded.

  HISTORY_ALL_GIVEN = False  →  lbllm behaviour:
      Only the last MAX_CONTEXT_TURNS turns are provided (sliding window).
      MAX_CONTEXT_TURNS is the k hyperparameter.

Experiment protocol: QA-Only Variant (global_readme.md)
  - Phase 1 builds dialogue context only; no response-generation prompt is constructed.
  - Phase 2 performs QA answering with a real LLM call.
  - Output contains QA results and LLM-call statistics only.
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
MAX_TOKENS = 500
JSON_RETRY = 5


# =============================================================================
# BATCH SETTINGS
# =============================================================================

BATCH_SIZE = 4
QA_BATCH_SIZE = 64


# =============================================================================
# CONTEXT SETTINGS
# =============================================================================

# ── Core switch ──────────────────────────────────────────────────────────────
# True  → provide ALL accumulated turns (token-budget trimmed)
# False → provide only the last MAX_CONTEXT_TURNS turns (sliding window)
HISTORY_ALL_GIVEN = True

# Used only when HISTORY_ALL_GIVEN = False
# Number of recent utterances kept in the sliding window (each "turn" = one utterance;
# e.g. 20 utterances ≈ 10 user–assistant exchanges)
MAX_CONTEXT_TURNS = 20

# Used only when HISTORY_ALL_GIVEN = True
# Tokens reserved for model output (should be >= MAX_TOKENS with some slack)
OUTPUT_TOKEN_RESERVE = 600
# Extra buffer for fixed prompt parts whose token count is only estimated
CONTEXT_SAFETY_MARGIN = 200


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

ENABLE_CHECKPOINTING = True
TIMING_CONV_ID = 0          # kept for compatibility; not used in QA-only variant
SAVE_MEMORY_SNAPSHOTS = False
ENABLE_LLM_CALL_LOGGING = True

# Virtual time model (must match global_readme.md)
CONV_IDS_PER_DAY = 2        # Number of consecutive conv_ids per virtual day
MINUTES_PER_TURN = 10       # Minutes per local turn_id within a conv


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


def get_llm_call_log_dir(model_path: str, subset: str, start_session: int, end_session: int,
                         config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "llm_call_log"


def get_merged_results_file(model_path: str, subset: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, subset, config_name) / f"results_{model_name}_{subset}_merged.json"


def ensure_directories(model_path: str, subset: str, start_session: int, end_session: int,
                       config_name: str = "config"):
    get_session_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_llm_call_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
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
