"""
Configuration for LD-Agent experiment on PersonaMem
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

DATASET_DIR = PROJECT_ROOT / "dataset"

DATASET_QUESTIONS_32K  = DATASET_DIR / "questions_32k.csv"
DATASET_CONTEXTS_32K   = DATASET_DIR / "shared_contexts_32k.jsonl"
DATASET_QUESTIONS_128K = DATASET_DIR / "questions_128k.csv"
DATASET_CONTEXTS_128K  = DATASET_DIR / "shared_contexts_128k.jsonl"
DATASET_QUESTIONS_1M   = DATASET_DIR / "questions_1M.csv"
DATASET_CONTEXTS_1M    = DATASET_DIR / "shared_contexts_1M.jsonl"

BENCHMARK_SIZES = ["32k", "128k", "1M"]

BASE_OUTPUT_DIR = CONFIG_DIR
LOG_DIR         = CONFIG_DIR / "logs"

# =============================================================================
# 3. DIALOGUE ROLES
# =============================================================================

USR_NAME   = "User"
AGENT_NAME = "Assistant"   # PersonaMem prefix uses "Assistant: ..."

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

# 1 PersonaMem block = 1 virtual day (matches amem / memorybank / theanine).
# block_idx is fed into the conv_id slot of compute_virtual_seconds.
CONV_IDS_PER_DAY       = 1
MINUTES_PER_TURN       = 10
FINALIZE_EVERY_N_CONVS = CONV_IDS_PER_DAY
DECAY_TEMP             = 1e-4

# =============================================================================
# 5. PERSONA  (Personas)
# =============================================================================

MAX_USER_PERSONAS  = 10
MAX_AGENT_PERSONAS = 10

# =============================================================================
# 6. LLM ENGINE
# =============================================================================

LLM_ENGINE = "vllm"

DEFAULT_VLLM_CONFIG = {
    "model_path":             "meta-llama/Llama-3.1-8B-Instruct",
    "tensor_parallel_size":   1,
    "gpu_memory_utilization": 0.5,
    "download_dir":           None,
    "max_model_len":          10_000,
}

# =============================================================================
# 7. GENERATION  (Generator)
# =============================================================================

MAX_TOKENS  = 750
TEMPERATURE = 0.7
JSON_RETRY  = 5

# Input-side token budget for STM flush LLM calls.
# Derived from DEFAULT_VLLM_CONFIG["max_model_len"]; overridden at runtime via
# --max-model-len CLI arg (see run_experiment.py).
FINALIZE_INPUT_CONTEXT_LIMIT = DEFAULT_VLLM_CONFIG["max_model_len"]
FINALIZE_CONTEXT_UTILIZATION = 0.85

# Number of sub-blocks to split each block into for STM flush granularity.
CHUNK_FACTOR = {"32k": 1, "128k": 1, "1M": 2}

# =============================================================================
# 8. EXPERIMENT SETTINGS
# =============================================================================

BATCH_SIZE    = 4
QA_BATCH_SIZE = 32

ENABLE_CHECKPOINTING    = True
SAVE_MEMORY_SNAPSHOTS   = True
ENABLE_LLM_CALL_LOGGING = True

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
# 10. OUTPUT PATH HELPERS
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


def get_output_dir(model_path: str, benchmark_size: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}_{benchmark_size}"


def get_session_dir(model_path: str, benchmark_size: str,
                    start_session: int, end_session: int,
                    config_name: str = "config") -> Path:
    return get_output_dir(model_path, benchmark_size, config_name) / f"session_{start_session}_{end_session}"


def get_results_file(model_path: str, benchmark_size: str,
                     start_session: int, end_session: int,
                     config_name: str = "config") -> Path:
    model_name  = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, benchmark_size, start_session, end_session, config_name)
    return session_dir / f"results_{model_name}_{benchmark_size}_session_{start_session}_{end_session}.json"


def get_checkpoint_file(model_path: str, benchmark_size: str,
                        start_session: int, end_session: int,
                        config_name: str = "config") -> Path:
    model_name  = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, benchmark_size, start_session, end_session, config_name)
    return session_dir / f"checkpoint_{model_name}_{benchmark_size}_session_{start_session}_{end_session}.json"


def get_retrieval_log_dir(model_path: str, benchmark_size: str,
                          start_session: int, end_session: int,
                          config_name: str = "config") -> Path:
    return get_session_dir(model_path, benchmark_size, start_session, end_session, config_name) / "retrieval_logs"


def get_memory_snapshots_dir(model_path: str, benchmark_size: str,
                              start_session: int, end_session: int,
                              config_name: str = "config") -> Path:
    return get_session_dir(model_path, benchmark_size, start_session, end_session, config_name) / "memory_snapshots"


def get_prompt_log_dir(model_path: str, benchmark_size: str,
                       start_session: int, end_session: int,
                       config_name: str = "config") -> Path:
    return get_session_dir(model_path, benchmark_size, start_session, end_session, config_name) / "prompt_log"


def get_merged_results_file(model_path: str, benchmark_size: str,
                             config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, benchmark_size, config_name) / f"results_{model_name}_{benchmark_size}_merged.json"


def ensure_directories(model_path: str, benchmark_size: str,
                        start_session: int, end_session: int,
                        config_name: str = "config"):
    get_session_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    if SAVE_MEMORY_SNAPSHOTS:
        get_memory_snapshots_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
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
