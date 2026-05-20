"""
Configuration for A-MEM Batch Experiment on PersonaMem
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
LOG_TO_FILE     = True
LOG_LEVEL       = "INFO"


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
MAX_TOKENS  = 750
JSON_RETRY  = 5


# =============================================================================
# BATCH SETTINGS
# =============================================================================

BATCH_SIZE    = 4
QA_BATCH_SIZE = 64


# =============================================================================
# MEMORY MODULE SETTINGS  (A-MEM specific)
# =============================================================================

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
RETRIEVE_K         = 5
EVOLUTION_THRESHOLD = 100


# =============================================================================
# TIME MODEL
# =============================================================================

CONV_IDS_PER_DAY = 1   # 1 block = 1 virtual day (PersonaMem has ~5 blocks/context)
MINUTES_PER_TURN = 10


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

ENABLE_CHECKPOINTING  = True
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
    get_memory_snapshots_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, benchmark_size, start_session, end_session, config_name).mkdir(parents=True, exist_ok=True)
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
