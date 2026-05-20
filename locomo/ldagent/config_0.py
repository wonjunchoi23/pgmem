"""
Configuration for LD-Agent Batch experiment (LoComo dataset).

Extends ldagent/config_0.py with batch-specific parameters:
  BATCH_SIZE    — number of samples processed simultaneously in one batch
  QA_BATCH_SIZE — number of QA prompts sent in a single generate_batch_raw() call

All other settings mirror ldagent/config_0.py exactly.
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

DATASET_PATH    = PROJECT_ROOT / "dataset" / "locomo10.json"
BASE_OUTPUT_DIR = CONFIG_DIR
LOG_DIR         = CONFIG_DIR / "logs"

# =============================================================================
# 3. MEMORY  (EventMemory)
# =============================================================================

RELEVANCE_MEMORY_NUMBER = 3
RETRIEVE_K              = 3
DIST_THRESHOLD          = 1.5
ORI_MEM_QUERY           = False

# =============================================================================
# 4. TEMPORAL MODEL
# =============================================================================

DECAY_TEMP            = 1e-4
STM_FLUSH_GAP_SECONDS = 3600
SECONDS_PER_TURN      = 120

# =============================================================================
# 5. PERSONA
# =============================================================================

MAX_SPEAKER_A_PERSONAS = 10
MAX_SPEAKER_B_PERSONAS = 10

# =============================================================================
# 6. GENERATION  (Generator)
# =============================================================================

MAX_TOKENS     = 750
TEMPERATURE    = 0.7
TEMPERATURE_C5 = 0.5
JSON_RETRY     = 3

# =============================================================================
# 7. BATCH SETTINGS
# =============================================================================

# Number of samples processed simultaneously in one run_batch() call.
BATCH_SIZE = 10

# Number of QA prompts sent in one generate_batch_raw() sub-call.
QA_BATCH_SIZE = 64

# =============================================================================
# 8. EXPERIMENT SETTINGS
# =============================================================================

ENABLE_CHECKPOINTING    = True
SAVE_MEMORY_SNAPSHOTS   = True
ENABLE_LLM_CALL_LOGGING = True

# =============================================================================
# 9. LLM ENGINE
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
# 10. LOGGING
# =============================================================================

LOG_LEVEL   = "INFO"
LOG_TO_FILE = True

# =============================================================================
# 11. OUTPUT PATH HELPERS
# =============================================================================

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_output_dir(model_path: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}"


def get_sample_dir(model_path: str, start_sample: int, end_sample: int,
                   config_name: str = "config") -> Path:
    return get_output_dir(model_path, config_name) / f"sample_{start_sample}_{end_sample}"


def get_results_file(model_path: str, start_sample: int, end_sample: int,
                     config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    sample_dir = get_sample_dir(model_path, start_sample, end_sample, config_name)
    return sample_dir / f"results_{model_name}_sample_{start_sample}_{end_sample}.json"


def get_checkpoint_file(model_path: str, start_sample: int, end_sample: int,
                        config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    sample_dir = get_sample_dir(model_path, start_sample, end_sample, config_name)
    return sample_dir / f"checkpoint_{model_name}_sample_{start_sample}_{end_sample}.json"


def get_retrieval_log_dir(model_path: str, start_sample: int, end_sample: int,
                          config_name: str = "config") -> Path:
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / "retrieval_logs"


def get_memory_snapshots_dir(model_path: str, start_sample: int, end_sample: int,
                              config_name: str = "config") -> Path:
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / "memory_snapshots"


def get_prompt_log_dir(model_path: str, start_sample: int, end_sample: int,
                       config_name: str = "config") -> Path:
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / "prompt_log"


def get_merged_results_file(model_path: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, config_name) / f"results_{model_name}_merged.json"


def ensure_directories(model_path: str, start_sample: int, end_sample: int,
                       config_name: str = "config"):
    get_sample_dir(model_path, start_sample, end_sample, config_name).mkdir(
        parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, start_sample, end_sample, config_name).mkdir(
        parents=True, exist_ok=True)
    if SAVE_MEMORY_SNAPSHOTS:
        get_memory_snapshots_dir(model_path, start_sample, end_sample, config_name).mkdir(
            parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, start_sample, end_sample, config_name).mkdir(
        parents=True, exist_ok=True)
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
