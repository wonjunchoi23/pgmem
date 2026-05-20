"""
Configuration for OnlyLLM Baseline Experiment on LoCoMo
"""

import os
import sys
from pathlib import Path


os.environ["HF_TOKEN"] = "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"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


CONFIG_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CONFIG_DIR.parent

LLM_MODULE_DIR = PROJECT_ROOT / "llm_module"
sys.path.insert(0, str(LLM_MODULE_DIR))

DATASET_PATH = PROJECT_ROOT / "dataset" / "locomo10.json"

BASE_OUTPUT_DIR = CONFIG_DIR
LOG_DIR = CONFIG_DIR / "logs"
LOG_TO_FILE = True
LOG_LEVEL = "INFO"


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
TEMPERATURE_C5 = 0.5
MAX_TOKENS = 500
JSON_RETRY = 5


HISTORY_ALL_GIVEN = True
MAX_CONTEXT_TURNS = 20
OUTPUT_TOKEN_RESERVE = 600
CONTEXT_SAFETY_MARGIN = 200


ENABLE_CHECKPOINTING = True
SAVE_MEMORY_SNAPSHOTS = False
ENABLE_LLM_CALL_LOGGING = True


BATCH_SIZE = 4
QA_BATCH_SIZE = 64


def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_output_dir(model_path: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}"


def get_sample_dir(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
) -> Path:
    return get_output_dir(model_path, config_name) / f"sample_{start_sample}_{end_sample}"


def get_results_file(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
) -> Path:
    model_name = extract_model_name(model_path)
    sample_dir = get_sample_dir(model_path, start_sample, end_sample, config_name)
    return sample_dir / f"results_{model_name}_sample_{start_sample}_{end_sample}.json"


def get_checkpoint_file(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
) -> Path:
    model_name = extract_model_name(model_path)
    sample_dir = get_sample_dir(model_path, start_sample, end_sample, config_name)
    return sample_dir / f"checkpoint_{model_name}_sample_{start_sample}_{end_sample}.json"


def get_retrieval_log_dir(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
) -> Path:
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / "retrieval_logs"


def get_prompt_log_dir(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
) -> Path:
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / "prompt_log"


def get_merged_results_file(model_path: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, config_name) / f"results_{model_name}_merged.json"


def ensure_directories(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config",
):
    get_sample_dir(model_path, start_sample, end_sample, config_name).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, start_sample, end_sample, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, start_sample, end_sample, config_name).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_llm_config(engine: str = None) -> dict:
    if engine is None:
        engine = LLM_ENGINE
    if engine == "vllm":
        return {"engine": "vllm", **DEFAULT_VLLM_CONFIG}
    if engine == "together":
        return {"engine": "together", **TOGETHER_CONFIG}
    if engine == "openai":
        return {"engine": "openai", **OPENAI_CONFIG}
    raise ValueError(f"Unknown LLM engine: {engine}")


__all__ = [name for name in globals() if not name.startswith("__")]
