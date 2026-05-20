"""Configuration for GraphMem v7 on LoComo.

config_7: retrieval top-k tuned per c7 spec.
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
    "model_name": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

TEMPERATURE = 0.7
TEMPERATURE_C5 = 0.5
MAX_TOKENS = 1000

MAX_TOKENS_STATE              = 1000
MAX_TOKENS_STATE_NEW_REL      = 800
MAX_TOKENS_EPISODE            = 1000
MAX_TOKENS_EPISODE_NEW_REL    = 2000
MAX_TOKENS_TRAIT              = 1200
MAX_TOKENS_TRAIT_EVIDENCE_5A  = 2500
MAX_TOKENS_TRAIT_EXTRA_REL_5B = 1500
MAX_TOKENS_STATE_STATE_5C     = 1000
MAX_TOKENS_STATE_EPISODE_5D   = 800

JSON_RETRY = 5
JUDGMENT_RETRY = 3


BATCH_SIZE = 4
QA_BATCH_SIZE = 64


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Extraction (same as config_0)
STATE_EXTRACTION_H = 1
STATE_MAX_COUNT = 1
STATE_REF_CONTEXT_TURNS = 3
CHUNK_SIZE_CONV = 1
TRAIT_EXTRACTION_CHUNKS = 2
TRAIT_MAX_COUNT = 1
MAX_KEYWORDS = 14
MAX_DOMAIN_LABELS = 14
MIN_DOMAIN_LABELS = 10

# --- config_7 retrieval top-k ---
K_CONTEXT = 32
K_EPISODE = 10
K_EPISODE_FINAL = 8
K_STATE = 27
K_TRAIT = 10
K_APS = 5

W_SEM_C = 0.65
W_OV_C = 0.35

W_SEM_NARROW = 0.60
W_OV_NARROW = 0.40
W_SEM_BROAD = 0.85
W_OV_BROAD = 0.15

W_PAIR_SEM = 0.7
W_PAIR_LEX = 0.3

# --- config_7 final-set top-k ---
K_T_FINAL = 7
K_SF = 23
TRAIT_VALIDATION_TAU = 0.7
W_SR = 0.5

SIGN_PROP_HOP_CAP = 10

SEED_TURN_NEIGHBOR_ENABLED = True
SEED_TURN_NEIGHBOR_DELTA = 1

APS_EXCLUDE_SHIFT_SOURCE = True
ENABLE_SHIFT_CHAIN_PRUNING = True
STRICT_HIGH_DEFAULT_LOW = True

ENABLE_EXTRA_RELATION_EXTRACTION = True
EXTRA_REL_ONLY_IF_UNCONNECTED = True
TRAIT_EXTRA_REL_TOPK_STATE = 7
TRAIT_EXTRA_REL_TOPK_EPISODE = 3
STATE_STATE_EXTRA_REL_TOPK = 5
STATE_EPISODE_EXTRA_REL_TOPK = 3
STATE_NEW_REL_PREV_WINDOW = 5

CONTEXT_CACHE_SIZE = 10
MINUTES_PER_TURN_IN_SESSION = 10

INCLUDE_RECENT_CONVERSATION_FOR_QA = False
QA_CONTEXT_TURNS = 10

ENABLE_CHECKPOINTING = True
CHECKPOINT_INTERVAL = 1
SAVE_MEMORY_SNAPSHOTS = True
ENABLE_LLM_CALL_LOGGING = True
LLM_CALL_LOG_FIRST_N_SAMPLES = 10


def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_output_dir(model_path: str, config_name: str = "config_7") -> Path:
    model_name = extract_model_name(model_path)
    return BASE_OUTPUT_DIR / f"{config_name}_outputs_{model_name}"


def get_sample_dir(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_7",
) -> Path:
    return get_output_dir(model_path, config_name) / f"sample_{start_sample}_{end_sample}"


def _suffix_tag(suffix: str | None) -> str:
    return f"__{suffix}" if suffix else ""


def get_results_file(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_7",
    results_suffix: str | None = None,
) -> Path:
    model_name = extract_model_name(model_path)
    sample_dir = get_sample_dir(model_path, start_sample, end_sample, config_name)
    tag = _suffix_tag(results_suffix)
    return sample_dir / f"results_{model_name}{tag}_sample_{start_sample}_{end_sample}.json"


def get_checkpoint_file(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_7",
    results_suffix: str | None = None,
) -> Path:
    model_name = extract_model_name(model_path)
    sample_dir = get_sample_dir(model_path, start_sample, end_sample, config_name)
    tag = _suffix_tag(results_suffix)
    return sample_dir / f"checkpoint_{model_name}{tag}_sample_{start_sample}_{end_sample}.json"


def get_retrieval_log_dir(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_7",
    results_suffix: str | None = None,
) -> Path:
    tag = _suffix_tag(results_suffix)
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / f"retrieval_logs{tag}"


def get_memory_snapshots_dir(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_7",
) -> Path:
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / "memory_snapshots"


def get_prompt_log_dir(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_7",
    results_suffix: str | None = None,
) -> Path:
    tag = _suffix_tag(results_suffix)
    return get_sample_dir(model_path, start_sample, end_sample, config_name) / f"prompt_log{tag}"


def get_merged_results_file(
    model_path: str,
    config_name: str = "config_7",
) -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, config_name) / f"results_{model_name}_merged.json"


def ensure_directories(
    model_path: str,
    start_sample: int,
    end_sample: int,
    config_name: str = "config_7",
    results_suffix: str | None = None,
) -> None:
    get_sample_dir(model_path, start_sample, end_sample, config_name).mkdir(parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, start_sample, end_sample, config_name, results_suffix).mkdir(parents=True, exist_ok=True)
    get_memory_snapshots_dir(model_path, start_sample, end_sample, config_name).mkdir(parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, start_sample, end_sample, config_name, results_suffix).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_llm_config(engine: str = None) -> dict:
    engine = engine or LLM_ENGINE
    if engine == "vllm":
        return {"engine": "vllm", **DEFAULT_VLLM_CONFIG}
    if engine == "together":
        return {"engine": "together", **TOGETHER_CONFIG}
    if engine == "openai":
        return {"engine": "openai", **OPENAI_CONFIG}
    raise ValueError(f"Unknown LLM engine: {engine}")
