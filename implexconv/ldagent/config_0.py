"""
Configuration for LD-Agent experiment (ImplexConv dataset).

All experiment-wide constants are defined here.
run_experiment.py and merge_results.py load this file dynamically via --config argument;
sub-modules (ldagent_module, generator, personas, event_memory) receive config values
as constructor parameters and do not import this file directly.

Sections
--------
1.  Environment & paths
2.  Dataset
3.  Dialogue roles
4.  Memory (EventMemory)
5.  Persona (Personas)
6.  Generation (Generator)
7.  Experiment settings
8.  LLM engine
9.  Logging
10. Output path helpers
"""
# Changelog (v2):
#   - TURN_DECAY_TEMP removed; replaced by DECAY_TEMP (seconds-based virtual time)
#   - Virtual time model added: CONV_IDS_PER_DAY, MINUTES_PER_TURN, FINALIZE_EVERY_N_CONVS
#   - ENABLE_LLM_CALL_LOGGING added
#   - get_prompt_log_dir() added; ensure_directories() updated accordingly

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

DATASET_DIR        = PROJECT_ROOT / "dataset" / "implexconv"
DATASET_OPPOSED    = DATASET_DIR / "ImplexConv_opposed_processed.json"
DATASET_SUPPORTIVE = DATASET_DIR / "ImplexConv_supportive_processed.json"

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

# Number of LTM entries fetched by relevance_retrieve during turn processing
RELEVANCE_MEMORY_NUMBER = 3

# Number of LTM entries fetched by relevance_retrieve during QA
RETRIEVE_K = 3

# L2 distance threshold for relevance retrieval (sentence-transformer embedding space)
DIST_THRESHOLD = 1.5

# Use original query text instead of noun-extracted query
ORI_MEM_QUERY = False

# =============================================================================
# 4a. VIRTUAL TIME MODEL
# =============================================================================

# 1 virtual day  = CONV_IDS_PER_DAY consecutive conv_ids
# 1 virtual turn = MINUTES_PER_TURN minutes
#
# virtual_seconds for a turn = compute_virtual_seconds(conv_id, turn_id)
# 1 day  ≈ CONV_IDS_PER_DAY × (avg turns/conv) × MINUTES_PER_TURN × 60 s
#         ≈ 2 × 10 × 10 × 60 = 12,000 virtual seconds
#
# DECAY_TEMP (seconds-based): exp(-DECAY_TEMP × elapsed_vs)
#   exp(-1e-4 × 12000) ≈ 0.30  →  30% weight retained after 1 virtual day
CONV_IDS_PER_DAY       = 2      # how many conv_ids = 1 virtual day (same as theanine)
MINUTES_PER_TURN       = 10     # minutes per local turn_id step (same as theanine)
FINALIZE_EVERY_N_CONVS = CONV_IDS_PER_DAY   # STM→LTM flush every N conv_ids (= 1 day)
DECAY_TEMP             = 1e-4   # time-decay coefficient (virtual seconds)

# =============================================================================
# 5. PERSONA  (Personas)
# =============================================================================

MAX_USER_PERSONAS  = 10   # keep latest N user traits   (0 = unlimited)
MAX_AGENT_PERSONAS = 10   # keep latest N agent traits  (0 = unlimited)

# =============================================================================
# 6. GENERATION  (Generator)
# =============================================================================

MAX_TOKENS  = 750
TEMPERATURE = 0.7
JSON_RETRY  = 3

# =============================================================================
# 7. EXPERIMENT SETTINGS
# =============================================================================

# Timing is measured only for turns in this conv_id during Phase 1
TIMING_CONV_ID = 0

BATCH_SIZE = 4
QA_BATCH_SIZE = 32

ENABLE_CHECKPOINTING    = True
SAVE_MEMORY_SNAPSHOTS   = True
ENABLE_LLM_CALL_LOGGING = True

# =============================================================================
# 8. LLM ENGINE
# =============================================================================

# One of: "vllm" | "openai" | "together"
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

LOG_LEVEL   = "INFO"    # DEBUG | INFO | WARNING | ERROR
LOG_TO_FILE = True

# =============================================================================
# 10. OUTPUT PATH HELPERS
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
    model_name  = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"results_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_checkpoint_file(model_path: str, subset: str, start_session: int, end_session: int,
                        config_name: str = "config") -> Path:
    model_name  = extract_model_name(model_path)
    session_dir = get_session_dir(model_path, subset, start_session, end_session, config_name)
    return session_dir / f"checkpoint_{model_name}_{subset}_session_{start_session}_{end_session}.json"


def get_retrieval_log_dir(model_path: str, subset: str, start_session: int, end_session: int,
                          config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "retrieval_logs"


def get_memory_snapshots_dir(model_path: str, subset: str, start_session: int, end_session: int,
                              config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "memory_snapshots"


def get_prompt_log_dir(model_path: str, subset: str, start_session: int, end_session: int,
                       config_name: str = "config") -> Path:
    return get_session_dir(model_path, subset, start_session, end_session, config_name) / "prompt_log"


def get_merged_results_file(model_path: str, subset: str, config_name: str = "config") -> Path:
    model_name = extract_model_name(model_path)
    return get_output_dir(model_path, subset, config_name) / f"results_{model_name}_{subset}_merged.json"


def ensure_directories(model_path: str, subset: str, start_session: int, end_session: int,
                       config_name: str = "config"):
    """Create all required output directories."""
    get_session_dir(model_path, subset, start_session, end_session, config_name).mkdir(
        parents=True, exist_ok=True)
    get_retrieval_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(
        parents=True, exist_ok=True)
    if SAVE_MEMORY_SNAPSHOTS:
        get_memory_snapshots_dir(model_path, subset, start_session, end_session, config_name).mkdir(
            parents=True, exist_ok=True)
    get_prompt_log_dir(model_path, subset, start_session, end_session, config_name).mkdir(
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
