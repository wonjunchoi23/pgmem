# LLM Client Guide

## Key Takeaways

- Create a client with `create_llm_client(engine=...)` and call `client.generate(...)`.
- Supported engines: **vLLM**, **Together AI**, **OpenAI**.
- JSON / structured output:
  - Use `guided_json` (JSON Schema or Pydantic model).
  - Use `validate_json=True` + `json_retry` to reduce parse failures.
- Engine setup:
  - vLLM: local inference (GPU), install `vllm`.
  - Together AI: hosted inference, set `TOGETHER_API_KEY`.
  - OpenAI: hosted inference, set `OPENAI_API_KEY`.

---

## Project Structure
```
Experiments/
├── llm_module/
│   ├── __init__.py
│   ├── llm_client.py      # main client
│   ├── models/            # model download location
│   └── cache/             # HF cache
└── test_llm_client.py
```

---

## Quick Start

```python
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "llm_module", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
sys.path.insert(0, PROJECT_ROOT)

from llm_module.llm_client import create_llm_client

# vLLM (local GPU)
client = create_llm_client(
    engine="vllm",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    download_dir=MODEL_DIR
)

# Together AI (hosted)
# Requirements: pip install requests, export TOGETHER_API_KEY=...
together_client = create_llm_client(
    engine="together",
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

# OpenAI (hosted)
# Requirements: pip install openai, export OPENAI_API_KEY=...
openai_client = create_llm_client(
    engine="openai",
    model_name="gpt-4o-mini",
)
```

---

## Generation Methods

### Basic
```python
response = client.generate(prompt="What is ML?")
```

### With System Prompt
```python
response = client.generate(
    prompt="Explain neural networks",
    system_prompt="You are a helpful tutor. Be concise.",
    max_tokens=200,
    temperature=0.7
)
```

### With Token Counting
```python
response = client.generate(
    prompt="What is machine learning?",
    return_usage=True  # vLLM only
)

# Response format:
# {"content": "Machine learning is...", "_usage": {"prompt_tokens": 150, "completion_tokens": 80}}

if isinstance(response, dict) and '_usage' in response:
    usage = response['_usage']
    print(f"Input tokens: {usage['prompt_tokens']}")
    print(f"Output tokens: {usage['completion_tokens']}")
```

### Chat Format
```python
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "What is Python?"}
]
response = client.generate(messages=messages, system_prompt="Be helpful.")
```

---

## JSON Mode

### With Schema (Structured Output)
```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "integer"}
    },
    "required": ["name", "score"]
}

response = client.generate(
    prompt="Generate a student profile",
    guided_json=schema,
    temperature=0.3
)
# response is a dict: {"name": "Alice", "score": 95}
```

### With Pydantic
```python
from pydantic import BaseModel

class Profile(BaseModel):
    name: str
    score: int

response = client.generate(prompt="Generate a profile", guided_json=Profile)
```

### JSON Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `guided_json` | JSON schema dict or Pydantic model | None |
| `validate_json` | Validate output against schema | True |
| `json_retry` | Retry count on parse failure | 3 |
| `return_usage` | Include token usage in response (vLLM only) | False |

---

## vLLM Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_path` | HuggingFace model ID | Required |
| `download_dir` | Model download path | None |
| `tensor_parallel_size` | Number of GPUs | 1 |
| `gpu_memory_utilization` | GPU memory fraction | 0.9 |
| `max_model_len` | Max token context length | 10000 |
| `dtype` | Model weight dtype (e.g. `"float16"`, `"bfloat16"`, `"auto"`) | `"auto"` |
| `quantization` | Quantization method (e.g. `"awq"`, `"gptq"`, `"fp8"`) | None |
| `enable_thinking` | Enable thinking mode (Qwen3 etc.) | False |

### Multi-GPU Example
```python
client = create_llm_client(
    engine="vllm",
    model_path="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    download_dir=MODEL_DIR
)
```

### Quantized Model Example
```python
client = create_llm_client(
    engine="vllm",
    model_path="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
    dtype="float16",
    download_dir=MODEL_DIR
)
```

---

## Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=hf_xxxxx
```

### Accessing gated models (e.g., Llama)
```bash
huggingface-cli login --token hf_xxxxx
```
> You must enable gated-repo access for your HuggingFace token.

---

## Together AI Setup

- Install: `pip install requests`
- Environment variable: `export TOGETHER_API_KEY=...`
- Engine: `engine="together"`

```python
client = create_llm_client(
    engine="together",
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
)
resp = client.generate(prompt="Say hello in one sentence.")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 401 Unauthorized (Together/OpenAI) | Check API key env variable |
| 403 Forbidden | Enable gated-repo access for your HF token |
| OOM Error | Lower `gpu_memory_utilization` or reduce `max_model_len` |
| JSON parse fails | Increase `json_retry`, explicitly request JSON in the prompt |
| Import error | `pip install vllm pydantic` |
