"""
Unified LLM client supporting multiple inference engines:
- Together AI
- vLLM
- OpenAI

Features:
- System prompt support (chat format)
- JSON mode (OpenAI-compatible response_format)
- Structured output (vLLM GuidedDecodingParams)
- Optional JSON validation with retry logic
"""

import os
import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

try:
    import openai
except ImportError:
    openai = None

try:
    import requests
except ImportError:
    requests = None

try:
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import GuidedDecodingParams
        _VLLM_GUIDED_KEY = 'guided_decoding'
        _VLLM_GUIDED_CLS = GuidedDecodingParams
    except ImportError:
        GuidedDecodingParams = None
        from vllm.sampling_params import StructuredOutputsParams as _VLLM_GUIDED_CLS
        _VLLM_GUIDED_KEY = 'structured_outputs'
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    GuidedDecodingParams = None
    _VLLM_GUIDED_KEY = None
    _VLLM_GUIDED_CLS = None
    VLLM_AVAILABLE = False

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    ValidationError = None
    PYDANTIC_AVAILABLE = False

# Type alias for messages
Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}
Messages = List[Message]

DEFAULT_MAX_MODEL_LEN = 10000


def _make_guided_sampling_kwargs(json_schema) -> dict:
    """Return SamplingParams kwargs for guided/structured output, compatible with vllm 0.11.x and 0.19.x."""
    if json_schema and _VLLM_GUIDED_CLS is not None:
        return {_VLLM_GUIDED_KEY: _VLLM_GUIDED_CLS(json=json_schema)}
    return {}


def _build_messages(prompt: Optional[str], messages: Optional[Messages], system_prompt: Optional[str]) -> Messages:
    """Build a message list from prompt/messages and optional system prompt."""
    if messages is not None:
        result = list(messages)
        if system_prompt:
            result.insert(0, {"role": "system", "content": system_prompt})
        return result

    result = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    if prompt:
        result.append({"role": "user", "content": prompt})
    return result


def _build_prompt_from_messages(messages: Messages, tokenizer=None, enable_thinking: bool = False) -> str:
    """Build a prompt string from messages using chat template if available."""
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                # Model does not support enable_thinking (non-thinking model)
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logging.warning(f"Failed to apply chat template: {e}. Using simple format.")

    # Simple fallback format
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}\n")
        elif role == "user":
            parts.append(f"[User]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}\n")
    parts.append("[Assistant]\n")
    return "\n".join(parts)


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3-style <think>...</think> blocks from text.

    Removes complete blocks and handles unclosed tags caused by token-limit truncation.
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape literal newlines/tabs inside JSON string values.

    xgrammar occasionally emits raw control characters inside string tokens
    instead of their escaped forms, producing invalid JSON.
    """
    result = []
    in_string = False
    prev_backslash = False
    for ch in text:
        if prev_backslash:
            result.append(ch)
            prev_backslash = False
        elif ch == '\\' and in_string:
            result.append(ch)
            prev_backslash = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
    return ''.join(result)


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM mistake)."""
    return re.sub(r',(\s*[}\]])', r'\1', text)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response with multiple fallback strategies.

    Strategy order:
    1. Direct json.loads after stripping markdown fences.
    2. Escape literal control characters inside string values.
    3. Remove trailing commas.
    4. Both fixes combined.
    5. Extract the first {...} block, then apply strategies 1-4.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    text = _strip_think_tags(text)

    if not text:
        raise json.JSONDecodeError("Empty response from model", "", 0)

    def _try_all(t: str) -> dict:
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_escape_control_chars_in_strings(t))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(t))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_remove_trailing_commas(_escape_control_chars_in_strings(t)))
        except json.JSONDecodeError:
            pass
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        original_error = e

    result = _try_all(text)
    if result is not None:
        return result

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        result = _try_all(match.group())
        if result is not None:
            return result

    raise original_error


def _validate_against_schema(data: dict, schema: Union[dict, Any]) -> bool:
    """Validate data against a JSON schema or Pydantic model."""
    if PYDANTIC_AVAILABLE and isinstance(schema, type) and issubclass(schema, BaseModel):
        schema.model_validate(data)
        return True
    elif isinstance(schema, dict):
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                type_checks = {
                    "string": str, "integer": int, "number": (int, float),
                    "boolean": bool, "array": list, "object": dict
                }
                if expected_type in type_checks:
                    if not isinstance(value, type_checks[expected_type]):
                        raise ValueError(f"Field '{field}' type mismatch")
        return True
    return True


_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_REASONING_MAX_TOKENS_FLOOR = 4096


def _is_reasoning_model(model_name: Optional[str]) -> bool:
    """Detect OpenAI reasoning models (o1*, o3*, o4*, gpt-5*).

    These models use a different chat-completions contract:
      - require `max_completion_tokens` instead of `max_tokens`
      - reject custom `temperature` / `top_p` (only defaults accepted)
      - the visible-answer cap also covers hidden chain-of-thought tokens,
        so the cap needs to be much higher than for normal models.
    """
    if not model_name:
        return False
    m = model_name.lower()
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _adjust_for_reasoning_model(api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a chat.completions.create kwargs dict for reasoning-model compatibility.

    Modifies and returns the same dict. No-op for non-reasoning models.
    """
    if not _is_reasoning_model(api_kwargs.get("model", "")):
        return api_kwargs
    if "max_tokens" in api_kwargs:
        mt = api_kwargs.pop("max_tokens")
        api_kwargs["max_completion_tokens"] = max(int(mt or 0), _REASONING_MAX_TOKENS_FLOOR)
    elif "max_completion_tokens" in api_kwargs:
        api_kwargs["max_completion_tokens"] = max(
            int(api_kwargs["max_completion_tokens"] or 0), _REASONING_MAX_TOKENS_FLOOR
        )
    api_kwargs.pop("temperature", None)
    api_kwargs.pop("top_p", None)
    return api_kwargs


def _run_with_json_retry(fetch_fn, use_json_mode, schema, validate_json, json_retry, error_prefix="API"):
    """Execute fetch_fn with JSON parse/validate retry logic.

    fetch_fn() must return raw text. JSON parsing, validation, and retry are
    handled here. Non-JSON/ValueError exceptions are raised immediately.
    """
    last_error = None
    for attempt in range(json_retry if use_json_mode else 1):
        try:
            text = fetch_fn()
            if use_json_mode:
                parsed = _parse_json_response(text)
                if validate_json and schema:
                    _validate_against_schema(parsed, schema)
                return parsed
            return text
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logging.warning(f"JSON error (attempt {attempt + 1}/{json_retry}): {e}")
        except Exception as e:
            logging.error(f"{error_prefix} error: {e}")
            raise
    raise last_error


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Messages] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None,
        guided_json: Optional[Union[dict, Any]] = None,
        validate_json: bool = True,
        json_retry: int = 3,
        **kwargs
    ) -> Union[str, dict]:
        pass

    @abstractmethod
    async def generate_async(self, **kwargs) -> Union[str, dict]:
        pass


class TogetherAIClient(BaseLLMClient):
    """Together AI client."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('TOGETHER_API_KEY')

        if not self.api_key:
            raise ValueError("Together AI API key not found. Set TOGETHER_API_KEY environment variable.")

        self.base_url = "https://api.together.xyz/v1/chat/completions"
        logging.info(f"Initialized Together AI client with model: {model_name}")

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Messages] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None,
        guided_json: Optional[Union[dict, Any]] = None,
        validate_json: bool = True,
        json_retry: int = 3,
        **kwargs
    ) -> Union[str, dict]:
        if not requests:
            raise ImportError("requests library required")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        msgs = _build_messages(prompt, messages, system_prompt)

        use_json_mode = response_format is not None or guided_json is not None
        schema = guided_json

        if use_json_mode and msgs:
            json_instruction = "\nRespond with valid JSON only."
            if schema and isinstance(schema, dict):
                json_instruction += f" Follow this schema: {json.dumps(schema)}"

            if msgs[0].get("role") == "system":
                msgs[0]["content"] += json_instruction
            else:
                msgs.insert(0, {"role": "system", "content": json_instruction.strip()})

        data = {"model": self.model_name, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, **kwargs}
        if response_format:
            data["response_format"] = response_format

        def fetch():
            response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        return _run_with_json_retry(fetch, use_json_mode, schema, validate_json, json_retry, "Together AI")

    async def generate_async(self, **kwargs) -> Union[str, dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))


class vLLMClient(BaseLLMClient):
    """vLLM local inference client with structured output support."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, download_dir: str = None,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN, enable_thinking: bool = False,
                 dtype: str = "auto", quantization: Optional[str] = None,
                 enable_prefix_caching: bool = False):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

        self.model_path = model_path
        self.tokenizer = None
        self.enable_thinking = enable_thinking

        if download_dir:
            os.environ['HF_HOME'] = download_dir
            logging.info(f"Set Hugging Face cache directory to: {download_dir}")

        logging.info(f"Loading vLLM model from: {model_path}")
        logging.info(f"Using max_model_len={max_model_len} for model {model_path}")

        llm_kwargs = {
            'model': model_path,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': max_model_len,
            'trust_remote_code': True,
            'dtype': dtype,
            'enable_prefix_caching': enable_prefix_caching,
        }
        if download_dir:
            llm_kwargs['download_dir'] = download_dir
        if quantization is not None:
            llm_kwargs['quantization'] = quantization

        self.llm = LLM(**llm_kwargs)

        try:
            self.tokenizer = self.llm.get_tokenizer()
            logging.info("Tokenizer loaded for chat template support")
        except Exception as e:
            logging.warning(f"Could not load tokenizer: {e}")

        logging.info("vLLM model loaded successfully")

    def _build_prompt(self, prompt: Optional[str], messages: Optional[Messages],
                      system_prompt: Optional[str], use_json_mode: bool, schema: Optional[dict]) -> str:
        msgs = _build_messages(prompt, messages, system_prompt)

        if use_json_mode:
            json_instruction = "\nRespond with valid JSON only."
            if schema and isinstance(schema, dict):
                json_instruction += f" Follow this schema: {json.dumps(schema)}"

            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] += json_instruction
            else:
                msgs.insert(0, {"role": "system", "content": json_instruction.strip()})

        return _build_prompt_from_messages(msgs, self.tokenizer, enable_thinking=self.enable_thinking)

    def _get_json_schema(self, guided_json: Optional[Union[dict, Any]]) -> Optional[dict]:
        """Convert guided_json to JSON schema dict."""
        if guided_json is None:
            return None

        if PYDANTIC_AVAILABLE and isinstance(guided_json, type) and issubclass(guided_json, BaseModel):
            return guided_json.model_json_schema()
        elif isinstance(guided_json, dict):
            return guided_json
        return None

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Messages] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None,
        guided_json: Optional[Union[dict, Any]] = None,
        validate_json: bool = True,
        json_retry: int = 3,
        return_usage: bool = False,
        **kwargs
    ) -> Union[str, dict]:
        use_json_mode = response_format is not None or guided_json is not None
        schema = guided_json
        json_schema = self._get_json_schema(guided_json)

        full_prompt = self._build_prompt(
            prompt=prompt, messages=messages, system_prompt=system_prompt,
            use_json_mode=use_json_mode, schema=json_schema
        )

        sampling_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        for key in ['guided_json', 'response_format', 'validate_json', 'json_retry', 'return_usage']:
            kwargs.pop(key, None)
        sampling_kwargs.update(kwargs)

        sampling_kwargs.update(_make_guided_sampling_kwargs(json_schema))

        sampling_params = SamplingParams(**sampling_kwargs)

        last_error = None
        for attempt in range(json_retry if use_json_mode else 1):
            try:
                outputs = self.llm.generate([full_prompt], sampling_params, use_tqdm=False)
                text = _strip_think_tags(outputs[0].outputs[0].text.strip())

                usage_info = None
                if return_usage:
                    output = outputs[0]
                    usage_info = {
                        'prompt_tokens': len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0,
                        'completion_tokens': len(output.outputs[0].token_ids) if output.outputs else 0,
                    }

                if use_json_mode:
                    parsed = _parse_json_response(text)
                    if validate_json and schema:
                        _validate_against_schema(parsed, schema)
                    if return_usage:
                        parsed['_usage'] = usage_info
                    return parsed

                if return_usage:
                    return {'content': text, '_usage': usage_info}
                return text

            except json.JSONDecodeError as e:
                last_error = e
                logging.warning(f"JSON parse error (attempt {attempt + 1}/{json_retry}): {e}")
                if attempt < json_retry - 1:
                    continue
            except Exception as e:
                last_error = e
                if "JSON" in str(e) or "validation" in str(e).lower():
                    logging.warning(f"Validation error (attempt {attempt + 1}/{json_retry}): {e}")
                    if attempt < json_retry - 1:
                        continue
                logging.error(f"vLLM generation error: {e}")
                raise
        raise last_error

    def generate_batch_raw(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        guided_json: Optional[Union[dict, Any]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[List[str], tuple]:
        """Generate for a list of prompt strings in one vLLM call.

        All prompts share the same sampling parameters. System prompt and JSON
        mode handling mirrors the single-item generate() method.

        Returns:
            If return_usage=False: List of output strings (think tags stripped).
            If return_usage=True: Tuple of (texts, usages) where each usage is
                {'prompt_tokens': int, 'completion_tokens': int}.
        """
        use_json_mode = guided_json is not None
        json_schema = self._get_json_schema(guided_json)

        full_prompts = []
        for p in prompts:
            full_prompt = self._build_prompt(
                prompt=p, messages=None, system_prompt=system_prompt,
                use_json_mode=use_json_mode, schema=json_schema,
            )
            full_prompts.append(full_prompt)

        sampling_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        sampling_kwargs.update(_make_guided_sampling_kwargs(json_schema))
        for key in ['guided_json', 'return_usage', 'max_concurrent']:
            kwargs.pop(key, None)
        sampling_kwargs.update(kwargs)

        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self.llm.generate(full_prompts, sampling_params, use_tqdm=False)

        texts = [_strip_think_tags(o.outputs[0].text.strip()) for o in outputs]

        if return_usage:
            usages = [
                {
                    'prompt_tokens': len(o.prompt_token_ids) if hasattr(o, 'prompt_token_ids') else 0,
                    'completion_tokens': len(o.outputs[0].token_ids) if o.outputs else 0,
                }
                for o in outputs
            ]
            return texts, usages

        return texts

    async def generate_async(self, **kwargs) -> Union[str, dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        if openai is None:
            raise ImportError("openai library required")

        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = openai.OpenAI(api_key=self.api_key)
        logging.info(f"Initialized OpenAI client with model: {model_name}")

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Messages] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None,
        guided_json: Optional[Union[dict, Any]] = None,
        validate_json: bool = True,
        json_retry: int = 3,
        return_usage: bool = False,
        **kwargs
    ) -> Union[str, dict]:
        msgs = _build_messages(prompt, messages, system_prompt)

        use_json_mode = response_format is not None or guided_json is not None
        schema = guided_json

        if guided_json and not response_format:
            response_format = {"type": "json_object"}
            if schema and isinstance(schema, dict):
                schema_instruction = f"\nRespond with JSON following this schema: {json.dumps(schema)}"
                if msgs and msgs[0].get("role") == "system":
                    msgs[0]["content"] += schema_instruction
                else:
                    msgs.insert(0, {"role": "system", "content": schema_instruction.strip()})

        api_kwargs = {"model": self.model_name, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, **kwargs}
        if response_format:
            api_kwargs["response_format"] = response_format
        _adjust_for_reasoning_model(api_kwargs)

        if return_usage:
            response = self.client.chat.completions.create(**api_kwargs)
            content = response.choices[0].message.content.strip()
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                'completion_tokens': response.usage.completion_tokens if response.usage else 0,
            }
            if use_json_mode:
                parsed = _parse_json_response(content)
                if validate_json and schema:
                    _validate_against_schema(parsed, schema)
                parsed['_usage'] = usage_info
                return parsed
            return {'content': content, '_usage': usage_info}

        def fetch():
            response = self.client.chat.completions.create(**api_kwargs)
            return response.choices[0].message.content.strip()

        return _run_with_json_retry(fetch, use_json_mode, schema, validate_json, json_retry, "OpenAI")

    def generate_batch_raw(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        guided_json: Optional[Union[dict, Any]] = None,
        return_usage: bool = False,
        max_concurrent: int = 10,
        **kwargs
    ) -> Union[List[str], tuple]:
        """Batch-generate via async concurrency. Mirrors vLLMClient.generate_batch_raw().

        Individual call failures are swallowed and reported as empty strings so the
        caller's per-item retry path can handle them.
        """
        kwargs.pop('max_concurrent', None)

        async def gen_one(p: str):
            return await self.generate_async(
                prompt=p,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                guided_json=guided_json,
                return_usage=return_usage,
                **kwargs,
            )

        async def run_all():
            sem = asyncio.Semaphore(max_concurrent)
            async def bounded(p: str):
                async with sem:
                    try:
                        return await gen_one(p)
                    except Exception as e:
                        logging.warning(f"OpenAI batch item failed: {e}")
                        return None
            return await asyncio.gather(*[bounded(p) for p in prompts])

        results = asyncio.run(run_all())

        if return_usage:
            texts: List[str] = []
            usages: List[Dict[str, int]] = []
            for r in results:
                if isinstance(r, dict):
                    texts.append(r.get('content', ''))
                    usages.append(r.get('_usage', {'prompt_tokens': 0, 'completion_tokens': 0}))
                else:
                    texts.append("" if r is None else str(r))
                    usages.append({'prompt_tokens': 0, 'completion_tokens': 0})
            return texts, usages

        return [
            (r if isinstance(r, str) else (r.get('content', '') if isinstance(r, dict) else ""))
            for r in results
        ]

    async def generate_async(self, **kwargs) -> Union[str, dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(**kwargs))


class UnifiedLLMClient:
    """Unified LLM client supporting multiple inference engines."""

    SUPPORTED_ENGINES = ["together", "vllm", "openai"]

    def __init__(self, engine: str, model_name: Optional[str] = None, model_path: Optional[str] = None,
                 api_key: Optional[str] = None, **engine_kwargs):
        self.engine = engine.lower()

        if self.engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported engine: {engine}. Choose from {self.SUPPORTED_ENGINES}")

        if self.engine == "together":
            model_name = model_name or "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
            self.client = TogetherAIClient(model_name, api_key)

        elif self.engine == "vllm":
            if not model_path:
                raise ValueError("model_path required for vLLM engine")

            vllm_kwargs = {k: v for k, v in engine_kwargs.items()
                if k in ['tensor_parallel_size', 'gpu_memory_utilization', 'download_dir', 'hf_cache_dir',
                         'max_model_len', 'enable_thinking', 'dtype', 'quantization', 'enable_prefix_caching']}
            if 'hf_cache_dir' in vllm_kwargs:
                vllm_kwargs['download_dir'] = vllm_kwargs.pop('hf_cache_dir')
            self.client = vLLMClient(model_path, **vllm_kwargs)

        elif self.engine == "openai":
            model_name = model_name or "gpt-3.5-turbo"
            self.client = OpenAIClient(model_name, api_key)

        logging.info(f"Unified LLM client initialized with engine: {self.engine}")

    def generate(self, prompt: Optional[str] = None, messages: Optional[Messages] = None,
                 system_prompt: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.7,
                 response_format: Optional[Dict[str, str]] = None, guided_json: Optional[Union[dict, Any]] = None,
                 validate_json: bool = True, json_retry: int = 3, **kwargs) -> Union[str, dict]:
        return self.client.generate(
            prompt=prompt, messages=messages, system_prompt=system_prompt,
            max_tokens=max_tokens, temperature=temperature, response_format=response_format,
            guided_json=guided_json, validate_json=validate_json, json_retry=json_retry, **kwargs
        )

    async def generate_async(self, **kwargs) -> Union[str, dict]:
        return await self.client.generate_async(**kwargs)

    def generate_batch(self, prompts: Optional[List[str]] = None, messages_list: Optional[List[Messages]] = None,
                       system_prompt: Optional[str] = None, **kwargs) -> List[Union[str, dict]]:
        if prompts is not None:
            items = [(p, None) for p in prompts]
        elif messages_list is not None:
            items = [(None, m) for m in messages_list]
        else:
            raise ValueError("Either prompts or messages_list must be provided")

        return [self.generate(prompt=p, messages=m, system_prompt=system_prompt, **kwargs) for p, m in items]

    def generate_batch_raw(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        guided_json: Optional[Union[dict, Any]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[List[str], tuple]:
        """Batch generation (vLLM or OpenAI). See vLLMClient/OpenAIClient.generate_batch_raw() for details."""
        if not hasattr(self.client, 'generate_batch_raw'):
            raise NotImplementedError(
                f"generate_batch_raw() is not supported for engine: {self.engine}"
            )
        return self.client.generate_batch_raw(
            prompts=prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            guided_json=guided_json,
            return_usage=return_usage,
            **kwargs,
        )

    async def generate_batch_async(self, prompts: Optional[List[str]] = None, messages_list: Optional[List[Messages]] = None,
                                   system_prompt: Optional[str] = None, max_concurrent: int = 10, **kwargs) -> List[Union[str, dict]]:
        if prompts is not None:
            items = [(p, None) for p in prompts]
        elif messages_list is not None:
            items = [(None, m) for m in messages_list]
        else:
            raise ValueError("Either prompts or messages_list must be provided")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def gen(p, m):
            async with semaphore:
                return await self.generate_async(prompt=p, messages=m, system_prompt=system_prompt, **kwargs)

        return await asyncio.gather(*[gen(p, m) for p, m in items])

    def get_engine_info(self) -> Dict[str, Any]:
        info = {"engine": self.engine}
        if hasattr(self.client, 'model_name'):
            info["model_name"] = self.client.model_name
        if hasattr(self.client, 'model_path'):
            info["model_path"] = self.client.model_path
        return info


def create_llm_client(engine: str, **kwargs) -> UnifiedLLMClient:
    """Factory function to create LLM client."""
    return UnifiedLLMClient(engine, **kwargs)


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""

    def __init__(self, mock_responses: Optional[List[Union[str, dict]]] = None):
        self.mock_responses = mock_responses or ["Mock response"]
        self.call_count = 0
        self.call_history = []

    def generate(self, **kwargs) -> Union[str, dict]:
        self.call_history.append(kwargs)
        response = self.mock_responses[self.call_count % len(self.mock_responses)]
        self.call_count += 1
        return response

    async def generate_async(self, **kwargs) -> Union[str, dict]:
        return self.generate(**kwargs)
