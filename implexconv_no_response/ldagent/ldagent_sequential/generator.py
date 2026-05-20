"""
Response Generator Module for LD-Agent (ImplexConv)

Changes from ldagent/generator.py:
  - response_build_with_drift_detection() removed (no drift detection in new protocol)
  - response_build_json() added: JSON structured response generation with token tracking
    Returns (response, token_info, prompt_snapshot) — prompt_snapshot is used for
    retrieval logging.
  - generate_qa_answer() updated:
      * subset parameter added ("opposed" free-form, "supportive" yes/no/unknown)
      * guided_json + return_usage=True for reliable token tracking
      * Returns (answer, token_info, prompt_snapshot)
  - generate_qa_answer_json() removed (merged into generate_qa_answer)
  - Prompts updated: "maximum 50 words" → "maximum 100 words"

Reference: "Hello Again! LLM-powered Personalized Agent for Long-term Dialogue"
           (Li et al., NAACL 2025)
"""

import json
import logging
import re
from typing import List, Dict, Tuple, Any

from load_dataset import convert_seconds_to_full_time

logger = logging.getLogger(__name__)


# =============================================================================
# JSON SCHEMAS
# =============================================================================

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
    },
    "required": ["response"],
}

QA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}

QA_SCHEMA_SUPPORTIVE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["answer"],
}


# =============================================================================
# GENERATOR CLASS
# =============================================================================

class Generator:
    """
    Response generation module for LD-Agent.

    Generates:
      1. Personalized conversational responses (JSON structured output)
      2. QA answers based on accumulated LTM (opposed: free-form; supportive: yes/no/unknown)

    Adapted from original LD-Agent (Generator.py):
      - LLM client interface unchanged (unified client)
      - Original prompts preserved; max word limit updated to 100
      - Drift detection removed; response_build_json() returns (response, token_info, prompt_snapshot)
    """

    def __init__(
        self,
        llm_client,
        logger: logging.Logger,
        usr_name:    str   = "User",
        agent_name:  str   = "Agent",
        max_tokens:  int   = 250,
        temperature: float = 0.7,
        json_retry:  int   = 3,
    ):
        self.llm_client  = llm_client
        self.logger      = logger
        self.usr_name    = usr_name
        self.agent_name  = agent_name
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.json_retry  = json_retry

        # LLMCallLogger — injected externally; None = no logging
        self.llm_logger = None

        logger.info(f"Generator initialized (usr={usr_name}, agent={agent_name})")

    # =========================================================================
    # LLM LOGGER INJECTION
    # =========================================================================

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

    # =========================================================================
    # TOKEN COUNTING (no LLM call)
    # =========================================================================

    def _count_prompt_tokens(self, prompt: str) -> int:
        """
        Estimate input token count for a prompt string.

        Uses the vLLM tokenizer when available; falls back to len // 4.
        Note: counts the raw prompt string, not the full chat-templated form,
        so the result is a slight underestimate of actual prompt_tokens.
        """
        inner = getattr(self.llm_client, 'client', None)
        tokenizer = getattr(inner, 'tokenizer', None) if inner else None
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(prompt))
            except Exception:
                pass
        return len(prompt) // 4

    # =========================================================================
    # PROMPT CONSTRUCTION  (from original LD-Agent)
    # =========================================================================

    def _select_prompts(
        self,
        inquiry:      str,
        context:      str,
        memories:     str,
        user_traits:  str,
        agent_traits: str,
    ) -> Tuple[str, str]:
        """
        Construct prompts for response generation.
        Keeps original LD-Agent structure; max word limit updated to 100.
        """
        sys_prompt = (
            f"As a communication expert with outstanding communication habits, "
            f"you embody the role of {self.agent_name} throughout the following dialogues. "
            f"Here are some of your distinctive personal traits: {agent_traits}.\n"
        )

        user_prompt = (
            f"<CONTEXT>\n"
            f"Drawing from your recent conversation with {self.usr_name}:\n{context}\n"
            f"<MEMORY>\n"
            f"The memories linked to the ongoing conversation are:\n{memories}\n"
            f"<USER_TRAITS>\n"
            f"During the conversation process between you and {self.usr_name} in the past, "
            f"you found that the {self.usr_name} has the following characteristics:\n{user_traits}\n"
            f"\nNow, please role-play as {self.agent_name} to continue the dialogue between "
            f"{self.agent_name} and {self.usr_name}.\n"
            f"{self.usr_name} just said: {inquiry}\n"
            f"Please respond to {self.usr_name}'s statement in English (maximum 100 words).\n"
            f"Respond in JSON format with key \"response\".\n"
        )

        return sys_prompt, user_prompt

    # =========================================================================
    # RESPONSE PROMPT CONSTRUCTION (no LLM call)
    # =========================================================================

    def build_response_prompt_only(
        self,
        inquiry:      str,
        context:      str,
        memories:     str,
        user_traits:  str,
        agent_traits: str,
    ) -> Tuple[str, int]:
        """
        Build the response generation prompt without calling the LLM.

        Constructs the same prompt as response_build_json() but skips the LLM
        call entirely. Logs to call_1_response with output=null, and counts
        input tokens via the tokenizer.

        Returns:
            (prompt_snapshot: str, input_token_count: int)
        """
        sys_prompt, user_prompt = self._select_prompts(
            inquiry, context, memories, user_traits, agent_traits
        )
        prompt_snapshot = f"[SYSTEM]\n{sys_prompt}\n[USER]\n{user_prompt}"

        if self.llm_logger is not None:
            self.llm_logger.log("call_1_response", sys_prompt, user_prompt, None)

        input_tokens = self._count_prompt_tokens(prompt_snapshot)
        return prompt_snapshot, input_tokens

    # =========================================================================
    # RESPONSE GENERATION — JSON STRUCTURED (primary method)
    # =========================================================================

    def response_build_json(
        self,
        inquiry:      str,
        context:      str,
        memories:     str,
        user_traits:  str,
        agent_traits: str,
    ) -> Tuple[str, Dict[str, int], str]:
        """
        Generate a personalized response (JSON structured output).

        Returns:
            (response, token_info, prompt_snapshot)
            - response: generated response string
            - token_info: {"input": int, "output": int}
            - prompt_snapshot: full prompt sent to LLM (for retrieval logging)
        """
        sys_prompt, user_prompt = self._select_prompts(
            inquiry, context, memories, user_traits, agent_traits
        )
        prompt_snapshot = f"[SYSTEM]\n{sys_prompt}\n[USER]\n{user_prompt}"

        token_info = {"input": 0, "output": 0}
        try:
            result = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                guided_json=RESPONSE_SCHEMA,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                json_retry=self.json_retry,
                return_usage=True,
            )

            if isinstance(result, dict) and "_usage" in result:
                usage = result["_usage"]
                token_info = {
                    "input":  usage.get("prompt_tokens",     0),
                    "output": usage.get("completion_tokens", 0),
                }

            if self.llm_logger is not None:
                self.llm_logger.log("call_1_response", sys_prompt, user_prompt, result)

            response = result.get("response", "") if isinstance(result, dict) else str(result)
            response = str(response).strip()

        except Exception as e:
            self.logger.error(f"Error in response_build_json: {e}")
            response = ""

        return response, token_info, prompt_snapshot

    # =========================================================================
    # RESPONSE GENERATION — PLAIN TEXT (kept for backward compatibility)
    # =========================================================================

    def response_build(
        self,
        inquiry:      str,
        context:      str,
        memories:     str,
        user_traits:  str,
        agent_traits: str,
    ) -> str:
        """Generate a response (plain text, no token tracking). Kept for compat."""
        sys_prompt, user_prompt = self._select_prompts(
            inquiry, context, memories, user_traits, agent_traits
        )
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if isinstance(response, dict):
                response = response.get("response", str(response))
            response = str(response).strip()
            if response.upper().startswith("RESPONSE:"):
                response = response[9:].strip()
            return response
        except Exception as e:
            self.logger.error(f"Error in response_build: {e}")
            return ""

    # =========================================================================
    # QA GENERATION
    # =========================================================================

    def generate_qa_answer(
        self,
        question:     str,
        memories:     str,
        user_traits:  str = "",
        agent_traits: str = "",
        subset:       str = "opposed",
        context:      str = "",
    ) -> Tuple[str, Dict[str, int], str, int]:
        """
        Generate an answer for a QA task based on accumulated LTM + STM context.

        Args:
            question:     Question string (3rd-person for supportive subset)
            memories:     Formatted retrieved memories string
            user_traits:  Current user trait summary (optional)
            agent_traits: Current agent trait summary (optional)
            subset:       "opposed" (free-form) or "supportive" (yes/no/unknown)
            context:      Recent conversation turns from STM (optional)

        Returns:
            (answer, token_info, prompt_snapshot, num_api_calls)
            num_api_calls: 1 if the call succeeded, 0 if an exception occurred.
            Failed retry attempts (json_retry) inside llm_client are not counted.
        """
        if subset == "supportive":
            sys_prompt, user_prompt = self._build_qa_prompt_supportive(
                question, memories, user_traits, agent_traits, context
            )
        else:
            sys_prompt, user_prompt = self._build_qa_prompt_opposed(
                question, memories, user_traits, agent_traits, context
            )

        prompt_snapshot = f"[SYSTEM]\n{sys_prompt}\n[USER]\n{user_prompt}"
        token_info = {"input": 0, "output": 0}

        schema = QA_SCHEMA_SUPPORTIVE if subset == "supportive" else QA_RESPONSE_SCHEMA

        try:
            result = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                guided_json=schema,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                json_retry=self.json_retry,
                return_usage=True,
            )

            if isinstance(result, dict) and "_usage" in result:
                usage = result["_usage"]
                token_info = {
                    "input":  usage.get("prompt_tokens",     0),
                    "output": usage.get("completion_tokens", 0),
                }

            if self.llm_logger is not None:
                self.llm_logger.log("call_5_qa", sys_prompt, user_prompt, result)

            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            answer = str(answer).strip()

        except Exception as e:
            self.logger.error(f"Error in generate_qa_answer: {e}")
            answer = ""
            return answer, token_info, prompt_snapshot, 0

        return answer, token_info, prompt_snapshot, 1

    def _build_qa_prompt_opposed(
        self,
        question:     str,
        memories:     str,
        user_traits:  str,
        agent_traits: str = "",
        context:      str = "",
    ) -> Tuple[str, str]:
        """Prompt for opposed subset: free-form answer."""
        sys_prompt = (
            "You are a helpful assistant that answers questions about a user "
            "based on recorded conversation memories. "
            "Use the provided memories to give accurate and concise answers."
        )
        if agent_traits:
            sys_prompt += f" Your personal traits as the assistant: {agent_traits}."
        user_prompt = ""
        if context:
            user_prompt += (
                f"<CONTEXT>\n"
                f"Recent conversation turns:\n{context}\n\n"
            )
        user_prompt += (
            f"<MEMORIES>\n"
            f"The following are conversation memories about the user:\n{memories}\n"
        )
        if user_traits:
            user_prompt += f"\n<USER_TRAITS>\nUser characteristics:\n{user_traits}\n"
        user_prompt += (
            f"\n<QUESTION>\n"
            f"Answer the following question based on the context and memories above.\n"
            f"Question: {question}\n"
            f"Provide a concise answer in English (maximum 100 words).\n"
            f"Respond in JSON format with key \"answer\".\n"
        )
        return sys_prompt, user_prompt

    def _build_qa_prompt_supportive(
        self,
        question:     str,
        memories:     str,
        user_traits:  str,
        agent_traits: str = "",
        context:      str = "",
    ) -> Tuple[str, str]:
        """Prompt for supportive subset: answer must be yes, no, or unknown."""
        sys_prompt = (
            "You are a helpful assistant that answers yes/no questions about a user "
            "based on recorded conversation memories. "
            "Answer with exactly one of: \"yes\" or \"no\".\n"
            "- \"yes\" if the memories support the statement in the question.\n"
            "- \"no\" if the memories contradict or do not support the statement."
        )
        if agent_traits:
            sys_prompt += f"\nYour personal traits as the assistant: {agent_traits}."
        user_prompt = ""
        if context:
            user_prompt += (
                f"<CONTEXT>\n"
                f"Recent conversation turns:\n{context}\n\n"
            )
        user_prompt += (
            f"<MEMORIES>\n"
            f"The following are conversation memories about the user:\n{memories}\n"
        )
        if user_traits:
            user_prompt += f"\n<USER_TRAITS>\nUser characteristics:\n{user_traits}\n"
        user_prompt += (
            f"\n<QUESTION>\n"
            f"Based on the context and memories above, answer the following question.\n"
            f"Question: {question}\n"
            f"Answer with exactly one of: \"yes\" or \"no\".\n"
            f"Respond in JSON format with key \"answer\".\n"
        )
        return sys_prompt, user_prompt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_memories_for_prompt(
    memories: List[Dict[str, Any]],
    current_virtual_seconds: float = 0.0,
) -> str:
    """Format retrieved LTM memories into a string for prompts."""
    if not memories:
        return "No relevant Memories."

    formatted = []
    for mem in memories:
        elapsed_vs = current_virtual_seconds - mem.get("virtual_seconds", 0.0)
        time_desc  = convert_seconds_to_full_time(max(elapsed_vs, 0.0))
        summary    = mem.get("summary", mem.get("dialog", "Unknown"))
        formatted.append(f"{time_desc} ago, {summary}.")

    return "\n".join(formatted)


def format_context_for_prompt(context_memories: List[Dict[str, Any]]) -> str:
    """Format STM context memories into a string for prompts."""
    if not context_memories:
        return "No previous context."
    return "\n".join(
        f"[TURN {m.get('idx', 0)}] : {m.get('dialog', '')}."
        for m in context_memories
    )
