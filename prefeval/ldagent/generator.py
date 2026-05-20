"""
LD-Agent Generator — PrefEval port.

Trimmed from exp_implexconv_no_response/ldagent/generator.py:
- Removed `subset` (no opposed/supportive split).
- Removed _build_qa_prompt_supportive and QA_SCHEMA_SUPPORTIVE.
- Removed response_build_json (response generation unused in QA-only protocol).
- QA word cap: 100 → 200 words.

Otherwise unchanged: same llm_client.generate() flow, same JSON schema,
same token tracking and per-call logging.
"""

import logging
from typing import List, Dict, Tuple, Any

from load_dataset import convert_seconds_to_full_time

logger = logging.getLogger(__name__)


# =============================================================================
# JSON SCHEMA
# =============================================================================

QA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


# =============================================================================
# GENERATOR CLASS
# =============================================================================

class Generator:
    """LD-Agent QA generator (Phase 2 only; no response generation)."""

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

        self.llm_logger = None

        logger.info(f"Generator initialized (usr={usr_name}, agent={agent_name})")

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

    # =========================================================================
    # QA GENERATION
    # =========================================================================

    def generate_qa_answer(
        self,
        question:     str,
        memories:     str,
        user_traits:  str = "",
        agent_traits: str = "",
        context:      str = "",
    ) -> Tuple[str, Dict[str, int], str, int]:
        """
        Generate a QA answer based on accumulated LTM + STM context.

        Returns:
            (answer, token_info, prompt_snapshot, num_llm_calls)
        """
        sys_prompt, user_prompt, prompt_snapshot, schema = self.build_qa_prompt(
            question=question,
            memories=memories,
            user_traits=user_traits,
            agent_traits=agent_traits,
            context=context,
        )
        token_info = {"input": 0, "output": 0}

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

    def build_qa_prompt(
        self,
        question: str,
        memories: str,
        user_traits: str = "",
        agent_traits: str = "",
        context: str = "",
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        sys_prompt, user_prompt = self._build_qa_prompt(
            question, memories, user_traits, agent_traits, context,
        )
        prompt_snapshot = f"[SYSTEM]\n{sys_prompt}\n[USER]\n{user_prompt}"
        return sys_prompt, user_prompt, prompt_snapshot, QA_RESPONSE_SCHEMA

    def _build_qa_prompt(
        self,
        question:     str,
        memories:     str,
        user_traits:  str,
        agent_traits: str = "",
        context:      str = "",
    ) -> Tuple[str, str]:
        """Free-form QA answer (max 200 words)."""
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
            f"Provide a concise answer in English (maximum 200 words).\n"
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
