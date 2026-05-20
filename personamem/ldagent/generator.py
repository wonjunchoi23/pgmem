"""
Response Generator for LD-Agent — PersonaMem variant

Phase 2 multiple-choice QA only. Response generation is removed.

QA prompt (Q4 option (A) — 4-section LD-Agent skeleton + multiple-choice suffix):
  [SYSTEM] agent_traits
  [USER]
    <CONTEXT>      Recent STM turns
    <MEMORIES>     LTM-retrieved memories (formatted)
    <USER_TRAITS>  User trait bank
    <QUESTION>     Question + Options + "Choose a/b/c/d"
"""

import logging
from typing import List, Dict, Tuple, Any

from load_dataset import convert_seconds_to_full_time

logger = logging.getLogger(__name__)


# =============================================================================
# JSON SCHEMA
# =============================================================================

QA_SCHEMA_MULTICHOICE = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["a", "b", "c", "d"]},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


# =============================================================================
# GENERATOR CLASS
# =============================================================================

class Generator:
    def __init__(
        self,
        llm_client,
        logger: logging.Logger,
        usr_name:    str   = "User",
        agent_name:  str   = "Assistant",
        max_tokens:  int   = 750,
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
        self.llm_logger  = None
        logger.info(f"Generator initialized (usr={usr_name}, agent={agent_name})")

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

    # =========================================================================
    # QA PROMPT
    # =========================================================================

    def _build_qa_prompt_multichoice(
        self,
        question:     str,
        memories:     str,
        options:      List[str],
        user_traits:  str = "",
        agent_traits: str = "",
        context:      str = "",
    ) -> Tuple[str, str]:
        sys_prompt = (
            "You are a helpful assistant answering a multiple-choice question "
            "about a user based on recorded conversation memories."
        )
        if agent_traits:
            sys_prompt += f" Your personal traits as the assistant: {agent_traits}."

        user_prompt = ""
        if context:
            user_prompt += f"<CONTEXT>\nRecent conversation turns:\n{context}\n\n"
        user_prompt += (
            f"<MEMORIES>\nThe following are conversation memories about the user:\n"
            f"{memories}\n"
        )
        if user_traits:
            user_prompt += f"\n<USER_TRAITS>\nUser characteristics:\n{user_traits}\n"

        options_text = "\n".join(options)
        user_prompt += (
            f"\n<QUESTION>\n"
            f"Answer the following question based on the context, memories, and traits above.\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Choose the single best answer (a, b, c, or d) and respond in JSON format "
            f"with key \"answer\" containing only the letter.\n"
        )
        return sys_prompt, user_prompt

    def build_qa_prompt(
        self,
        question:     str,
        memories:     str,
        options:      List[str],
        user_traits:  str = "",
        agent_traits: str = "",
        context:      str = "",
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        sys_prompt, user_prompt = self._build_qa_prompt_multichoice(
            question, memories, options, user_traits, agent_traits, context
        )
        prompt_snapshot = f"[SYSTEM]\n{sys_prompt}\n[USER]\n{user_prompt}"
        return sys_prompt, user_prompt, prompt_snapshot, QA_SCHEMA_MULTICHOICE

    # =========================================================================
    # ANSWER NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_letter(answer: str) -> str:
        label = (answer or "").strip().lower()
        return label if label in ("a", "b", "c", "d") else "unknown"

    # =========================================================================
    # SEQUENTIAL FALLBACK PATH
    # =========================================================================

    def generate_qa_answer(
        self,
        question:     str,
        memories:     str,
        options:      List[str],
        user_traits:  str = "",
        agent_traits: str = "",
        context:      str = "",
    ) -> Tuple[str, Dict[str, int], str, int]:
        sys_prompt, user_prompt, prompt_snapshot, schema = self.build_qa_prompt(
            question=question,
            memories=memories,
            options=options,
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
            answer = self.normalize_letter(answer)

        except Exception as e:
            self.logger.error(f"Error in generate_qa_answer: {e}")
            return "unknown", token_info, prompt_snapshot, 0

        return answer, token_info, prompt_snapshot, 1


# =============================================================================
# UTILITY
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
