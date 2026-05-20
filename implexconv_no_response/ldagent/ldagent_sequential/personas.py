"""
Persona Extraction Module for LD-Agent

v2 changes:
  - _user_traits_update() and _agent_traits_update() use return_usage=True
    and store token info in last_user_token_info / last_agent_token_info
    so ldagent_module can collect internal token counts each turn.
  - Token info is stored regardless of whether a trait is extracted
    (the API call still happens even for NO_TRAIT responses).

Reference: "Hello Again! LLM-powered Personalized Agent for Long-term Dialogue"
           (Li et al., NAACL 2025)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

TRAIT_SCHEMA = {
    "type": "object",
    "properties": {"trait": {"type": "string"}},
    "required": ["trait"],
}

logger = logging.getLogger(__name__)


class Personas:
    """
    User and Agent persona management module.

    Adapted from original LD-Agent (Personas.py):
    - LLM client interface changed to unified client
    - Prompts kept identical to original
    - max_personas=0 means unlimited
    - traits_update() merge order corrected to match original
    - [v2] return_usage=True on all generate() calls; token info stored
      in last_user_token_info / last_agent_token_info
    """

    def __init__(
        self,
        llm_client,
        logger: logging.Logger,
        usr_name:           str = "User",
        agent_name:         str = "Agent",
        max_user_personas:  int = 0,
        max_agent_personas: int = 0,
    ):
        self.llm_client         = llm_client
        self.logger             = logger
        self.usr_name           = usr_name
        self.agent_name         = agent_name
        self.max_user_personas  = max_user_personas
        self.max_agent_personas = max_agent_personas

        self.user_traits:  List[str] = []
        self.agent_traits: List[str] = []

        # [v2] Token info for the most recent extraction call (zeros if not yet called)
        self.last_user_token_info:  Dict[str, int] = {"input": 0, "output": 0}
        self.last_agent_token_info: Dict[str, int] = {"input": 0, "output": 0}

        # LLMCallLogger — injected externally; None = no logging
        self.llm_logger = None

        logger.info(
            f"Personas initialized "
            f"(max_user={'unlimited' if max_user_personas == 0 else max_user_personas}, "
            f"max_agent={'unlimited' if max_agent_personas == 0 else max_agent_personas})"
        )

    # =========================================================================
    # LLM LOGGER INJECTION
    # =========================================================================

    def set_llm_logger(self, llm_logger):
        self.llm_logger = llm_logger

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _merge_traits(self, traits: List[str], max_n: int) -> str:
        if max_n == 0 or len(traits) <= max_n:
            return "\n".join(traits)
        return "\n".join(traits[-max_n:])

    @staticmethod
    def _extract_token_info(result) -> Dict[str, int]:
        if isinstance(result, dict) and "_usage" in result:
            usage = result["_usage"]
            return {
                "input":  usage.get("prompt_tokens",     0),
                "output": usage.get("completion_tokens", 0),
            }
        return {"input": 0, "output": 0}

    # =========================================================================
    # MAIN UPDATE METHOD
    # =========================================================================

    def traits_update(self, inquiry: str, response: str) -> Tuple[str, str]:
        """
        Update both user and agent personas based on a dialogue turn.

        Execution order (matches original LD-Agent):
          1. _user_traits_update(inquiry)   → user bank updated
          2. merged_user  = join(user_traits)
          3. merged_agent = join(agent_traits)   ← before current response
          4. _agent_traits_update(response) → agent bank updated AFTER merge
        """
        self._user_traits_update(inquiry)
        merged_user_traits  = self._merge_traits(self.user_traits,  self.max_user_personas)
        merged_agent_traits = self._merge_traits(self.agent_traits, self.max_agent_personas)
        self._agent_traits_update(response)
        return merged_user_traits, merged_agent_traits

    def get_current_traits(self) -> Tuple[str, str]:
        merged_user_traits  = self._merge_traits(self.user_traits,  self.max_user_personas)
        merged_agent_traits = self._merge_traits(self.agent_traits, self.max_agent_personas)
        return merged_user_traits, merged_agent_traits

    # =========================================================================
    # PRIVATE EXTRACTION METHODS
    # =========================================================================

    def _user_traits_update(self, sentence: str) -> List[str]:
        """Extract and append user traits; store token info in last_user_token_info."""
        sys_prompt = (
            "You excel at extracting user personal traits from their words, "
            "a renowned local communication expert."
        )
        cot_example = (
            "If no traits can be extracted in the sentence, set trait to 'NO_TRAIT'. "
            "Given you some format examples of traits extraction, such as:\n"
            "1. No, I have no longer serve in the millitary, I had served up the full term "
            "that I signed up for, and now work outside of the millitary.\n"
            "Extracted Traits: 'I now work elsewhere. I used to be in the military.'\n"
            "2. That must a been some kind of endeavor. Its great that people are aware of "
            "issues that arise in their homes, otherwise it can be very problematic in the future.\n"
            "Extracted Traits: 'NO_TRAIT'\n"
        )
        user_prompt = (
            cot_example
            + f"Please extract the personal traits who said this sentence "
              f"(no more than 20 words):\n{sentence}\n"
              f"Respond in JSON format with key \"trait\".\n"
        )

        # Reset before call so stale values are never read on exception
        self.last_user_token_info = {"input": 0, "output": 0}

        try:
            result = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                max_tokens=100,
                temperature=0.7,
                guided_json=TRAIT_SCHEMA,
                return_usage=True,
            )
            self.last_user_token_info = self._extract_token_info(result)
            if self.llm_logger is not None:
                self.llm_logger.log("call_2_user_persona", sys_prompt, user_prompt, result)

            summarized_traits = str(result.get("trait", "")).strip()

            if "NO_TRAIT" not in summarized_traits and len(summarized_traits) > 3:
                self.user_traits.append(summarized_traits)
                self.logger.debug(f"Extracted user trait: {summarized_traits[:50]}...")

        except Exception as e:
            self.logger.error(f"Error extracting user traits: {e}")

        return self.user_traits

    def _agent_traits_update(self, sentence: str) -> List[str]:
        """Extract and append agent traits; store token info in last_agent_token_info."""
        sys_prompt = (
            "You excel at extracting user personal traits from their words, "
            "a renowned local communication expert."
        )
        cot_example = (
            "If no traits can be extracted in the sentence, set trait to 'NO_TRAIT'. "
            "Given you some format examples of traits extraction, such as:\n"
            "1. No, I have no longer serve in the millitary, I had served up the full term "
            "that I signed up for, and now work outside of the millitary.\n"
            "Extracted Traits: 'I now work elsewhere. I used to be in the military.'\n"
            "2. That must a been some kind of endeavor. Its great that people are aware of "
            "issues that arise in their homes, otherwise it can be very problematic in the future.\n"
            "Extracted Traits: 'NO_TRAIT'\n"
        )
        user_prompt = (
            cot_example
            + f"Please extract the personal traits who said this sentence "
              f"(no more than 20 words):\n{sentence}\n"
              f"Respond in JSON format with key \"trait\".\n"
        )

        self.last_agent_token_info = {"input": 0, "output": 0}

        try:
            result = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                max_tokens=100,
                temperature=0.7,
                guided_json=TRAIT_SCHEMA,
                return_usage=True,
            )
            self.last_agent_token_info = self._extract_token_info(result)
            if self.llm_logger is not None:
                self.llm_logger.log("call_3_agent_persona", sys_prompt, user_prompt, result)

            summarized_traits = str(result.get("trait", "")).strip()

            if "NO_TRAIT" not in summarized_traits and len(summarized_traits) > 3:
                self.agent_traits.append(summarized_traits)
                self.logger.debug(f"Extracted agent trait: {summarized_traits[:50]}...")

        except Exception as e:
            self.logger.error(f"Error extracting agent traits: {e}")

        return self.agent_traits

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def clear(self):
        self.user_traits          = []
        self.agent_traits         = []
        self.last_user_token_info  = {"input": 0, "output": 0}
        self.last_agent_token_info = {"input": 0, "output": 0}
        self.logger.info("Persona banks cleared")

    def save_snapshot(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "user_traits":        self.user_traits,
            "agent_traits":       self.agent_traits,
            "max_user_personas":  self.max_user_personas,
            "max_agent_personas": self.max_agent_personas,
        }
        with open(directory / "personas.json", "w") as f:
            json.dump(snapshot, f, indent=2)
        self.logger.info(f"Persona snapshot saved to {directory}")

    def load_snapshot(self, directory: Path):
        directory     = Path(directory)
        snapshot_file = directory / "personas.json"
        if snapshot_file.exists():
            with open(snapshot_file, "r") as f:
                snapshot = json.load(f)
            self.user_traits  = snapshot.get("user_traits",  [])
            self.agent_traits = snapshot.get("agent_traits", [])
            self.logger.info(f"Persona snapshot loaded from {directory}")

    def get_user_trait_count(self)  -> int: return len(self.user_traits)
    def get_agent_trait_count(self) -> int: return len(self.agent_traits)
    def get_all_user_traits(self)   -> List[str]: return self.user_traits.copy()
    def get_all_agent_traits(self)  -> List[str]: return self.agent_traits.copy()
