"""
Persona Extraction Module for LD-Agent (LoComo) — Batch-enabled variant

Extends ldagent/personas.py with step-wise batch methods so that trait
extraction LLM calls can be collected across multiple samples and issued
as a single generate_batch_raw() call.

Added constants:
  TRAIT_SYS_PROMPT  — shared system prompt (no speaker names → safe for batch)
  TRAIT_COT_EXAMPLE — CoT example text used in user prompt

Added methods (batch API):
  build_trait_prompt(sentence)      — return (sys_prompt, user_prompt) without calling LLM
  apply_speaker1_trait_result(result, usage) — apply speaker_a extraction result from batch
  apply_speaker2_trait_result(result, usage) — apply speaker_b extraction result from batch

All original sequential methods are preserved unchanged.
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

# Module-level constants for batch usage (no speaker names → shared across samples)
TRAIT_SYS_PROMPT = (
    "You excel at extracting user personal traits from their words, "
    "a renowned local communication expert."
)

TRAIT_COT_EXAMPLE = (
    "If no traits can be extracted in the sentence, set trait to 'NO_TRAIT'. "
    "Given you some format examples of traits extraction, such as:\n"
    "1. No, I have no longer serve in the millitary, I had served up the full term "
    "that I signed up for, and now work outside of the millitary.\n"
    "Extracted Traits: 'I now work elsewhere. I used to be in the military.'\n"
    "2. That must a been some kind of endeavor. Its great that people are aware of "
    "issues that arise in their homes, otherwise it can be very problematic in the future.\n"
    "Extracted Traits: 'NO_TRAIT'\n"
)

logger = logging.getLogger(__name__)


class Personas:
    """
    Speaker persona management for LD-Agent (LoComo batch variant).
    See ldagent/personas.py for full documentation of sequential API.
    """

    def __init__(
        self,
        llm_client,
        logger: logging.Logger,
        speaker_a:              str = "Speaker A",
        speaker_b:              str = "Speaker B",
        max_speaker_a_personas: int = 0,
        max_speaker_b_personas: int = 0,
    ):
        self.llm_client              = llm_client
        self.logger                  = logger
        self.speaker_a               = speaker_a
        self.speaker_b               = speaker_b
        self.max_speaker_a_personas  = max_speaker_a_personas
        self.max_speaker_b_personas  = max_speaker_b_personas

        self.speaker_a_traits: List[str] = []
        self.speaker_b_traits: List[str] = []

        self.last_speaker_a_token_info: Dict[str, int] = {"input": 0, "output": 0}
        self.last_speaker_b_token_info: Dict[str, int] = {"input": 0, "output": 0}

        self.llm_logger = None

        logger.info(
            f"Personas initialized "
            f"(speaker_a={speaker_a}, max={max_speaker_a_personas or 'unlimited'}; "
            f"speaker_b={speaker_b}, max={max_speaker_b_personas or 'unlimited'})"
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

    def _extract_traits(self, sentence: str) -> Tuple[str, Dict[str, int]]:
        """Sequential trait extraction LLM call."""
        sys_prompt  = TRAIT_SYS_PROMPT
        user_prompt = (
            TRAIT_COT_EXAMPLE
            + f"Please extract the personal traits who said this sentence "
              f"(no more than 20 words):\n{sentence}\n"
              f"Respond in JSON format with key \"trait\".\n"
        )

        result = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            max_tokens=100,
            temperature=0.7,
            guided_json=TRAIT_SCHEMA,
            return_usage=True,
        )
        token_info = self._extract_token_info(result)
        trait_str  = str(result.get("trait", "")).strip() if isinstance(result, dict) else ""
        return trait_str, token_info

    def _do_trait_update(
        self, sentence: str, traits_list: List[str], token_attr: str,
        call_type: str, label: str,
    ) -> List[str]:
        setattr(self, token_attr, {"input": 0, "output": 0})
        try:
            trait_str, token_info = self._extract_traits(sentence)
            setattr(self, token_attr, token_info)
            if self.llm_logger is not None:
                self.llm_logger.log(call_type, TRAIT_SYS_PROMPT, sentence,
                                    {"trait": trait_str, "_usage": token_info})
            if "NO_TRAIT" not in trait_str and len(trait_str) > 3:
                traits_list.append(trait_str)
                self.logger.debug(f"Speaker {label} trait: {trait_str[:60]}")
        except Exception as e:
            self.logger.error(f"Error extracting speaker_{label.lower()} traits: {e}")
        return traits_list

    def _do_apply_trait(
        self, result: Dict, usage: Dict, traits_list: List[str],
        token_attr: str, call_type: str, label: str,
    ) -> None:
        token_info = {
            "input":  usage.get("prompt_tokens",     0),
            "output": usage.get("completion_tokens", 0),
        }
        setattr(self, token_attr, token_info)
        trait_str = str(result.get("trait", "")).strip() if isinstance(result, dict) else ""
        if self.llm_logger is not None:
            self.llm_logger.log(call_type, TRAIT_SYS_PROMPT, "",
                                {"trait": trait_str, "_usage": token_info})
        if "NO_TRAIT" not in trait_str and len(trait_str) > 3:
            traits_list.append(trait_str)
            self.logger.debug(f"Speaker {label} trait (batch): {trait_str[:60]}")

    # =========================================================================
    # SEQUENTIAL PUBLIC UPDATE METHODS
    # =========================================================================

    def _speaker_a_traits_update(self, sentence: str) -> List[str]:
        return self._do_trait_update(
            sentence, self.speaker_a_traits,
            "last_speaker_a_token_info", "call_1_speaker1_persona", "A",
        )

    def _speaker_b_traits_update(self, sentence: str) -> List[str]:
        return self._do_trait_update(
            sentence, self.speaker_b_traits,
            "last_speaker_b_token_info", "call_2_speaker2_persona", "B",
        )

    # =========================================================================
    # BATCH API
    # =========================================================================

    def build_trait_prompt(self, sentence: str) -> Tuple[str, str]:
        """Build trait extraction prompt without calling LLM."""
        user_prompt = (
            TRAIT_COT_EXAMPLE
            + f"Please extract the personal traits who said this sentence "
              f"(no more than 20 words):\n{sentence}\n"
              f"Respond in JSON format with key \"trait\".\n"
        )
        return TRAIT_SYS_PROMPT, user_prompt

    def apply_speaker1_trait_result(self, result: Dict, usage: Dict) -> None:
        self._do_apply_trait(result, usage, self.speaker_a_traits,
                             "last_speaker_a_token_info", "call_1_speaker1_persona", "A")

    def apply_speaker2_trait_result(self, result: Dict, usage: Dict) -> None:
        self._do_apply_trait(result, usage, self.speaker_b_traits,
                             "last_speaker_b_token_info", "call_2_speaker2_persona", "B")

    # =========================================================================
    # TRAIT RETRIEVAL
    # =========================================================================

    def get_current_traits(self) -> Tuple[str, str]:
        a = self._merge_traits(self.speaker_a_traits, self.max_speaker_a_personas)
        b = self._merge_traits(self.speaker_b_traits, self.max_speaker_b_personas)
        return a, b

    def get_speaker_a_trait_count(self) -> int: return len(self.speaker_a_traits)
    def get_speaker_b_trait_count(self) -> int: return len(self.speaker_b_traits)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def clear(self):
        self.speaker_a_traits          = []
        self.speaker_b_traits          = []
        self.last_speaker_a_token_info = {"input": 0, "output": 0}
        self.last_speaker_b_token_info = {"input": 0, "output": 0}
        self.logger.info("Persona banks cleared")

    def save_snapshot(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "speaker_a":               self.speaker_a,
            "speaker_b":               self.speaker_b,
            "speaker_a_traits":        self.speaker_a_traits,
            "speaker_b_traits":        self.speaker_b_traits,
            "max_speaker_a_personas":  self.max_speaker_a_personas,
            "max_speaker_b_personas":  self.max_speaker_b_personas,
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
            self.speaker_a_traits = snapshot.get("speaker_a_traits", [])
            self.speaker_b_traits = snapshot.get("speaker_b_traits", [])
            self.logger.info(f"Persona snapshot loaded from {directory}")
