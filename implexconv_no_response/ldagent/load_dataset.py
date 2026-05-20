"""
Dataset Loader for ImplexConv Dataset

Loads the ImplexConv dataset (opposed / supportive subsets).

Dataset schema per session:
{
    "metadata": {"session_id": int, "total_conversations": int, "total_turns": int},
    "conversations": [
        {"session_id": int, "conv_id": int, "turn_id": int, "global_turn_id": int,
         "speaker": str,  # "user" or "assistant"
         "utterance": str},
        ...
    ],
    "qa": [
        {"question": str, "answer": str,
         "opposed_implicit_reasoning": str,
         "retrieved_conv_ids": [str, ...]},
        ...
    ]
}

Changes from ldagent/load_dataset.py:
  - parse_role(): updated for new dataset where speaker is "user" or "assistant" directly
    (old dataset had a word "speaker" embedded in the speaker field)
  - generate_session_pairs() / get_session_pair_by_index() removed (not needed)
  - load_implexconv_dataset() hardcoded default path removed
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Turn:
    """Represents a single turn in a conversation."""
    session_id: int
    conv_id:    int
    turn_id:    int
    role:       str   # 'user' or 'assistant'
    utterance:  str
    original_speaker: str = ""

    def to_memory_content(self) -> str:
        speaker_name = self.original_speaker if self.original_speaker else self.role.capitalize()
        return f"Speaker {speaker_name} says : {self.utterance}"


@dataclass
class QAPair:
    """Represents a QA pair for evaluation."""
    question:                   str
    answer:                     str
    opposed_implicit_reasoning: str       = ""
    retrieved_conv_ids:         List[str] = field(default_factory=list)


@dataclass
class Session:
    """Represents a session containing multiple conversation turns and QA pairs."""
    session_id:          int
    total_conversations: int
    total_turns:         int
    turns: List[Turn]    = field(default_factory=list)
    qa:    List[QAPair]  = field(default_factory=list)

    @property
    def user_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == 'user']

    @property
    def assistant_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == 'assistant']

    def get_turn_pairs(self) -> List[Tuple[Turn, Optional[Turn]]]:
        """
        Return (user_turn, assistant_turn) pairs in order.
        assistant_turn is None if a user turn has no following assistant turn.
        """
        pairs = []
        i = 0
        while i < len(self.turns):
            if self.turns[i].role == 'user':
                user_turn = self.turns[i]
                if i + 1 < len(self.turns) and self.turns[i + 1].role == 'assistant':
                    assistant_turn = self.turns[i + 1]
                    i += 2
                else:
                    assistant_turn = None
                    i += 1
                pairs.append((user_turn, assistant_turn))
            else:
                i += 1
        return pairs


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_role(speaker: str) -> str:
    """
    Map speaker field to 'user' or 'assistant'.

    New dataset: speaker is "user" or "assistant" directly.
    (Old ldagent dataset had speaker fields containing the word "speaker" for user turns.)
    """
    if 'user' in speaker.lower():
        return 'user'
    return 'assistant'


def parse_turn(turn_data: dict) -> Turn:
    speaker = turn_data.get('speaker', '')
    role    = parse_role(speaker)
    return Turn(
        session_id=turn_data.get('session_id', 0),
        conv_id=turn_data.get('conv_id', 0),
        turn_id=turn_data.get('turn_id', 0),
        role=role,
        utterance=turn_data.get('utterance', ''),
        original_speaker=speaker,
    )


def parse_qa(qa_data: dict) -> QAPair:
    return QAPair(
        question=qa_data.get('question', ''),
        answer=qa_data.get('answer', ''),
        opposed_implicit_reasoning=qa_data.get('opposed_implicit_reasoning', ''),
        retrieved_conv_ids=qa_data.get('retrieved_conv_ids', []),
    )


def parse_session(session_data: dict) -> Session:
    metadata  = session_data.get('metadata', {})
    turns     = [parse_turn(t) for t in session_data.get('conversations', [])]
    qa_pairs  = [parse_qa(q) for q in session_data.get('qa', [])]
    return Session(
        session_id=metadata.get('session_id', 0),
        total_conversations=metadata.get('total_conversations', 0),
        total_turns=metadata.get('total_turns', len(turns)),
        turns=turns,
        qa=qa_pairs,
    )


# =============================================================================
# VIRTUAL TIME UTILITIES
# =============================================================================

def compute_virtual_seconds(
    conv_id: int,
    turn_id: int,
    conv_ids_per_day: int = 2,
    minutes_per_turn: int = 10,
) -> float:
    """
    Compute the virtual time (in seconds from session start) for a turn.

    Model:
      - conv_ids_per_day consecutive conv_ids = 1 virtual day (1440 minutes)
      - turn_id within a conv = minutes offset (turn_id × minutes_per_turn)

    Args:
        conv_id:          Conversation ID of the turn.
        turn_id:          Local turn ID within the conversation.
        conv_ids_per_day: How many conv_ids constitute one virtual day.
        minutes_per_turn: Minutes per local turn_id step.

    Returns:
        Virtual elapsed seconds from session start.
    """
    minutes_per_day  = 1440.0
    minutes_per_half = minutes_per_day / conv_ids_per_day
    day  = conv_id // conv_ids_per_day
    half = conv_id % conv_ids_per_day
    virtual_minutes  = day * minutes_per_day + half * minutes_per_half + turn_id * minutes_per_turn
    return virtual_minutes * 60.0


def convert_seconds_to_full_time(seconds: float) -> str:
    """
    Convert elapsed virtual seconds to a human-readable string.

    Ported from original LD-Agent MSC.py:
    'XX years XX months XX days XX hours XX minutes'
    (only non-zero units included)
    """
    units = [
        ("years",   31536000),
        ("months",   2592000),
        ("days",       86400),
        ("hours",       3600),
        ("minutes",       60),
    ]
    parts   = []
    remaining = int(seconds)
    for name, count in units:
        value, remaining = divmod(remaining, count)
        if value:
            parts.append(f"{value} {name}")
    return " ".join(parts) if parts else "just now"


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_implexconv_dataset(file_path: str) -> List[Session]:
    """
    Load ImplexConv dataset from a JSON file.

    Args:
        file_path: Path to ImplexConv_opposed_processed.json or
                   ImplexConv_supportive_processed.json

    Returns:
        List of Session objects
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    print(f"Loading dataset from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sessions = [parse_session(s) for s in data]
    _print_statistics(sessions)
    return sessions


def _print_statistics(sessions: List[Session]):
    total_turns     = sum(s.total_turns for s in sessions)
    total_user      = sum(len(s.user_turns) for s in sessions)
    total_assistant = sum(len(s.assistant_turns) for s in sessions)
    total_qa        = sum(len(s.qa) for s in sessions)

    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total sessions:            {len(sessions)}")
    print(f"Total turns:               {total_turns}")
    print(f"  - User turns:            {total_user}")
    print(f"  - Assistant turns:       {total_assistant}")
    print(f"Total QA pairs:            {total_qa}")
    print(f"Avg turns per session:     {total_turns / max(len(sessions), 1):.1f}")
    print(f"Avg QA per session:        {total_qa / max(len(sessions), 1):.1f}")
    print("=" * 50 + "\n")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "dataset/implexconv/ImplexConv_opposed_processed.json"
    try:
        sessions = load_implexconv_dataset(path)
        if sessions:
            s = sessions[0]
            print(f"\nSample session {s.session_id}: {s.total_turns} turns, {len(s.qa)} QA")
            if s.turns:
                t = s.turns[0]
                print(f"  First turn: [{t.role}] {t.utterance[:80]}...")
    except FileNotFoundError as e:
        print(f"Error: {e}")
