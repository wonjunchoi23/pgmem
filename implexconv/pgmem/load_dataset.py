"""
Dataset Loader for ImplexConv Dataset

Dataset Schema:
---------------
[
    {
        'metadata': {
            'session_id': int,
            'total_conversations': int,
            'total_turns': int
        },
        'conversations': [
            {
                'session_id': int,
                'conv_id': int,
                'turn_id': int,
                'global_turn_id': int,
                'speaker': str,   # "user" or "assistant"
                'utterance': str
            },
            ...
        ],
        'qa': [
            {
                'question': str,
                'answer': str,
                'opposed_implicit_reasoning': str,  # opposed subset only
                'retrieved_conv_ids': List[str]
            },
            ...
        ]
    },
    ...
]

Usage:
------
    from load_dataset import load_implexconv_dataset

    sessions = load_implexconv_dataset(file_path, "opposed")   # or "supportive"

    for session in sessions:
        for user_turn, assistant_turn in session.get_turn_pairs():
            ...
"""

import json
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Turn:
    """Single utterance in a session."""
    session_id: int
    conv_id: int
    turn_id: int
    global_turn_id: int
    role: str           # 'user' or 'assistant'
    utterance: str

    def to_message(self) -> str:
        """Format as 'User: ...' or 'Assistant: ...'."""
        role_label = "User" if self.role == "user" else "Assistant"
        return f"{role_label}: {self.utterance}"

@dataclass
class QAPair:
    """
    QA pair for evaluation.

    - opposed subset: answer is a free-form string;
      opposed_implicit_reasoning is populated.
    - supportive subset: answer is one of {yes, no, i don't know};
      opposed_implicit_reasoning is always "".
    """
    question: str
    answer: str
    retrieved_conv_ids: List[str] = field(default_factory=list)
    opposed_implicit_reasoning: str = ""


@dataclass
class Session:
    """Session containing all utterances (flattened) and QA pairs."""
    session_id: int
    total_conversations: int
    total_turns: int
    turns: List[Turn] = field(default_factory=list)
    qa: List[QAPair] = field(default_factory=list)

    @property
    def user_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def assistant_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def get_turn_pairs(self) -> List[Tuple[Turn, Optional[Turn]]]:
        """
        Return list of (user_turn, assistant_turn) pairs in order.
        Assumes turns alternate user → assistant.
        """
        pairs = []
        i = 0
        while i < len(self.turns):
            user_turn = None
            assistant_turn = None

            if i < len(self.turns) and self.turns[i].role == "user":
                user_turn = self.turns[i]
                i += 1

            if i < len(self.turns) and self.turns[i].role == "assistant":
                assistant_turn = self.turns[i]
                i += 1

            if user_turn is not None:
                pairs.append((user_turn, assistant_turn))

        return pairs

    def get_turns_by_conv_id(self, conv_id: int) -> List[Turn]:
        return [t for t in self.turns if t.conv_id == conv_id]

    def get_unique_conv_ids(self) -> List[int]:
        seen = set()
        result = []
        for turn in self.turns:
            if turn.conv_id not in seen:
                seen.add(turn.conv_id)
                result.append(turn.conv_id)
        return result


# =============================================================================
# PARSING
# =============================================================================

def _parse_turn(turn_data: dict) -> Turn:
    return Turn(
        session_id=turn_data["session_id"],
        conv_id=turn_data["conv_id"],
        turn_id=turn_data["turn_id"],
        global_turn_id=turn_data["global_turn_id"],
        role=turn_data["speaker"],      # already "user" or "assistant"
        utterance=turn_data["utterance"],
    )


def _parse_qa(qa_data: dict, subset: str) -> QAPair:
    return QAPair(
        question=qa_data["question"],
        answer=qa_data["answer"],
        retrieved_conv_ids=qa_data.get("retrieved_conv_ids", []),
        opposed_implicit_reasoning=(
            qa_data.get("opposed_implicit_reasoning", "")
            if subset == "opposed" else ""
        ),
    )


def _parse_session(session_data: dict, subset: str) -> Session:
    metadata = session_data["metadata"]
    turns = [_parse_turn(t) for t in session_data.get("conversations", [])]
    qa_pairs = [_parse_qa(q, subset) for q in session_data.get("qa", [])]

    return Session(
        session_id=metadata["session_id"],
        total_conversations=metadata["total_conversations"],
        total_turns=metadata.get("total_turns", len(turns)),
        turns=turns,
        qa=qa_pairs,
    )


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_implexconv_dataset(file_path: str, subset: str) -> List[Session]:
    """
    Load ImplexConv dataset for the given subset.

    Args:
        file_path: Path to the dataset JSON file.
        subset: "opposed" or "supportive"

    Returns:
        List of Session objects, ordered by session_id.

    Raises:
        ValueError: If subset is not "opposed" or "supportive".
        FileNotFoundError: If the dataset file does not exist.
    """
    if subset not in ("opposed", "supportive"):
        raise ValueError(f"Unknown subset '{subset}'. Expected 'opposed' or 'supportive'.")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    print(f"Loading '{subset}' dataset from: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = [_parse_session(s, subset) for s in data]
    _print_statistics(sessions, subset)
    return sessions


def _print_statistics(sessions: List[Session], subset: str):
    total_turns = sum(len(s.turns) for s in sessions)
    total_qa = sum(len(s.qa) for s in sessions)
    print(f"  Subset       : {subset}")
    print(f"  Sessions     : {len(sessions)}")
    print(f"  Total turns  : {total_turns}")
    print(f"  Total QA     : {total_qa}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import importlib.util as _ilu
    _cfg_file = Path(__file__).parent / "config.py"
    _spec = _ilu.spec_from_file_location("config", _cfg_file)
    _cfg = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg)

    _paths = {"opposed": _cfg.DATASET_OPPOSED, "supportive": _cfg.DATASET_SUPPORTIVE}
    for subset in ("opposed", "supportive"):
        print(f"\n{'='*50}")
        sessions = load_implexconv_dataset(_paths[subset], subset)

        s = sessions[0]
        print(f"\nSession 0 — first 4 turns:")
        for turn in s.turns[:4]:
            print(f"  [{turn.conv_id}/{turn.turn_id}] {turn.to_message()[:70]}")

        print(f"\nSession 0 — first QA:")
        q = s.qa[0]
        print(f"  Q: {q.question}")
        print(f"  A: {q.answer}")
        if q.opposed_implicit_reasoning:
            print(f"  Reasoning: {q.opposed_implicit_reasoning[:60]}")
