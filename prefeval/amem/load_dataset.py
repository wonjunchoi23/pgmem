"""
Dataset Loader for PrefEval (implicit-persona) Dataset.

Each PrefEval sample becomes one Session in the chain. For chain position
`chain_pos` (= sample index in the chain, 0..K), the loader assigns:

  - session_id      = 0                         # one chain = one logical session
  - conv_id         = chain_pos                 # PrefEval sample index in chain
  - turn_id         = local utterance index     # user→2k, assistant→2k+1
  - global_turn_id  = cumulative across chain   # never resets

Raw PrefEval `conversation` dict stores `assistant` first then `user` per turn,
but the conversational order is user→assistant. Reorder accordingly.

The `question`, `preference`, `explanation`, `persona`, `topic` fields are NOT
fed into the memory module. They live on Session for evaluator-side use only.
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
    """Single utterance in a session."""
    session_id:     int
    conv_id:        int
    turn_id:        int
    global_turn_id: int
    role:           str   # 'user' or 'assistant'
    utterance:      str

    def to_message(self) -> str:
        role_label = "User" if self.role == "user" else "Assistant"
        return f"{role_label}: {self.utterance}"

    def to_memory_content(self) -> str:
        return f"Speaker {self.role} says: {self.utterance}"


@dataclass
class QAPair:
    """
    One QA per PrefEval sample. Ground-truth fields (preference, explanation,
    persona, topic) are kept here for the JSONL output but are never passed
    into the memory module or the answering model.
    """
    question:    str
    preference:  str = ""
    explanation: str = ""
    persona:     str = ""
    topic:       str = ""


@dataclass
class Session:
    """One PrefEval sample mapped to a Session-shaped object."""
    session_id:          int
    conv_id:             int     # chain position
    total_turns:         int
    turns:               List[Turn]   = field(default_factory=list)
    qa:                  List[QAPair] = field(default_factory=list)

    @property
    def user_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def assistant_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def get_turn_pairs(self) -> List[Tuple[Turn, Optional[Turn]]]:
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


# =============================================================================
# PARSING
# =============================================================================

def _parse_sample(
    sample: dict,
    chain_pos: int,
    global_turn_offset: int,
) -> Tuple[Session, int]:
    """
    Convert one PrefEval sample dict into a Session and return (session, next_offset).
    """
    conversation = sample.get("conversation", {}) or {}

    # Sort by integer key so turn order is stable
    keys = sorted((int(k) for k in conversation.keys()))

    turns: List[Turn] = []
    local_turn_id = 0
    g = global_turn_offset

    for k in keys:
        turn_dict = conversation.get(str(k))
        if turn_dict is None:
            continue
        user_text      = (turn_dict.get("user")      or "").strip()
        assistant_text = (turn_dict.get("assistant") or "").strip()

        # User → Assistant order (raw data has assistant first per dict, but the
        # conversational flow is user-first within a turn).
        if user_text:
            turns.append(Turn(
                session_id=0,
                conv_id=chain_pos,
                turn_id=local_turn_id,
                global_turn_id=g,
                role="user",
                utterance=user_text,
            ))
            local_turn_id += 1
            g += 1
        if assistant_text:
            turns.append(Turn(
                session_id=0,
                conv_id=chain_pos,
                turn_id=local_turn_id,
                global_turn_id=g,
                role="assistant",
                utterance=assistant_text,
            ))
            local_turn_id += 1
            g += 1

    qa = QAPair(
        question=sample.get("question", ""),
        preference=sample.get("preference", ""),
        explanation=sample.get("explanation", ""),
        persona=sample.get("persona", ""),
        topic=sample.get("topic", ""),
    )

    session = Session(
        session_id=0,
        conv_id=chain_pos,
        total_turns=len(turns),
        turns=turns,
        qa=[qa],
    )
    return session, g


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_prefeval_chain(file_path: str, end_session: int) -> List[Session]:
    """
    Load PrefEval samples [0..end_session] and convert them into a list of
    Session objects (one per chain position).

    Args:
        file_path:   path to dataset/implicit_persona.json
        end_session: K — chain length is K+1 (samples[0..K])

    Returns:
        List of Session objects, indexed 0..K.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if end_session < 0:
        raise ValueError(f"end_session must be >= 0, got {end_session}")

    print(f"Loading PrefEval from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if end_session >= len(data):
        raise ValueError(
            f"end_session={end_session} out of range "
            f"(dataset has {len(data)} samples, max index {len(data)-1})"
        )

    sessions: List[Session] = []
    g_offset = 0
    for chain_pos in range(end_session + 1):
        session, g_offset = _parse_sample(data[chain_pos], chain_pos, g_offset)
        sessions.append(session)

    _print_statistics(sessions)
    return sessions


def _print_statistics(sessions: List[Session]):
    total_turns = sum(len(s.turns) for s in sessions)
    print(f"  Chain length : {len(sessions)} sessions (chain_pos 0..{len(sessions)-1})")
    print(f"  Total turns  : {total_turns}")
    print(f"  Total QA     : {len(sessions)}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import importlib.util as _ilu
    _cfg_file = Path(__file__).parent / "config_0.py"
    _spec = _ilu.spec_from_file_location("config_0", _cfg_file)
    _cfg = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg)

    sessions = load_prefeval_chain(_cfg.DATASET_PATH, end_session=2)
    for s in sessions:
        print(f"\n--- chain_pos={s.conv_id} ({s.total_turns} turns) ---")
        for t in s.turns[:4]:
            print(f"  [c{t.conv_id}/t{t.turn_id}/g{t.global_turn_id}] {t.role}: {t.utterance[:60]}")
        q = s.qa[0]
        print(f"  question: {q.question[:80]}")
        print(f"  topic   : {q.topic}")
