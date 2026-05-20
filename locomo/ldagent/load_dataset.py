"""
Dataset Loader for LoComo Dataset

Dataset Schema:
---------------
[
    {
        "sample_id": str,
        "conversation": {
            "speaker_a": str,
            "speaker_b": str,
            "session_1_date_time": str,
            "session_1": [
                {
                    "speaker": str,
                    "dia_id": str,
                    "text": str,
                    # optional image fields:
                    "img_url": list,
                    "blip_caption": str,
                    "query": str
                },
                ...
            ],
            "session_2_date_time": str,
            "session_2": [...],
            ...
        },
        "qa": [
            {
                "question": str,
                "answer": str,              # present for category 1-4
                "adversarial_answer": str,  # present for category 5
                "evidence": [str],          # list of dia_ids
                "category": int             # 1-5
            },
            ...
        ],
        "event_summary": {...},   # parsed but not used by experiment runner
        "observation": {...},     # parsed but not used by experiment runner
        "session_summary": {...}  # parsed but not used by experiment runner
    },
    ...
]

Usage:
------
    from load_dataset import load_locomo_dataset

    samples = load_locomo_dataset(file_path)

    for sample in samples:
        for session in sample.sessions:
            for turn in session.turns:
                print(turn.speaker, turn.dia_id, turn.text)
        for qa in sample.qa:
            print(qa.question, qa.final_answer, qa.category)
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Turn:
    """Single utterance in a session.

    Images (if any) are pre-processed at load time:
    blip_caption is prepended as '[Image: caption]' to the text field.
    img_url is discarded. Callers see only the final text string.
    """
    speaker: str
    dia_id: str
    text: str       # may include '[Image: caption]' prefix if original turn had an image

    def to_memory_content(self, prefix_format: str = "Speaker {speaker} says: {text}") -> str:
        """
        Format turn for memory storage.

        Default matches AMEM original: 'Speaker Caroline says: Hey Mel!'
        Other modules may pass a different prefix_format, e.g. '{speaker}: {text}'.
        Pass prefix_format='{text}' to store without any speaker prefix.
        """
        return prefix_format.format(speaker=self.speaker, text=self.text)


@dataclass
class QAPair:
    """
    QA pair for evaluation.

    - category 1-4: 'answer' is the ground truth, 'adversarial_answer' is None.
    - category 5 (Adversarial): 'adversarial_answer' is the ground truth,
      'answer' may be absent (None). The model must choose between
      adversarial_answer and 'Not mentioned in the conversation'.
    """
    question: str
    category: int
    evidence: List[str] = field(default_factory=list)   # list of dia_ids
    answer: Optional[str] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Ground truth answer for evaluation.

        Returns adversarial_answer for category 5, answer otherwise.
        """
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


@dataclass
class LoCoMoSession:
    """One session within a sample (a single conversation on one date)."""
    session_id: int
    date_time: str          # raw string, e.g. "1:56 pm on 8 May, 2023"
    turns: List[Turn] = field(default_factory=list)


@dataclass
class Sample:
    """
    A single LoComo sample: one pair of speakers across multiple sessions.

    event_summary, observation, session_summary are loaded as raw dicts.
    They are not used by the experiment runner but are available for
    memory modules that choose to exploit them.
    """
    sample_id: str
    sessions: List[LoCoMoSession]       # ordered by session_id ascending
    qa: List[QAPair]
    speaker_a: str
    speaker_b: str
    event_summary: Dict = field(default_factory=dict)
    observation: Dict = field(default_factory=dict)
    session_summary: Dict = field(default_factory=dict)


# =============================================================================
# PARSING HELPERS
# =============================================================================

def _parse_turn(turn_data: dict) -> Turn:
    """Parse one turn dict, converting image fields to text if present."""
    text = turn_data.get("text", "")

    if "img_url" in turn_data and "blip_caption" in turn_data:
        caption = f"[Image: {turn_data['blip_caption']}]"
        text = f"{caption} {text}" if text else caption

    return Turn(
        speaker=turn_data["speaker"],
        dia_id=turn_data["dia_id"],
        text=text,
    )


def _parse_sessions(conv_data: dict) -> List[LoCoMoSession]:
    """Extract and parse all session_N / session_N_date_time pairs."""
    sessions = []
    session_id = 1
    while True:
        session_key = f"session_{session_id}"
        date_key = f"session_{session_id}_date_time"
        if session_key not in conv_data:
            break
        raw_turns = conv_data[session_key]
        if not isinstance(raw_turns, list):
            session_id += 1
            continue
        date_time = conv_data.get(date_key, "")
        turns = [_parse_turn(t) for t in raw_turns]
        if turns:
            sessions.append(LoCoMoSession(
                session_id=session_id,
                date_time=date_time,
                turns=turns,
            ))
        session_id += 1
    return sessions


def _parse_qa(qa_data: dict) -> QAPair:
    return QAPair(
        question=qa_data["question"],
        category=int(qa_data["category"]),
        evidence=qa_data.get("evidence", []),
        answer=qa_data.get("answer"),
        adversarial_answer=qa_data.get("adversarial_answer"),
    )


def _parse_sample(sample_data: dict, sample_idx: int) -> Sample:
    conv = sample_data["conversation"]
    sessions = _parse_sessions(conv)
    qa_list = [_parse_qa(q) for q in sample_data.get("qa", [])]
    return Sample(
        sample_id=str(sample_data.get("sample_id", sample_idx)),
        sessions=sessions,
        qa=qa_list,
        speaker_a=conv.get("speaker_a", ""),
        speaker_b=conv.get("speaker_b", ""),
        event_summary=sample_data.get("event_summary", {}),
        observation=sample_data.get("observation", {}),
        session_summary=sample_data.get("session_summary", {}),
    )


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_locomo_dataset(file_path: str) -> List[Sample]:
    """
    Load the LoComo dataset from a JSON file.

    Args:
        file_path: Path to the dataset JSON file (e.g. 'dataset/locomo10.json').

    Returns:
        List of Sample objects, one per conversation pair, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    print(f"Loading LoComo dataset from: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [_parse_sample(s, idx) for idx, s in enumerate(data)]
    _print_statistics(samples)
    return samples


def _print_statistics(samples: List[Sample]) -> None:
    total_sessions = sum(len(s.sessions) for s in samples)
    total_turns = sum(
        len(sess.turns) for s in samples for sess in s.sessions
    )
    total_qa = sum(len(s.qa) for s in samples)
    cat_counts: Dict[int, int] = {}
    for s in samples:
        for qa in s.qa:
            cat_counts[qa.category] = cat_counts.get(qa.category, 0) + 1

    print(f"  Samples   : {len(samples)}")
    print(f"  Sessions  : {total_sessions}")
    print(f"  Turns     : {total_turns}")
    print(f"  QA pairs  : {total_qa}")
    for cat in sorted(cat_counts):
        print(f"    Category {cat}: {cat_counts[cat]}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import importlib.util as _ilu
    _cfg_file = Path(__file__).parent / "config_0.py"
    _spec = _ilu.spec_from_file_location("config_0", _cfg_file)
    _cfg = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg)

    samples = load_locomo_dataset(_cfg.DATASET_PATH)

    s = samples[0]
    print(f"\nSample '{s.sample_id}'  ({s.speaker_a} \u2194 {s.speaker_b})")
    print(f"  Sessions: {len(s.sessions)}")

    sess = s.sessions[0]
    print(f"\n  Session 1 [{sess.date_time}] \u2014 first 3 turns:")
    for turn in sess.turns[:3]:
        print(f"    [{turn.dia_id}] {turn.speaker}: {turn.text[:70]}")
        print(f"           memory: {turn.to_memory_content()[:70]}")

    print(f"\n  First QA:")
    qa = s.qa[0]
    print(f"    Q (cat {qa.category}): {qa.question}")
    print(f"    A: {qa.final_answer}")
    print(f"    Evidence: {qa.evidence}")
