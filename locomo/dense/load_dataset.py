"""
Dataset Loader for LoCoMo Dataset
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str

    def to_memory_content(self) -> str:
        return f"Speaker {self.speaker} says: {self.text}"


@dataclass
class QAPair:
    question: str
    category: int
    evidence: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


@dataclass
class LoCoMoSession:
    session_id: int
    date_time: str
    turns: List[Turn] = field(default_factory=list)


@dataclass
class Sample:
    sample_id: str
    sessions: List[LoCoMoSession]
    qa: List[QAPair]
    speaker_a: str
    speaker_b: str
    event_summary: Dict = field(default_factory=dict)
    observation: Dict = field(default_factory=dict)
    session_summary: Dict = field(default_factory=dict)


def _parse_turn(turn_data: dict) -> Turn:
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
    sessions: List[LoCoMoSession] = []
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

        turns = [_parse_turn(turn_data) for turn_data in raw_turns]
        if turns:
            sessions.append(
                LoCoMoSession(
                    session_id=session_id,
                    date_time=conv_data.get(date_key, ""),
                    turns=turns,
                )
            )
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
    return Sample(
        sample_id=str(sample_data.get("sample_id", sample_idx)),
        sessions=_parse_sessions(conv),
        qa=[_parse_qa(qa_data) for qa_data in sample_data.get("qa", [])],
        speaker_a=conv.get("speaker_a", ""),
        speaker_b=conv.get("speaker_b", ""),
        event_summary=sample_data.get("event_summary", {}),
        observation=sample_data.get("observation", {}),
        session_summary=sample_data.get("session_summary", {}),
    )


def load_locomo_dataset(file_path: str) -> List[Sample]:
    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading LoCoMo dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [_parse_sample(sample, idx) for idx, sample in enumerate(data)]
    _print_statistics(samples)
    return samples


def _print_statistics(samples: List[Sample]) -> None:
    total_sessions = sum(len(sample.sessions) for sample in samples)
    total_turns = sum(len(session.turns) for sample in samples for session in sample.sessions)
    total_qa = sum(len(sample.qa) for sample in samples)
    category_counts: Dict[int, int] = {}

    for sample in samples:
        for qa in sample.qa:
            category_counts[qa.category] = category_counts.get(qa.category, 0) + 1

    print(f"  Samples   : {len(samples)}")
    print(f"  Sessions  : {total_sessions}")
    print(f"  Turns     : {total_turns}")
    print(f"  QA pairs  : {total_qa}")
    for category in sorted(category_counts):
        print(f"    Category {category}: {category_counts[category]}")


if __name__ == "__main__":
    import importlib.util as _ilu

    config_file = Path(__file__).parent / "config_0.py"
    spec = _ilu.spec_from_file_location("config_0", config_file)
    cfg = _ilu.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    samples = load_locomo_dataset(cfg.DATASET_PATH)
    sample = samples[0]
    print(f"\nSample '{sample.sample_id}'  ({sample.speaker_a} <-> {sample.speaker_b})")
    print(f"  Sessions: {len(sample.sessions)}")

    session = sample.sessions[0]
    print(f"\n  Session 1 [{session.date_time}] - first 3 turns:")
    for turn in session.turns[:3]:
        print(f"    [{turn.dia_id}] {turn.speaker}: {turn.text[:70]}")
        print(f"           memory: {turn.to_memory_content()[:70]}")

    qa = sample.qa[0]
    print(f"\n  First QA:")
    print(f"    Q (cat {qa.category}): {qa.question}")
    print(f"    A: {qa.final_answer}")
    print(f"    Evidence: {qa.evidence}")
