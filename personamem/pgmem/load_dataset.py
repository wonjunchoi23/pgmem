"""
Dataset Loader for PersonaMem
"""

import ast
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PersonaMemMessage:
    """Single user or assistant message within a block."""
    context_index: int
    block_idx: int        # which block (system message boundary count)
    local_msg_idx: int    # index within block (user/assistant only, 0-based)
    role: str             # "user" or "assistant"
    content: str          # as-is; already contains "User: " / "Assistant: " prefix

    def to_memory_content(self) -> str:
        """Return content as-is for storage."""
        return self.content


@dataclass
class PersonaMemQAPair:
    """QA pair for evaluation."""
    question_id: str
    question_type: str
    topic: str
    question: str
    correct_answer: str           # bare letter: "a", "b", "c", or "d"
    all_options: List[str]        # 4 option strings, e.g. ["(a) ...", "(b) ...", ...]
    end_index_in_shared_context: int
    # metadata (stored in results for analysis)
    persona_id: int
    context_length_in_tokens: int
    distance_to_ref_in_blocks: int
    distance_to_ref_in_tokens: int
    num_irrelevant_tokens: int
    distance_to_ref_proportion_in_context: str


@dataclass
class PersonaMemContext:
    """
    One shared context = one experiment session.

    messages : user + assistant messages only (system excluded), in order.
    qa_pairs : sorted by end_index_in_shared_context ascending.
    """
    context_index: int          # integer position in QA-count-sorted list
    shared_context_id: str      # SHA-256 hash key
    persona_id: int
    messages: List[PersonaMemMessage] = field(default_factory=list)
    qa_pairs: List[PersonaMemQAPair] = field(default_factory=list)

    def get_num_messages(self) -> int:
        return len(self.messages)


# =============================================================================
# PARSING HELPERS
# =============================================================================

def _parse_correct_answer(raw: str) -> str:
    """Extract bare letter from "(a)", "(b)", "(c)", "(d)"."""
    return raw.strip().strip("() ").lower()


def _parse_all_options(raw: str) -> List[str]:
    """Parse stringified Python list of option strings from CSV."""
    try:
        opts = ast.literal_eval(raw)
        return [str(o) for o in opts]
    except Exception:
        return []


def _parse_messages(raw_messages: List[dict], context_index: int) -> List[PersonaMemMessage]:
    """
    Parse raw message list into PersonaMemMessage objects.

    system messages are used only as block boundary markers — never stored.
    block_idx increments at each system message.
    local_msg_idx counts user/assistant messages within the current block.
    """
    messages = []
    block_idx = -1
    local_msg_idx = 0

    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            block_idx += 1
            local_msg_idx = 0
            continue

        if role in ("user", "assistant"):
            messages.append(PersonaMemMessage(
                context_index=context_index,
                block_idx=block_idx,
                local_msg_idx=local_msg_idx,
                role=role,
                content=content,
            ))
            local_msg_idx += 1

    return messages


# =============================================================================
# JSONL INDEX (fast random access)
# =============================================================================

def _build_jsonl_index(jsonl_path: Path) -> Dict[str, int]:
    """Scan JSONL once to build {shared_context_id: file_offset} mapping."""
    index: Dict[str, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index


def _load_context_by_offset(jsonl_path: Path, offset: int) -> List[dict]:
    """Load message list for one context using a pre-built file offset."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_personamem_dataset(
    questions_path: str,
    contexts_path: str,
) -> List[PersonaMemContext]:
    """
    Load PersonaMem dataset for a given benchmark size.

    Args:
        questions_path : Path to questions_[SIZE].csv
        contexts_path  : Path to shared_contexts_[SIZE].jsonl

    Returns:
        List of PersonaMemContext objects sorted by QA count descending.
        (context_index 0 = context with most QA questions)

    Raises:
        FileNotFoundError : If either file does not exist.
    """
    questions_path = Path(questions_path)
    contexts_path  = Path(contexts_path)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not contexts_path.exists():
        raise FileNotFoundError(f"Contexts file not found: {contexts_path}")

    print(f"Loading questions from : {questions_path}")
    print(f"Loading contexts from  : {contexts_path}")

    jsonl_index = _build_jsonl_index(contexts_path)

    qa_by_context: Dict[str, List[PersonaMemQAPair]] = {}
    persona_by_context: Dict[str, int] = {}

    with open(questions_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctx_id     = row["shared_context_id"]
            persona_id = int(row["persona_id"])
            persona_by_context[ctx_id] = persona_id

            qa = PersonaMemQAPair(
                question_id=row["question_id"],
                question_type=row["question_type"],
                topic=row["topic"],
                question=row["user_question_or_message"],
                correct_answer=_parse_correct_answer(row["correct_answer"]),
                all_options=_parse_all_options(row["all_options"]),
                end_index_in_shared_context=int(float(row["end_index_in_shared_context"])),
                persona_id=persona_id,
                context_length_in_tokens=int(row["context_length_in_tokens"]),
                distance_to_ref_in_blocks=int(row["distance_to_ref_in_blocks"]),
                distance_to_ref_in_tokens=int(row["distance_to_ref_in_tokens"]),
                num_irrelevant_tokens=int(row["num_irrelevant_tokens"]),
                distance_to_ref_proportion_in_context=row["distance_to_ref_proportion_in_context"],
            )
            qa_by_context.setdefault(ctx_id, []).append(qa)

    for qa_list in qa_by_context.values():
        qa_list.sort(key=lambda q: q.end_index_in_shared_context)

    sorted_ctx_ids = sorted(
        qa_by_context.keys(),
        key=lambda cid: len(qa_by_context[cid]),
        reverse=True,
    )

    contexts: List[PersonaMemContext] = []
    for context_index, ctx_id in enumerate(sorted_ctx_ids):
        if ctx_id not in jsonl_index:
            print(f"  WARNING: shared_context_id {ctx_id[:16]}... not in JSONL, skipping")
            continue

        raw_messages = _load_context_by_offset(contexts_path, jsonl_index[ctx_id])
        messages = _parse_messages(raw_messages, context_index)

        contexts.append(PersonaMemContext(
            context_index=context_index,
            shared_context_id=ctx_id,
            persona_id=persona_by_context[ctx_id],
            messages=messages,
            qa_pairs=qa_by_context[ctx_id],
        ))

    _print_statistics(contexts)
    return contexts


def _print_statistics(contexts: List[PersonaMemContext]) -> None:
    total_msgs = sum(len(c.messages) for c in contexts)
    total_qa   = sum(len(c.qa_pairs) for c in contexts)
    print(f"  Contexts : {len(contexts)}")
    print(f"  Total messages (user+assistant) : {total_msgs}")
    print(f"  Total QA : {total_qa}")
    if contexts:
        qa_counts = [len(c.qa_pairs) for c in contexts]
        print(f"  QA per context : min={min(qa_counts)}, max={max(qa_counts)}, "
              f"mean={sum(qa_counts)/len(qa_counts):.1f}")
        print(f"  (context_index 0 = most QA = {max(qa_counts)} questions)")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import importlib.util as _ilu
    _cfg_file = Path(__file__).parent / "config_0.py"
    _spec = _ilu.spec_from_file_location("config_0", _cfg_file)
    _cfg = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg)

    contexts = load_personamem_dataset(
        _cfg.DATASET_QUESTIONS_32K,
        _cfg.DATASET_CONTEXTS_32K,
    )

    c = contexts[0]
    print(f"\nContext 0 — shared_context_id : {c.shared_context_id[:24]}...")
    print(f"  persona_id : {c.persona_id}")
    print(f"  messages   : {len(c.messages)}")
    print(f"  qa_pairs   : {len(c.qa_pairs)}")

    print(f"\nFirst 4 messages:")
    for m in c.messages[:4]:
        print(f"  [block={m.block_idx}, local={m.local_msg_idx}] {m.role}: {m.content[:70]}")

    print(f"\nFirst QA pair:")
    q = c.qa_pairs[0]
    print(f"  type    : {q.question_type}")
    print(f"  Q       : {q.question}")
    print(f"  answer  : {q.correct_answer}")
    print(f"  options : {len(q.all_options)} options")
