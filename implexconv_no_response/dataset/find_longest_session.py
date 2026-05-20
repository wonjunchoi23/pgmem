"""
ImplexConv_opposed_processed.json에서 turn 수가 가장 많은 session을 찾고,
해당 session의 총 토큰 수를 계산합니다.

토큰 수 계산:
  - tiktoken (cl100k_base, GPT-4 기준)
  - 근사값 (whitespace split x 1.3) — tiktoken 미설치 시 fallback

Install:
  pip install tiktoken
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "implexconv" / "ImplexConv_opposed_processed.json"


def count_tokens_approx(texts: list[str]) -> int:
    return round(sum(len(t.split()) for t in texts) * 1.3)


def count_tokens_tiktoken(texts: list[str], enc) -> int:
    return sum(len(enc.encode(t)) for t in texts)


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    max_session = max(data, key=lambda x: len(x["conversations"]))
    conversations = max_session["conversations"]
    utterances = [turn["utterance"] for turn in conversations]

    print("=" * 60)
    print("Turn 수가 가장 많은 session")
    print("=" * 60)
    print(f"  session_id   : {max_session['metadata']['session_id']}")
    print(f"  total_turns  : {len(conversations)}")
    print(f"  total_conv   : {max_session['metadata']['total_conversations']}")
    print()
    print("[ 토큰 수 계산 대상: conversations의 모든 utterance ]")
    print()

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        total = count_tokens_tiktoken(utterances, enc)
        print(f"  총 토큰 수 (tiktoken cl100k_base) : {total:,}")
        print()

        speaker_tokens: dict[str, int] = {}
        for turn in conversations:
            sp = turn["speaker"]
            speaker_tokens[sp] = speaker_tokens.get(sp, 0) + len(enc.encode(turn["utterance"]))

        print("  [ speaker별 토큰 분포 ]")
        for sp, t in sorted(speaker_tokens.items()):
            print(f"    {sp:15s}: {t:,} tokens")

    except ImportError:
        total = count_tokens_approx(utterances)
        print(f"  총 토큰 수 (근사값, tiktoken 미설치) : {total:,}")
        print()
        print("  tiktoken 설치 후 정확한 값을 얻으려면:")
        print("    pip install tiktoken")

    print("=" * 60)


if __name__ == "__main__":
    main()
