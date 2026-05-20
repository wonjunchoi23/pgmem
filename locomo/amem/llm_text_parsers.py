"""
Plain-text prompt templates and section-marker parsers for A-MEM.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(response: str, plain_text_parser: Callable, *parser_args) -> Any:
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


def _parse_list_items(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    items: List[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        if "," in line:
            for part in line.split(","):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)
    return items


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    pattern = re.compile(
        rf"^\s*{re.escape(marker)}\s*:\s*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    first_line = match.group(1).strip()

    end = len(text)
    if next_markers:
        for next_marker in next_markers:
            next_pattern = re.compile(
                rf"^\s*{re.escape(next_marker)}\s*:",
                re.IGNORECASE | re.MULTILINE,
            )
            next_match = next_pattern.search(text, start)
            if next_match and next_match.start() < end:
                end = next_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text = _extract_section(resp, "CONTEXT", ["TAGS", "KEYWORDS"])
        tags_text = _extract_section(resp, "TAGS", ["KEYWORDS", "CONTEXT"])

        return {
            "keywords": _parse_list_items(keywords_text),
            "context": context_text.strip() if context_text.strip() else "",
            "tags": _parse_list_items(tags_text),
        }

    result = parse_with_json_fallback(response, _section_parse, content)
    return validate_analysis_result(result, content)


def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    if not keywords and content:
        keywords = _heuristic_keywords(content)
    if not context and content:
        context = _heuristic_context(content)
    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "about", "up", "it", "its", "i", "me", "my",
        "you", "your", "he", "she", "they", "we", "this", "that", "these",
        "those", "what", "which", "who", "whom", "says", "said", "speaker",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", content)
    scored = []
    seen = set()
    for word in words:
        lowered = word.lower()
        if lowered in stop_words or lowered in seen:
            continue
        seen.add(lowered)
        score = 2 if word[0].isupper() else 1
        scored.append((lowered, score))

    scored.sort(key=lambda item: -item[1])
    return [word for word, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    match = re.match(r"(.+?[.!?])\s", content)
    if match:
        return match.group(1).strip()
    return content[:200].strip()
