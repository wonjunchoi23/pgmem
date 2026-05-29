"""
Plain-text prompt templates and section-marker parsers for A-MEM.

Copied from original_code/AMEM/llm_text_parsers.py with no logic changes.
Used by memory_layer.py for note construction and 3-call evolution.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(response: str, plain_text_parser: Callable, *parser_args) -> Any:
    """Try JSON parsing first; fall back to section-marker parsing."""
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


# ---------------------------------------------------------------------------
# List parsing helpers
# ---------------------------------------------------------------------------

def _parse_list_items(text: str) -> List[str]:
    """Parse a section of text into a list of items.

    Handles bullet points, comma-separated values, one item per line.
    """
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\-\*\u2022]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)

    return items


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    """Extract the text between *marker*: and the next known marker (or end)."""
    pattern = re.compile(
        rf'^\s*{re.escape(marker)}\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    first_line = match.group(1).strip()

    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

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


EVOLUTION_DECISION_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


STRENGTHEN_DETAILS_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


UPDATE_NEIGHBORS_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


FOCUSED_KEYWORDS_PROMPT = """List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}"""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    """Parse the analyze_content LLM response.

    Returns:
        {"keywords": [...], "context": "...", "tags": [...]}
    """
    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text  = _extract_section(resp, "CONTEXT",  ["TAGS", "KEYWORDS"])
        tags_text     = _extract_section(resp, "TAGS",     ["KEYWORDS", "CONTEXT"])

        keywords = _parse_list_items(keywords_text)
        context  = context_text.strip() if context_text.strip() else ""
        tags     = _parse_list_items(tags_text)

        return {"keywords": keywords, "context": context, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse, content)
    result = validate_analysis_result(result, content)
    return result


def parse_evolution_decision(response: str) -> Dict[str, str]:
    """Parse the evolution decision response.

    Returns:
        {"decision": "NO_EVOLUTION|STRENGTHEN|UPDATE_NEIGHBOR|STRENGTHEN_AND_UPDATE",
         "reason": "..."}
    """
    def _section_parse(resp: str) -> Dict[str, str]:
        decision_text = _extract_section(resp, "DECISION", ["REASON"])
        reason_text   = _extract_section(resp, "REASON",   ["DECISION"])

        decision = decision_text.strip().upper().replace(" ", "_")
        valid_decisions = {
            "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"
        }
        if decision not in valid_decisions:
            resp_upper = resp.upper()
            if "STRENGTHEN" in resp_upper and "UPDATE" in resp_upper:
                decision = "STRENGTHEN_AND_UPDATE"
            elif "STRENGTHEN" in resp_upper:
                decision = "STRENGTHEN"
            elif "UPDATE" in resp_upper:
                decision = "UPDATE_NEIGHBOR"
            else:
                decision = "NO_EVOLUTION"

        return {"decision": decision, "reason": reason_text.strip()}

    result = parse_with_json_fallback(response, _section_parse)

    # Map JSON keys if we got JSON back
    if "should_evolve" in result:
        should_evolve = result.get("should_evolve", False)
        actions = result.get("actions", [])
        if not should_evolve:
            decision = "NO_EVOLUTION"
        elif "strengthen" in actions and "update_neighbor" in actions:
            decision = "STRENGTHEN_AND_UPDATE"
        elif "strengthen" in actions:
            decision = "STRENGTHEN"
        elif "update_neighbor" in actions:
            decision = "UPDATE_NEIGHBOR"
        else:
            decision = "NO_EVOLUTION"
        result = {"decision": decision, "reason": ""}

    if "decision" not in result:
        result = {"decision": "NO_EVOLUTION", "reason": ""}

    return result


def parse_strengthen_details(response: str) -> Dict[str, Any]:
    """Parse the strengthen details response.

    Returns:
        {"connections": [int, ...], "tags": [str, ...]}
    """
    def _section_parse(resp: str) -> Dict[str, Any]:
        conn_text = _extract_section(resp, "CONNECTIONS", ["TAGS"])
        tags_text = _extract_section(resp, "TAGS",        ["CONNECTIONS"])

        connections = []
        for item in _parse_list_items(conn_text):
            try:
                connections.append(int(item.strip()))
            except (ValueError, TypeError):
                pass

        tags = _parse_list_items(tags_text)
        return {"connections": connections, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse)

    if "suggested_connections" in result and "connections" not in result:
        result["connections"] = [
            int(x) for x in result.get("suggested_connections", [])
            if isinstance(x, (int, float))
        ]
    if "tags_to_update" in result and "tags" not in result:
        result["tags"] = result.get("tags_to_update", [])

    result.setdefault("connections", [])
    result.setdefault("tags", [])
    return result


def parse_update_neighbors(response: str, num_neighbors: int) -> List[Dict[str, Any]]:
    """Parse the update neighbors response.

    Returns:
        [{"context": "...", "tags": [...]}, ...] — one per neighbor
    """
    def _section_parse(resp: str, n_neighbors: int) -> List[Dict[str, Any]]:
        neighbors = []
        for i in range(n_neighbors):
            pattern = re.compile(rf'NEIGHBOR\s+{i}\s*:', re.IGNORECASE)
            match = pattern.search(resp)
            if not match:
                neighbors.append({"context": "", "tags": []})
                continue

            next_pattern = re.compile(rf'NEIGHBOR\s+{i + 1}\s*:', re.IGNORECASE)
            next_match = next_pattern.search(resp, match.end())
            block_end = next_match.start() if next_match else len(resp)
            block = resp[match.end():block_end]

            ctx       = _extract_section(block, "CONTEXT", ["TAGS"])
            tags_text = _extract_section(block, "TAGS",    ["CONTEXT"])
            tags      = _parse_list_items(tags_text)

            neighbors.append({"context": ctx.strip(), "tags": tags})

        return neighbors

    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            contexts  = data.get("new_context_neighborhood", [])
            tags_list = data.get("new_tags_neighborhood", [])
            neighbors = []
            for i in range(num_neighbors):
                ctx  = contexts[i]  if i < len(contexts)   else ""
                tags = tags_list[i] if i < len(tags_list)  else []
                neighbors.append({"context": ctx, "tags": tags})
            return neighbors
    except (json.JSONDecodeError, ValueError):
        pass

    return _section_parse(response, num_neighbors)


# ---------------------------------------------------------------------------
# Validation / heuristic repair
# ---------------------------------------------------------------------------

def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Validate and repair the analysis result."""
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context  = result.get("context", "")
    tags     = result.get("tags", [])

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
    result["context"]  = context
    result["tags"]     = tags
    return result


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """Extract heuristic keywords from content text."""
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
        'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my',
        'you', 'your', 'he', 'she', 'they', 'we', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'says', 'said', 'speaker',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    scored = []
    seen = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))

    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    """Extract a heuristic context sentence from content."""
    match = re.match(r'(.+?[.!?])\s', content)
    if match:
        return match.group(1).strip()
    return content[:200].strip()
