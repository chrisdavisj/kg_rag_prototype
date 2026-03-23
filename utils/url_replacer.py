import re
from html import unescape
from typing import List, Tuple

import requests
from rdflib import Graph, Literal, URIRef


URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
SCRIPT_STYLE_PATTERN = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
DEFAULT_TIMEOUT_SECONDS = 5
DEFAULT_MAX_CONTENT_CHARS = 10000
FETCHED_CONTENT_PREDICATE = URIRef("urn:kg-rag:source_content")


def _extract_url(node) -> str:
    if isinstance(node, URIRef):
        value = str(node)
        return value if URL_PATTERN.match(value) else ""

    if isinstance(node, Literal):
        value = str(node).strip()
        return value if URL_PATTERN.match(value) else ""

    return ""


def _normalize_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _extract_excerpt(text: str, content_type: str, max_chars: int) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    if "html" in content_type:
        cleaned = SCRIPT_STYLE_PATTERN.sub(" ", cleaned)
        cleaned = TAG_PATTERN.sub(" ", cleaned)
        cleaned = unescape(cleaned)

    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    if not cleaned:
        return ""

    return _normalize_text(cleaned, max_chars)


def _iter_url_triples(graph: Graph) -> List[Tuple[object, object, object, str]]:
    matches = []
    for s, p, o in list(graph):
        url = _extract_url(o)
        if url:
            matches.append((s, p, o, url))
    return matches


def _fetch_url_text(url: str, timeout: int, max_chars: int) -> Tuple[str, str]:
    response = requests.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        headers={"User-Agent": "kg-rag-prototype/0.1"},
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text" not in content_type and "json" not in content_type and "xml" not in content_type:
        return "", f"skipped: unsupported content type {content_type or 'unknown'}"

    text = response.text
    if not text or not text.strip():
        return "", "skipped: empty response body"

    excerpt = _extract_excerpt(text, content_type, max_chars)
    if not excerpt:
        return "", "skipped: content could not be reduced to usable text"

    return excerpt, "ok"


def replace_urls_with_content(
    graph: Graph,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> Graph:
    if timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    url_triples = _iter_url_triples(graph)
    if not url_triples:
        return graph

    updates: List[Tuple[object, object, Literal]] = []
    for subject, predicate, original_object, url in url_triples:
        try:
            content, _ = _fetch_url_text(url, timeout=timeout, max_chars=max_chars)
        except requests.RequestException as exc:
            content = ""

        if content:
            updates.append((subject, FETCHED_CONTENT_PREDICATE, Literal(content)))

    for triple in updates:
        graph.add(triple)

    return graph
