from collections import defaultdict
from typing import Dict, List, Set, Tuple

from rdflib import Graph
from sentence_transformers import util

from embeddings.embedder import model, device


ScoredTriple = Tuple[float, str, str, str]
SubgraphChunk = Dict[str, object]


def _compact_node_text(node) -> str:
    return str(node).strip().replace("\n", " ")


def _approx_token_count(text: str) -> int:
    # Lightweight heuristic to keep the prototype dependency-free.
    return max(1, len(text.split()))


def _score_triples(graph: Graph, prompt: str) -> List[ScoredTriple]:
    triples = list(graph)
    if not triples:
        return []

    triple_texts = []
    triple_parts = []
    for s, p, o in triples:
        subject = _compact_node_text(s)
        predicate = _compact_node_text(p)
        obj = _compact_node_text(o)
        triple_texts.append(f"{subject} {predicate} {obj}")
        triple_parts.append((subject, predicate, obj))

    prompt_embedding = model.encode([prompt], convert_to_tensor=True).to(device)
    triple_embeddings = model.encode(
        triple_texts, convert_to_tensor=True
    ).to(device)
    similarities = util.pytorch_cos_sim(prompt_embedding, triple_embeddings)[0]

    scored_triples = []
    for i, score in enumerate(similarities):
        subject, predicate, obj = triple_parts[i]
        scored_triples.append((float(score), subject, predicate, obj))

    scored_triples.sort(key=lambda item: item[0], reverse=True)
    return scored_triples


def _compute_filtration_threshold(scores: List[float]) -> float:
    if not scores:
        return 0.0
    if len(scores) == 1:
        return scores[0]

    sorted_scores = sorted(scores, reverse=True)
    avg_score = sum(sorted_scores) / len(sorted_scores)
    median_score = sorted_scores[len(sorted_scores) // 2]
    top_score = sorted_scores[0]
    floor_score = sorted_scores[-1]

    threshold = max(avg_score, (median_score + top_score) / 2)
    threshold = min(threshold, top_score)
    threshold = max(threshold, floor_score)
    return threshold


def _build_components(scored_triples: List[ScoredTriple]) -> List[SubgraphChunk]:
    node_to_triples: Dict[str, List[int]] = defaultdict(list)
    triples_by_index: Dict[int, ScoredTriple] = {}

    for index, triple in enumerate(scored_triples):
        score, subject, _, obj = triple
        triples_by_index[index] = triple
        node_to_triples[subject].append(index)
        node_to_triples[obj].append(index)

    visited_triples: Set[int] = set()
    components: List[SubgraphChunk] = []

    for start_index in triples_by_index:
        if start_index in visited_triples:
            continue

        stack = [start_index]
        component_indices: Set[int] = set()
        component_nodes: Set[str] = set()

        while stack:
            current = stack.pop()
            if current in visited_triples:
                continue

            visited_triples.add(current)
            component_indices.add(current)

            _, subject, _, obj = triples_by_index[current]
            component_nodes.add(subject)
            component_nodes.add(obj)

            for node in (subject, obj):
                for neighbor_index in node_to_triples[node]:
                    if neighbor_index not in visited_triples:
                        stack.append(neighbor_index)

        component_triples = [triples_by_index[index] for index in component_indices]
        component_triples.sort(key=lambda item: item[0], reverse=True)
        node_scores = defaultdict(float)
        for score, subject, _, obj in component_triples:
            node_scores[subject] = max(node_scores[subject], score)
            node_scores[obj] = max(node_scores[obj], score)

        components.append(
            {
                "triples": component_triples,
                "nodes": sorted(component_nodes, key=lambda node: node_scores[node], reverse=True),
                "score": max(score for score, _, _, _ in component_triples),
                "avg_score": sum(score for score, _, _, _ in component_triples) / len(component_triples),
            }
        )

    components.sort(
        key=lambda component: (component["score"], component["avg_score"]),
        reverse=True,
    )
    return components


def _serialize_subgraph(index: int, component: SubgraphChunk) -> str:
    nodes = component["nodes"]
    triples = component["triples"]
    score = component["score"]

    lines = [
        f"Subgraph {index}",
        f"Priority score: {score:.4f}",
        f"Key nodes: {', '.join(nodes[:5])}" if nodes else "Key nodes:",
        "Ranked facts:",
    ]

    for triple_score, subject, predicate, obj in triples:
        lines.append(f"- ({triple_score:.4f}) {subject} --{predicate}--> {obj}")

    return "\n".join(lines)


def _trim_component_to_budget(
    component: SubgraphChunk,
    index: int,
    remaining_tokens: int,
) -> str:
    triples = component["triples"]
    if not triples or remaining_tokens <= 0:
        return ""

    base_lines = [
        f"Subgraph {index}",
        f"Priority score: {component['score']:.4f}",
        f"Key nodes: {', '.join(component['nodes'][:5])}" if component["nodes"] else "Key nodes:",
        "Ranked facts:",
    ]
    base_text = "\n".join(base_lines)
    base_tokens = _approx_token_count(base_text)
    if base_tokens >= remaining_tokens:
        return ""

    kept_lines = list(base_lines)
    used_tokens = base_tokens

    for triple_score, subject, predicate, obj in triples:
        fact_line = f"- ({triple_score:.4f}) {subject} --{predicate}--> {obj}"
        fact_tokens = _approx_token_count(fact_line)
        if used_tokens + fact_tokens > remaining_tokens:
            break
        kept_lines.append(fact_line)
        used_tokens += fact_tokens

    if len(kept_lines) == len(base_lines):
        return ""

    return "\n".join(kept_lines)


def prune_context(
    graph: Graph,
    token_limit: int = 2048,
    prompt: str = "",
    max_subgraphs: int = 8,
) -> str:
    if token_limit <= 0:
        raise ValueError("token_limit must be a positive integer")
    if max_subgraphs <= 0:
        raise ValueError("max_subgraphs must be a positive integer")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must be a non-empty string for context pruning")

    scored_triples = _score_triples(graph, prompt)
    if not scored_triples:
        return ""

    threshold = _compute_filtration_threshold([score for score, _, _, _ in scored_triples])
    filtered_triples = [
        triple for triple in scored_triples if triple[0] >= threshold
    ]

    if not filtered_triples:
        filtered_triples = scored_triples[:1]

    components = _build_components(filtered_triples)
    if not components:
        return ""

    header = "Knowledge Graph Context:"

    sections = [header]
    used_tokens = _approx_token_count(header)
    included_subgraphs = 0

    for index, component in enumerate(components, start=1):
        if included_subgraphs >= max_subgraphs:
            break

        chunk_text = _serialize_subgraph(index, component)
        chunk_tokens = _approx_token_count(chunk_text)
        remaining_tokens = token_limit - used_tokens
        if sections and used_tokens + chunk_tokens > token_limit:
            trimmed_chunk = _trim_component_to_budget(
                component,
                index,
                remaining_tokens,
            )
            if not trimmed_chunk:
                continue
            sections.append(trimmed_chunk)
            used_tokens += _approx_token_count(trimmed_chunk)
            included_subgraphs += 1
            continue

        sections.append(chunk_text)
        used_tokens += chunk_tokens
        included_subgraphs += 1

    if included_subgraphs == 0:
        fallback_budget = max(1, token_limit - _approx_token_count(header))
        fallback_chunk = _trim_component_to_budget(components[0], 1, fallback_budget)
        sections.append(fallback_chunk or _serialize_subgraph(1, components[0]))

    return "\n\n".join(sections)
