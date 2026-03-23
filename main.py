from typing import Callable
from sparql.ontology import get_spo
from filters.similarity import select_classes, default_class_selector
from sparql.hops import find_min_hops_sparql
from sparql.matcher import brute_force_match_resources_sparql
from filters.resource_filter import filter_matched_resources
from sparql.expander import expand_paths_sparql
from errors import KGRAGError
from utils.pruning import prune_context
from utils.url_replacer import replace_urls_with_content


def _dedupe_preserve_order(values):
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def kg_rag_agent(prompt: str, class_selector: Callable = default_class_selector) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    # ========== Step 0: Load Ontology Class Triples from Graph ==========
    classes, _ = get_spo()
    if not classes:
        raise KGRAGError("Ontology query returned no classes to match against")

    # ========== Step 1: Ontology Class Selection ==========
    selected_classes = select_classes(prompt, classes, class_selector)
    selected_classes = _dedupe_preserve_order(selected_classes)
    if not selected_classes:
        raise KGRAGError("No ontology classes matched the input prompt")

    # ========== Step 2: Query Minimal Hops Between Classes ==========
    find_min_hops_sparql(selected_classes)

    # ========== Step 3: Brute Force Matching of Resources ==========
    matched = brute_force_match_resources_sparql(selected_classes, prompt)
    matched = _dedupe_preserve_order(matched)
    if not matched:
        raise KGRAGError("No graph resources matched the selected ontology classes")

    # ========== Step 4: Filter Matched Resources ==========
    filtered = filter_matched_resources(prompt, matched)
    filtered = _dedupe_preserve_order(filtered)
    if not filtered:
        raise KGRAGError("Resource filtering removed all matched graph resources")

    # ========== Step 5: Exhaustive Graph Expansion (to RDFLib) for filtered resources ==========
    context_graph = expand_paths_sparql(filtered)
    if len(context_graph) == 0:
        raise KGRAGError("Graph expansion returned no triples for the filtered resources")

    # ========== Step 6: Replace External sources with Real Content ==========
    # full_context_graph = replace_urls_with_content(context_graph)

    # ========== Step 7: Intelligent Context Pruning ==========
    pruned = prune_context(context_graph, prompt=prompt)
    if not pruned.strip():
        raise KGRAGError("Context pruning returned an empty payload")

    return pruned


if __name__ == "__main__":
    user_prompt = input("Enter your query: ")
    print(kg_rag_agent(user_prompt))
