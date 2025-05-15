from typing import Callable
from sparql.ontology import get_spo
from filters.similarity import extract_entities, default_similarity_func
from sparql.hops import find_min_hops_sparql
from sparql.matcher import brute_force_match_resources_sparql
from filters.resource_filter import filter_matched_resources
from sparql.expander import expand_paths_sparql
from utils.pruning import prune_context
from utils.url_replacer import replace_urls_with_content


def kg_rag_agent(prompt: str, similarity_func: Callable = default_similarity_func) -> str:
    # ========== Step 0: Load Ontology Class Triples from Graph ==========
    classes, _ = get_spo()

    # ========== Step 1: Entity Extraction from Graph ==========
    entities = extract_entities(prompt, classes, similarity_func)

    # ========== Step 2: Query Minimal Hops Between Classes ==========
    min_hops = find_min_hops_sparql(entities)

    # ========== Step 3: Brute Force Matching of Resources ==========
    matched = brute_force_match_resources_sparql(entities, prompt)

    # ========== Step 4: Filter Matched Resources ==========
    filtered = filter_matched_resources(prompt, matched)

    # ========== Step 5: Exhaustive Graph Expansion (to RDFLib) for filtered entities ==========
    context_graph = expand_paths_sparql(filtered)

    # ========== Step 6: Replace External sources with Real Content ==========
    # full_context_graph = replace_urls_with_content(context_graph)

    # ========== Step 7: Intelligent Context Pruning ==========
    pruned = prune_context(context_graph, prompt=prompt)

    return pruned


if __name__ == "__main__":
    user_prompt = input("Enter your query: ")
    print(kg_rag_agent(user_prompt))
