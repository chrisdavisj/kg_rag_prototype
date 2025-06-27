from typing import List
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, TURTLE
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import get

SPARQL_ENDPOINT = get("sparql.endpoint")

PREFERRED_CONTEXT_HOPS = get("thresholds.preferred_context_hops")
MIN_HOPS_TO_BE_EXPLORED = get("thresholds.min_hops_to_be_explored")
MAX_HOPS_THRESHOLD = get("thresholds.max_hops_threshold")

MAX_WORKERS_FOR_THREAD_POOL = get("multi_threading.num_workers_to_be_used")


def query_entity_graph(entity: str, max_hops: int) -> Graph:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(f"""
    CONSTRUCT {{
        ?s ?p ?o
    }}
    WHERE {{
        <{entity}> ((^<>|<>)){{0,{max_hops}}} ?s .
        ?s ?p ?o .
    }}
    """)
    sparql.setReturnFormat(TURTLE)

    g = Graph()
    try:
        results = sparql.query().convert()
        g.parse(data=results, format="turtle")
    except Exception as e:
        print(f"[ERROR] Failed to query {entity}: {e}")
    return g


def expand_paths_sparql(entities: List[str], max_workers: int = MAX_WORKERS_FOR_THREAD_POOL) -> Graph:
    max_hops = min(MAX_HOPS_THRESHOLD, max(
        MIN_HOPS_TO_BE_EXPLORED, PREFERRED_CONTEXT_HOPS))
    final_graph = Graph()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(
            query_entity_graph, entity, max_hops) for entity in entities]

        for future in as_completed(futures):
            partial_graph = future.result()
            final_graph += partial_graph  # RDFLib allows graph union with +=

    return final_graph
