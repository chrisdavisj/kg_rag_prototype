from typing import List
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, TURTLE
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import get, get_required
from errors import GraphExpansionError


def query_entity_graph(entity: str, max_hops: int) -> Graph:
    sparql = SPARQLWrapper(get_required("sparql.endpoint"))
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
    except Exception as exc:
        raise GraphExpansionError(f"Failed to expand graph around entity {entity}") from exc

    try:
        g.parse(data=results, format="turtle")
    except Exception as exc:
        raise GraphExpansionError(
            f"SPARQL expansion for {entity} returned invalid Turtle data"
        ) from exc
    return g


def expand_paths_sparql(entities: List[str], max_workers: int = None) -> Graph:
    if max_workers is None:
        max_workers = get("multi_threading.num_workers_to_be_used")

    preferred_context_hops = get("thresholds.preferred_context_hops")
    min_hops_to_be_explored = get("thresholds.min_hops_to_be_explored")
    max_hops_threshold = get("thresholds.max_hops_threshold")
    max_hops = min(max_hops_threshold, max(
        min_hops_to_be_explored, preferred_context_hops))
    final_graph = Graph()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(query_entity_graph, entity, max_hops): entity
            for entity in entities
        }

        for future in as_completed(futures):
            partial_graph = future.result()
            final_graph += partial_graph  # RDFLib allows graph union with +=

    return final_graph
