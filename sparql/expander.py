from typing import List
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, TURTLE
from config import get

SPARQL_ENDPOINT = get("sparql.endpoint")

PREFFERED_CONTEXT_HOPS = get("thresholds.preferred_context_hops")
MIN_HOPS_TO_BE_EXPLORED = get("thresholds.min_hops_to_be_explored")
MAX_HOPS_THRESHOLD = get("thresholds.max_hops_threshold")


def expand_paths_sparql(entities: List[str]) -> Graph:
    g = Graph()
    max_hops = min(
        MAX_HOPS_THRESHOLD, max(MIN_HOPS_TO_BE_EXPLORED, PREFFERED_CONTEXT_HOPS))
    for entity in entities:
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
        try:
            results = sparql.query().convert()
            g.parse(data=results, format="turtle")
        except Exception as e:
            print(f"Error querying {entity}: {e}")
    return g
