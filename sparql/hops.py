from config import get, put
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = get("sparql.endpoint")
PREFERRED_CONTEXT_HOPS = get("thresholds.preferred_context_hops")


def find_min_hops_sparql(classes):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    min_hops = float('inf')
    for i, class1 in enumerate(classes):
        for class2 in classes[i+1:]:
            query = f"""
            SELECT (COUNT(?mid) AS ?hops) WHERE {{
              <{class1}> (<>|!<>)* ?mid .
              ?mid (<>|!<>)* <{class2}> .
            }} LIMIT 1
            """
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            try:
                results = sparql.query().convert()
                value = results['results']['bindings'][0].get(
                    'hops', {}).get('value')
                if value:
                    min_hops = max(min_hops, int(value))
            except:
                continue
    preferred_hops = min_hops if min_hops != float(
        'inf') else PREFERRED_CONTEXT_HOPS
    put("thresholds.preferred_context_hops", preferred_hops)
    return preferred_hops
