from concurrent.futures import ThreadPoolExecutor, as_completed
from config import get, put
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = get("sparql.endpoint")
PREFERRED_CONTEXT_HOPS = get("thresholds.preferred_context_hops")
MAX_WORKERS_FOR_THREAD_POOL = get("multi_threading.num_workers_to_be_used")


def query_hops(class1, class2):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    SELECT (COUNT(?mid) AS ?hops) WHERE {{
      <{class1}> ?p1 ?mid .
      ?mid ?p2 <{class2}> .
    }} LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        value = results['results']['bindings'][0].get('hops', {}).get('value')
        if value:
            return int(value)
    except:
        return None


def find_min_hops_sparql(classes):
    min_hops_required = float('inf')
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FOR_THREAD_POOL) as executor:
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                futures.append(executor.submit(query_hops, class1, class2))

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                min_hops_required = max(min_hops_required, result)

    preferred_hops = min_hops_required if min_hops_required != float(
        'inf') else PREFERRED_CONTEXT_HOPS
    put("thresholds.preferred_context_hops", preferred_hops)
    return preferred_hops
