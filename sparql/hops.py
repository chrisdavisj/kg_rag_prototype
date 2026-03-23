from concurrent.futures import ThreadPoolExecutor, as_completed
from config import get, get_required, put
from errors import SPARQLQueryError
from SPARQLWrapper import SPARQLWrapper, JSON


def query_hops(class1, class2):
    sparql = SPARQLWrapper(get_required("sparql.endpoint"))
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
    except Exception as exc:
        raise SPARQLQueryError(
            f"Failed to query graph hops between {class1} and {class2}"
        ) from exc

    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    value = bindings[0].get("hops", {}).get("value")
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SPARQLQueryError(
            f"Invalid hop count returned for {class1} and {class2}: {value}"
        ) from exc


def find_min_hops_sparql(classes):
    preferred_context_hops = get("thresholds.preferred_context_hops")
    max_workers = get("multi_threading.num_workers_to_be_used")

    if not classes or len(classes) < 2:
        put("thresholds.preferred_context_hops", preferred_context_hops)
        return preferred_context_hops

    min_hops_required = float('inf')
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                futures.append(executor.submit(query_hops, class1, class2))

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                min_hops_required = max(min_hops_required, result)

    preferred_hops = min_hops_required if min_hops_required != float(
        'inf') else preferred_context_hops
    put("thresholds.preferred_context_hops", preferred_hops)
    return preferred_hops
