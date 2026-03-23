from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from sentence_transformers import util
from embeddings.embedder import model, device
from config import get, get_required
from errors import SPARQLQueryError
from SPARQLWrapper import SPARQLWrapper, JSON


def get_dynamic_properties(class_uri: str):
    sparql = SPARQLWrapper(get_required("sparql.endpoint"))
    query = f"""
    SELECT DISTINCT ?property WHERE {{
        ?instance a <{class_uri}> ; ?property ?value .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except Exception as exc:
        raise SPARQLQueryError(
            f"Failed to load dynamic properties for class {class_uri}"
        ) from exc

    bindings = results.get("results", {}).get("bindings", [])
    return [
        binding["property"]["value"]
        for binding in bindings
        if "property" in binding and "value" in binding["property"]
    ]


def match_resources_sparql(class_uri: str, prompt: str):
    confidence_threshold = get("thresholds.preferred_confidence")
    properties = get_dynamic_properties(class_uri)
    if not properties:
        return []

    # Build the dynamic SPARQL query
    values_query_parts = "\n".join(
        [f"OPTIONAL {{ ?resource <{prop}> ?val{idx} . }}" for idx,
            prop in enumerate(properties)]
    )
    query = f"""
    SELECT ?resource {' '.join([f'?val{idx}' for idx in range(len(properties))])} WHERE {{
        ?resource a <{class_uri}> .
        {values_query_parts}
    }}
    """
    sparql = SPARQLWrapper(get_required("sparql.endpoint"))
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
    except Exception as exc:
        raise SPARQLQueryError(
            f"Resource matching query failed for class {class_uri}"
        ) from exc

    # Prepare batch texts and resources
    texts = []
    resources = []
    bindings = results.get("results", {}).get("bindings", [])
    for result in bindings:
        combined_text = ' '.join(
            v["value"] for k, v in result.items() if k.startswith('val')
        )
        resource_value = result.get("resource", {}).get("value")
        if combined_text.strip() and resource_value:
            texts.append(combined_text)
            resources.append(resource_value)

    if not texts:
        return []

    # Batch encode
    prompt_embedding = model.encode(prompt, convert_to_tensor=True).to(device)
    combined_embeddings = model.encode(
        texts, convert_to_tensor=True).to(device)

    similarities = util.pytorch_cos_sim(
        prompt_embedding, combined_embeddings)[0]

    # Return matched resources over threshold
    matched = [
        resources[i] for i, score in enumerate(similarities)
        if score >= confidence_threshold
    ]
    return matched


def brute_force_match_resources_sparql(entities: List[str], prompt: str, max_workers: int = None) -> List[str]:
    if max_workers is None:
        max_workers = get("multi_threading.num_workers_to_be_used")

    matched = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(match_resources_sparql, cls, prompt): cls
            for cls in entities
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    matched.extend(result)
            except Exception as exc:
                class_uri = futures[future]
                raise SPARQLQueryError(
                    f"Failed to match resources for class {class_uri}"
                ) from exc

    return matched
