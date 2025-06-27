from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from sentence_transformers import util
from embeddings.embedder import model, device
from config import get
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = get("sparql.endpoint")
CONFIDENCE_THRESHOLD = get("thresholds.preferred_confidence")
MAX_WORKERS_FOR_THREAD_POOL = get("multi_threading.num_workers_to_be_used")


def get_dynamic_properties(class_uri: str):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    SELECT DISTINCT ?property WHERE {{
        ?instance a <{class_uri}> ; ?property ?value .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [binding["property"]["value"] for binding in results["results"]["bindings"]]


def match_resources_sparql(class_uri: str, prompt: str):
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
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"[ERROR] SPARQL query failed for {class_uri}: {e}")
        return []

    # Prepare batch texts and resources
    texts = []
    resources = []
    for result in results["results"]["bindings"]:
        combined_text = ' '.join(
            v["value"] for k, v in result.items() if k.startswith('val')
        )
        if combined_text.strip():
            texts.append(combined_text)
            resources.append(result["resource"]["value"])

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
        if score >= CONFIDENCE_THRESHOLD
    ]
    return matched


def brute_force_match_resources_sparql(entities: List[str], prompt: str, max_workers: int = MAX_WORKERS_FOR_THREAD_POOL) -> List[str]:
    matched = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(
            match_resources_sparql, cls, prompt) for cls in entities]

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    matched.extend(result)
            except Exception as e:
                print(f"[ERROR] Failed to match entity: {e}")

    return matched
