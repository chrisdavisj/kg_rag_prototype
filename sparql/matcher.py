

from sentence_transformers import util
from embeddings.embedder import model, device
from config import get
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = get("sparql.endpoint")
CONFIDENCE_THRESHOLD = get("thresholds.preferred_confidence")


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
    values_query_parts = "\n".join(
        [f"OPTIONAL {{ ?resource <{prop}> ?val{idx} . }}" for idx, prop in enumerate(properties)])
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    SELECT ?resource {' '.join([f'?val{idx}' for idx in range(len(properties))])} WHERE {{
        ?resource a <{class_uri}> .
        {values_query_parts}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    prompt_embedding = model.encode(prompt).to(device)
    matched = []
    for result in results["results"]["bindings"]:
        combined_text = ' '.join(v["value"]
                                 for k, v in result.items() if k.startswith('val'))
        if combined_text:
            combined_embedding = model.encode(combined_text).to(device)

            score = util.pytorch_cos_sim(
                prompt_embedding, combined_embedding)[0][0]
            if score >= CONFIDENCE_THRESHOLD:
                matched.append(result["resource"]["value"])
    return matched
