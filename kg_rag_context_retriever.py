# Prototype: KG-RAG Pipeline (Agentic, SPARQL-based, RDFLib local graph)

from typing import List, Dict, Any, Callable
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
import requests
from sentence_transformers import SentenceTransformer, util
import re
from rdflib import Graph, URIRef
import torch

# ========== Configuration ==========
PREFERRED_CONTEXT_HOPS = 3
PREFERRED_CONFIDENCE = 0.23
SPARQL_ENDPOINT = "<SPARQL_ENDPOINT_URL>"  # Replace with your SPARQL endpoint URL

# Sentence embedding model for similarity search
from torch import device as torch_device
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder_device = torch_device('cuda' if embedder.device.type == 'cuda' else 'cpu')

# ========== Step 0: Load Ontology Class Triples from Graph ==========
def get_spo() -> str:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT DISTINCT ?s ?p ?o
    WHERE {
      ?s ?p ?o .
      ?s a owl:Class .
    }
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    spo = [
        f'{result["s"]["value"]}, {result["p"]["value"]}, {result["o"]["value"]}' for result in results["results"]["bindings"]
    ]
    classes = set()
    
    for result in results["results"]["bindings"]:
        classes.add(result['s']['value'])
    classes = list(classes)
    return classes, spo

# ========== Step 1: Entity Extraction from Graph ==========
def extract_entities(prompt: str, ontology_classes: List[str], similarity_func: Callable[[str, List[str]], List[str]]) -> List[str]:
    return similarity_func(prompt, ontology_classes)

def default_similarity_func(prompt: str, ontology_classes: List[str]) -> List[str]:
    if not ontology_classes:
        return []
    prompt_embedding = embedder.encode([prompt], convert_to_tensor=True).to(embedder_device)
    class_embeddings = embedder.encode(ontology_classes, convert_to_tensor=True).to(embedder_device)
    similarities = util.pytorch_cos_sim(prompt_embedding, class_embeddings)[0]
    return [ontology_classes[i] for i, score in enumerate(similarities) if score > PREFERRED_CONFIDENCE]

# ========== Step 2: Query Minimal Hops Between Classes ==========
def find_min_hops_sparql(classes: List[str]) -> int:
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
                value = results['results']['bindings'][0].get('hops', {}).get('value')
                if value:
                    min_hops = max(min_hops, int(value))
            except:
                continue
    return min_hops if min_hops != float('inf') else 1

# ========== Step 3: Brute Force Matching of Resources ==========
def get_dynamic_properties(class_uri: str) -> List[str]:
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

def match_resources_sparql(class_uri: str, prompt: str) -> List[str]:
    properties = get_dynamic_properties(class_uri)
    values_query_parts = "\n".join([f"OPTIONAL {{ ?resource <{prop}> ?val{idx} . }}" for idx, prop in enumerate(properties)])
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
    matched = []
    for result in results["results"]["bindings"]:
        combined_text = ' '.join(v["value"] for k, v in result.items() if k.startswith('val'))
        if combined_text:
            score = util.pytorch_cos_sim(embedder.encode(prompt), embedder.encode(combined_text))[0][0]
            if score >= PREFERRED_CONFIDENCE:
                matched.append(result["resource"]["value"])
    return matched

# ========== Step 4: Filter Matched Resources ==========
def compute_dynamic_threshold(min_val, avg_val, max_val, median):
    if avg_val == min_val:
        return min_val  # avoid division by zero; fallback threshold

    # Normalize the position of median between min and avg
    norm_median_pos = (median - min_val) / (avg_val - min_val)

    # Invert it: closer to min = higher weight to avg, closer to avg = higher weight to min
    weight_min = 1 - norm_median_pos
    weight_avg = norm_median_pos

    # Optional: normalize weights (just in case, though they should sum to 1)
    total_weight = weight_min + weight_avg
    weight_min /= total_weight
    weight_avg /= total_weight

    # Dynamic threshold
    threshold = (weight_min * min_val) + (weight_avg * avg_val)

    return threshold

def filter_matched_resources(prompt, matched_resources):
    prompt_embedding = embedder.encode([prompt], convert_to_tensor=True).to(embedder_device)
    res_embeddings = embedder.encode(matched_resources, convert_to_tensor=True).to(embedder_device)
    
    similarities = util.pytorch_cos_sim(prompt_embedding, res_embeddings)[0]
    
    min_val, avg_val, max_val, median = min(similarities), sum(similarities)/len(similarities), max(similarities), sorted(similarities)[len(similarities)//2]
    
    filteration_threshold = compute_dynamic_threshold(min_val, avg_val, max_val, median)
    
    return [matched_resources[i] for i, score in enumerate(similarities) if score >= filteration_threshold]    

# ========== Step 5: Exhaustive Graph Expansion (to RDFLib) for filtered entities ==========
def expand_paths_sparql(entities: List[str], max_hops: int) -> Graph:
    g = Graph()
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

# ========== Step 6: Replace URLs with Real Content ==========
# Used to replace the real content in place, in case of it being linked to a source rather than the source being in the KG
def replace_urls_with_content(graph: Graph):
    for s, p, o in graph:
        if isinstance(o, str) and re.match(r'^https?://', o):
            try:
                response = requests.get(o, timeout=5)
                if response.status_code == 200:
                    graph.remove((s, p, o))
                    graph.add((s, p, response.text))
            except Exception:
                continue

# ========== Step 7: Intelligent Context Pruning ==========
# TODO: Not fully developed but should have an intelligent fileteration rather than token length
def prune_context(graph: Graph, token_limit: int = 2048, prompt: str = "") -> str:
    prompt_emb = embedder.encode(prompt, convert_to_tensor=True).to(embedder_device)
    triples = list(graph)
    scored_triples = []
    for s, p, o in triples:
        text = f"{s} {p} {o}"
        # score = util.pytorch_cos_sim(embedder.encode(text,convert_to_tensor=True).to(embedder_device), prompt_emb)[0][0].item()
        score = None
        scored_triples.append((score, text))
    # scored_triples.sort(reverse=True)
    context = ''
    for score, triple_text in scored_triples:
        # if len(context.split()) + len(triple_text.split()) <= token_limit:
        if True:
            context += '-------\n' + triple_text + '\n-------\n'
        else:
            break
    return context

# ========== Final KG-RAG Pipeline ==========
def kg_rag_agent(prompt: str, similarity_func: Callable = default_similarity_func) -> str:
    try:
        classes, ontology = get_spo()
        entities = extract_entities(prompt, classes, similarity_func)
        min_hops = find_min_hops_sparql(entities)
        context_hops = max(min_hops, PREFERRED_CONTEXT_HOPS)
        matched_resources = []
        for cls in entities:
            matched_resources += match_resources_sparql(cls, prompt)
        matched_resources_filtered = filter_matched_resources(prompt, matched_resources)
        context_graph = expand_paths_sparql(matched_resources_filtered, context_hops)
        # replace_urls_with_content(context_graph)
        pruned = prune_context(context_graph, prompt=prompt)
        return pruned
    except:
        return ""