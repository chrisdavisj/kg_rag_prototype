from typing import List, Dict, Any, Callable
from embeddings.embedder import model, device
from sentence_transformers import util
from config import get

PREFERRED_CONFIDENCE = get("thresholds.preferred_confidence")


def default_similarity_func(prompt: str, ontology_classes: List[str]) -> List[str]:
    if not ontology_classes:
        return []
    prompt_embedding = model.encode(
        [prompt], convert_to_tensor=True).to(device)
    class_embeddings = model.encode(
        ontology_classes, convert_to_tensor=True).to(device)
    similarities = util.pytorch_cos_sim(prompt_embedding, class_embeddings)[0]
    return [ontology_classes[i] for i, score in enumerate(similarities) if score > PREFERRED_CONFIDENCE]


def extract_entities(prompt: str, ontology_classes: List[str], similarity_func: Callable[[str, List[str]], List[str]]) -> List[str]:
    return similarity_func(prompt, ontology_classes)
