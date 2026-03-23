from typing import Callable, List
from urllib.parse import unquote, urlparse

from embeddings.embedder import model, device
from sentence_transformers import util
from config import get

PREFERRED_CONFIDENCE = get("thresholds.preferred_confidence")


def _class_text_representation(class_uri: str) -> str:
    parsed = urlparse(class_uri)
    tail = parsed.fragment or parsed.path.rstrip("/").split("/")[-1]
    tail = unquote(tail).replace("_", " ").replace("-", " ").strip()

    path_parts = [
        unquote(part).replace("_", " ").replace("-", " ").strip()
        for part in parsed.path.split("/")
        if part.strip()
    ]
    hierarchy = " > ".join(path_parts[-3:])

    parts = [class_uri]
    if tail and tail != class_uri:
        parts.append(tail)
    if hierarchy and hierarchy not in parts:
        parts.append(hierarchy)
    return " | ".join(parts)


def default_class_selector(prompt: str, ontology_classes: List[str]) -> List[str]:
    if not ontology_classes:
        return []
    class_texts = [_class_text_representation(class_uri) for class_uri in ontology_classes]
    prompt_embedding = model.encode(
        [prompt], convert_to_tensor=True).to(device)
    class_embeddings = model.encode(
        class_texts, convert_to_tensor=True).to(device)
    similarities = util.pytorch_cos_sim(prompt_embedding, class_embeddings)[0]
    return [ontology_classes[i] for i, score in enumerate(similarities) if score > PREFERRED_CONFIDENCE]


def select_classes(
    prompt: str,
    ontology_classes: List[str],
    class_selector: Callable[[str, List[str]], List[str]],
) -> List[str]:
    return class_selector(prompt, ontology_classes)
