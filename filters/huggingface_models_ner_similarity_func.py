from typing import List
from transformers import pipeline
from sentence_transformers import util
from embeddings.embedder import model, device

hf_ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)


def hf_similarity_func(prompt: str, ontology_classes: List[str], top_k: int = 3) -> List[str]:
    if not ontology_classes:
        return []

    entities = hf_ner(prompt)
    words = [ent["word"] for ent in entities if ent["score"] > 0.85]
    query_text = " ".join(words) if words else prompt

    class_texts = [f"{cls} concept" for cls in ontology_classes]
    query_embedding = model.encode(
        [query_text], convert_to_tensor=True).to(device)
    class_embeddings = model.encode(
        class_texts, convert_to_tensor=True).to(device)
    similarities = util.pytorch_cos_sim(query_embedding, class_embeddings)[0]

    top_indices = similarities.topk(
        k=min(top_k, len(ontology_classes))).indices
    return [ontology_classes[i] for i in top_indices]
