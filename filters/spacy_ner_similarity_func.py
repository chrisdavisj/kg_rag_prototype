from typing import List
import spacy
from sentence_transformers import util
from embeddings.embedder import model, device

nlp_spacy = spacy.load("en_core_web_sm")


def spacy_similarity_func(prompt: str, ontology_classes: List[str], top_k: int = 3) -> List[str]:
    if not ontology_classes:
        return []

    doc = nlp_spacy(prompt.lower())
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    if not key_phrases:
        key_phrases = [ent.text for ent in doc.ents]
    query_text = " ".join(key_phrases) if key_phrases else prompt

    class_texts = [f"{cls} concept" for cls in ontology_classes]
    query_embedding = model.encode(
        [query_text], convert_to_tensor=True).to(device)
    class_embeddings = model.encode(
        class_texts, convert_to_tensor=True).to(device)
    similarities = util.pytorch_cos_sim(query_embedding, class_embeddings)[0]

    top_indices = similarities.topk(
        k=min(top_k, len(ontology_classes))).indices
    return [ontology_classes[i] for i in top_indices]
