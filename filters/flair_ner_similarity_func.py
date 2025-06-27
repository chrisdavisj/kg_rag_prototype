from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from sentence_transformers import util
from embeddings.embedder import model, device

flair_tagger = SequenceTagger.load("ner-fast")


def flair_similarity_func(prompt: str, ontology_classes: List[str], top_k: int = 3) -> List[str]:
    if not ontology_classes:
        return []

    sentence = Sentence(prompt)
    flair_tagger.predict(sentence)
    entities = [ent.text for ent in sentence.get_spans("ner")]
    query_text = " ".join(entities) if entities else prompt

    class_texts = [f"{cls} concept" for cls in ontology_classes]
    query_embedding = model.encode(
        [query_text], convert_to_tensor=True).to(device)
    class_embeddings = model.encode(
        class_texts, convert_to_tensor=True).to(device)
    similarities = util.pytorch_cos_sim(query_embedding, class_embeddings)[0]

    top_indices = similarities.topk(
        k=min(top_k, len(ontology_classes))).indices
    return [ontology_classes[i] for i in top_indices]
