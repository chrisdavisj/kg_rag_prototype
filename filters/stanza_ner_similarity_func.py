import stanza
from typing import List
from sentence_transformers import util
from embeddings.embedder import model, device

stanza.download("en", processors="tokenize,ner", verbose=False)
nlp_stanza = stanza.Pipeline("en", processors="tokenize,ner", verbose=False)


def stanza_similarity_func(prompt: str, ontology_classes: List[str], top_k: int = 3) -> List[str]:
    if not ontology_classes:
        return []

    doc = nlp_stanza(prompt)
    entities = [ent.text for sent in doc.sentences for ent in sent.ents]
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
