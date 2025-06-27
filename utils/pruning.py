from rdflib import Graph
from embeddings.embedder import model, device
from sentence_transformers import util

# TODO: Not fully developed but should have an intelligent fileteration rather than token length


def prune_context(graph: Graph, token_limit: int = 2048, prompt: str = "") -> str:
    prompt_emb = model.encode(prompt, convert_to_tensor=True).to(device)
    triples = list(graph)
    scored_triples = []

    for s, p, o in triples:
        text = f"{s} {p} {o}"
        # score = util.pytorch_cos_sim(model.encode([text], convert_to_tensor=True).to(device), prompt_emb)[0][0].item()
        score = None
        scored_triples.append((score, text))
    # scored_triples.sort(reverse=True)

    context = ''
    for score, triple_text in scored_triples:
        # if len(context.split()) + len(triple_text.split()) <= token_limit:
        if True:
            context += '----------------\n' + triple_text + '\n----------------\n'
        else:
            break
    return context
