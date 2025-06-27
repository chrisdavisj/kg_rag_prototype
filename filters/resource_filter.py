from embeddings.embedder import model, device
from sentence_transformers import util
from .threshold import compute_dynamic_threshold


def filter_matched_resources(prompt, matched_resources):
    prompt_embedding = model.encode(
        [prompt], convert_to_tensor=True).to(device)
    res_embeddings = model.encode(
        matched_resources, convert_to_tensor=True).to(device)

    similarities = util.pytorch_cos_sim(prompt_embedding, res_embeddings)[0]

    min_val, avg_val, max_val, median = min(similarities), sum(
        similarities)/len(similarities), max(similarities), sorted(similarities)[len(similarities)//2]

    filteration_threshold = compute_dynamic_threshold(
        min_val, avg_val, max_val, median)
    return [matched_resources[i] for i, score in enumerate(similarities) if score >= filteration_threshold]
