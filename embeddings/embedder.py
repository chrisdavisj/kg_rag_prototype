from sentence_transformers import SentenceTransformer
import torch
from config import get

model_name = get("model.name")
use_cuda = get("model.use_cuda_if_available")

model = SentenceTransformer(model_name)
device = torch.device(
    "cuda" if use_cuda and torch.cuda.is_available() else "cpu")
model.to(device)
