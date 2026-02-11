import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from resume_forge.config import settings

def get_embeddings():
    """Returns the HuggingFace embeddings model configured in settings."""
    device = settings.DEVICE
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
