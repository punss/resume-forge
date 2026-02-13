import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from resume_forge.config import settings

_cached_embeddings = None


def get_embeddings():
    """Returns the HuggingFace embeddings model, cached as a singleton."""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

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
    _cached_embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return _cached_embeddings


def clear_embeddings_cache():
    """Clears the cached embeddings instance. Called on re-ingest."""
    global _cached_embeddings
    _cached_embeddings = None
