from langchain_community.embeddings import HuggingFaceEmbeddings
from resume_forge.config import settings

def get_embeddings():
    """Returns the HuggingFace embeddings model configured in settings."""
    model_kwargs = {'device': 'cpu'} # Force CPU unless user has CUDA/MPS correctly set up, safe default
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
