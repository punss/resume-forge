import os
import shutil
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from resume_forge.config import settings
from resume_forge.embeddings import get_embeddings, clear_embeddings_cache

def ingest_vault(vault_path: str) -> int:
    """
    Loads markdown files from vault_path, splits them, and indexes into ChromaDB.
    Returns the number of chunks indexed.
    """
    if not os.path.exists(vault_path):
        raise FileNotFoundError(f"Vault directory not found: {vault_path}")

    # Clear caches so the embedding model is re-initialized fresh
    clear_embeddings_cache()

    # Recreate collection to avoid duplicates on re-ingest
    if os.path.exists(settings.CHROMA_PERSIST_DIR):
        shutil.rmtree(settings.CHROMA_PERSIST_DIR)

    loader = DirectoryLoader(
        vault_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    documents = loader.load()
    
    # Filter out templates (files starting with _)
    documents = [
        doc for doc in documents 
        if not os.path.basename(doc.metadata.get("source", "")).startswith("_")
    ]
    
    if not documents:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME
    )
    
    return len(chunks)

def get_retriever() -> VectorStoreRetriever:
    """Returns a retriever connected to the local ChromaDB."""
    vectorstore = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=settings.COLLECTION_NAME
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})
