import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from resume_forge.vectorstore import ingest_vault, get_retriever
from resume_forge.config import settings

@pytest.fixture
def mock_settings(tmp_path):
    """Override settings for testing to use a temporary directory."""
    original_persist_dir = settings.CHROMA_PERSIST_DIR
    settings.CHROMA_PERSIST_DIR = str(tmp_path / f"chromadb_test_{os.getpid()}")
    yield
    # Cleanup
    if os.path.exists(settings.CHROMA_PERSIST_DIR):
        try:
            shutil.rmtree(settings.CHROMA_PERSIST_DIR)
        except OSError:
            pass # Ignore errors if files are locked (common on Windows, maybe Mac?)
    settings.CHROMA_PERSIST_DIR = original_persist_dir

@pytest.fixture
def temp_vault(tmp_path):
    """Create a temporary vault with dummy markdown files."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    
    (vault_dir / "project1.md").write_text("# Project 1\nBuilt a cool thing with Python.", encoding="utf-8")
    (vault_dir / "project2.md").write_text("# Project 2\nOptimized a database with SQL.", encoding="utf-8")
    
    return vault_dir

def test_ingest_vault(mock_settings, temp_vault):
    """Test that ingest_vault correctly processes files."""
    # We mock get_embeddings to avoid downloading the model during tests if possible, 
    # but for integration we might want real ones. 
    # Let's use real embeddings but on a small set of data for a true integration test 
    # of the vectorstore logic, assuming sentence-transformers is installed.
    
    count = ingest_vault(str(temp_vault))
    assert count == 2 # 2 files, small enough to be 1 chunk each

    # Verify persistence directory exists
    assert os.path.exists(settings.CHROMA_PERSIST_DIR)

def test_retriever(mock_settings, temp_vault):
    """Test that the retriever finds relevant documents."""
    ingest_vault(str(temp_vault))
    retriever = get_retriever()
    
    # Query for Python project
    docs = retriever.invoke("Python")
    assert len(docs) > 0
    assert "Project 1" in docs[0].page_content
    
    # Query for SQL project
    docs = retriever.invoke("SQL")
    assert len(docs) > 0
    assert "Project 2" in docs[0].page_content
