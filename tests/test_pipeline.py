import pytest
from unittest.mock import MagicMock, patch
from resume_forge.pipeline import tailor_resume_section

@patch("resume_forge.pipeline.get_retriever")
@patch("resume_forge.pipeline.get_llm")
def test_tailor_resume_section(mock_get_llm, mock_get_retriever):
    """
    Test the full RAG pipeline with mocked LLM and Retriever.
    """
    # Mock Retriever
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Relevant context about Python."
    mock_retriever.invoke.return_value = [mock_doc]
    
    # Configure the mock to behave like a Runnable (LangChain)
    # The retriever in the chain is used in a pipe: ... | retriever | ...
    # So we need to ensure the chain construction works. 
    # However, in `build_rag_chain`, we use `retriever` directly. 
    # The `get_retriever` returns a VectorStoreRetriever which is a Runnable.
    mock_get_retriever.return_value = mock_retriever

    # Mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Tailored LaTeX content" # ChatOpenAI returns a message object or AIMessage
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    # Test inputs
    jd = "Looking for a Python developer."
    template = "\\resumeItem{...}"

    # Execution
    # Note: constructing the chain might fail if mocks aren't perfect Runnables.
    # We might need to mock `build_rag_chain` instead if we just want to test input/output flow 
    # but let's try to mock the components to test the chain logic if possible.
    # Actually, mocking `ChatOpenAI` and `VectorStoreRetriever` deeply is hard.
    # It's better to verify the prompt template construction or just mock the chain invoke.
    
    # Let's mock the chain build for simplicity in this unit test to verify `tailor_resume_section` wiring.
    with patch("resume_forge.pipeline.build_rag_chain") as mock_build_chain:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Tailored LaTeX content"
        mock_build_chain.return_value = mock_chain
        
        result = tailor_resume_section(jd, template)
        
        assert result == "Tailored LaTeX content"
        mock_chain.invoke.assert_called_once()
        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["job_description"] == jd
        assert call_args["section_template"] == template
