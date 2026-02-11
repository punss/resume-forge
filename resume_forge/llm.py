from langchain_openai import ChatOpenAI
from resume_forge.config import settings

def get_llm():
    """Returns a ChatOpenAI instance configured for the local LM Studio server."""
    return ChatOpenAI(
        base_url=settings.LM_STUDIO_BASE_URL,
        api_key="lm-studio",  # Placeholder, not used but required by client
        model=settings.LM_STUDIO_MODEL,
        temperature=0.7, # Slightly higher for creative writing/phrasing
        streaming=True
    )
