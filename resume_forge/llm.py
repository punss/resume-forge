import requests
from langchain_openai import ChatOpenAI
from resume_forge.config import settings

def get_llm():
    """Returns a ChatOpenAI instance configured for the local LM Studio server."""
    return ChatOpenAI(
        base_url=settings.LM_STUDIO_BASE_URL,
        api_key="lm-studio",  # Placeholder, not used but required by client
        model=settings.LM_STUDIO_MODEL,
        temperature=0.1, # Lower for higher precision and deterministic formatting
        streaming=True
    )

def check_llm_status() -> bool:
    """Checks if the LLM endpoint is reachable and usable."""
    try:
        # 1. Connectivity Check
        response = requests.get(f"{settings.LM_STUDIO_BASE_URL}/models", timeout=2)
        if response.status_code != 200:
            return False
            
        # 2. Model Availability Check
        data = response.json()
        if not data.get("data"):
            return False
            
        # 3. Usability Check (ensure a model is actually LOADED)
        # We send a tiny request to verify the server can actually generate
        test_payload = {
            "model": settings.LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        res = requests.post(f"{settings.LM_STUDIO_BASE_URL}/chat/completions", json=test_payload, timeout=3)
        return res.status_code == 200
        
    except (requests.exceptions.RequestException, ValueError):
        return False
