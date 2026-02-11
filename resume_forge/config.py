import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LM_STUDIO_MODEL: str = "local-model"  # Placeholder, LM Studio ignores this
    CHROMA_PERSIST_DIR: str = ".chromadb"
    COLLECTION_NAME: str = "resume_vault"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K: int = 10
    PROMPTS_FILE: str = "templates/prompts.yaml"

    class Config:
        env_file = ".env"

settings = Settings()
