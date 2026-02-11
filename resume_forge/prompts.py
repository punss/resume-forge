import yaml
import os
from resume_forge.config import settings

def load_prompts() -> dict:
    """Loads prompts from the YAML file defined in settings."""
    prompts_path = settings.PROMPTS_FILE
    
    if not os.path.exists(prompts_path):
        # Fallback prompts if file is missing (sanity check)
        return {
            "system_prompt": "You are a helpful assistant assisting with resume tailoring.",
            "user_prompt": "Context: {context}\nJob Description: {job_description}"
        }

    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)

def get_system_prompt() -> str:
    return load_prompts().get("system_prompt", "")

def get_user_prompt() -> str:
    return load_prompts().get("user_prompt", "")
