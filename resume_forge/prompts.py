import hashlib
import os

import yaml

from resume_forge.config import settings

_cached_prompts = None
_cached_prompts_hash = None


def _file_hash(path: str) -> str:
    """Returns the SHA-256 hex digest of a file's contents."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_prompts() -> dict:
    """
    Loads prompts from the YAML file defined in settings.
    Results are cached and only reloaded when the file changes (hash-based check).
    """
    global _cached_prompts, _cached_prompts_hash
    prompts_path = settings.PROMPTS_FILE

    if not os.path.exists(prompts_path):
        # Fallback prompts if file is missing
        return {
            "system_prompt": "You are a helpful assistant assisting with resume tailoring.",
            "user_prompt": "Context: {context}\nJob Description: {job_description}"
        }

    current_hash = _file_hash(prompts_path)

    if _cached_prompts is not None and _cached_prompts_hash == current_hash:
        return _cached_prompts

    with open(prompts_path, "r") as f:
        _cached_prompts = yaml.safe_load(f)
    _cached_prompts_hash = current_hash
    return _cached_prompts


def get_system_prompt() -> str:
    return load_prompts().get("system_prompt", "")


def get_user_prompt() -> str:
    return load_prompts().get("user_prompt", "")
