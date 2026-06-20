"""Gemini 3.5 Flash helper. See gemini_client.py / README.md / AGENTS.md."""
from .gemini_client import (  # noqa: F401
    API_KEY,
    DEFAULT_MODEL,
    Chat,
    ask,
    ask_json,
    describe_media,
    generate,
    get_client,
    resolve_key,
    selftest,
)

__all__ = [
    "ask", "ask_json", "describe_media", "generate", "Chat",
    "get_client", "resolve_key", "selftest", "API_KEY", "DEFAULT_MODEL",
]
