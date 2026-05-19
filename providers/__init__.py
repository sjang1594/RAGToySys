from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_VISION_MODEL,
    VISION_PROVIDER,
)
from providers.base import LLMProvider, VisionProvider


def get_llm_provider() -> LLMProvider:
    if LLM_PROVIDER == "ollama":
        from providers.ollama import OllamaProvider
        return OllamaProvider(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    from providers.claude import ClaudeProvider
    return ClaudeProvider(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL, model=LLM_MODEL)


def get_vision_provider() -> VisionProvider:
    if VISION_PROVIDER == "ollama":
        from providers.ollama import OllamaVisionProvider
        return OllamaVisionProvider(base_url=OLLAMA_BASE_URL, model=OLLAMA_VISION_MODEL)
    from providers.claude import ClaudeVisionProvider
    return ClaudeVisionProvider(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL, model=LLM_MODEL)
