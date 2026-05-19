import base64

import anthropic

from providers.base import LLMProvider, LLMResponse, TextBlock, ToolUseBlock, VisionProvider


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str | None, model: str):
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict = {"model": self._model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)
        return LLMResponse(
            stop_reason=response.stop_reason,
            content=_convert_content(response.content),
        )


class ClaudeVisionProvider(VisionProvider):
    def __init__(self, api_key: str, base_url: str | None, model: str):
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def analyze(self, image_bytes: bytes, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.standard_b64encode(image_bytes).decode(),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text


def _convert_content(blocks) -> list:
    result = []
    for b in blocks:
        if b.type == "text":
            result.append(TextBlock(text=b.text))
        elif b.type == "tool_use":
            result.append(ToolUseBlock(id=b.id, name=b.name, input=b.input))
    return result
