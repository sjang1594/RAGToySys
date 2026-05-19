import base64
import uuid

import requests

from providers.base import LLMProvider, LLMResponse, TextBlock, ToolUseBlock, VisionProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        ollama_messages = _to_ollama_messages(messages)
        if system:
            ollama_messages.insert(0, {"role": "system", "content": system})

        payload: dict = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = _convert_tools(tools)

        resp = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        msg = data["message"]

        content: list = []
        if msg.get("content"):
            content.append(TextBlock(text=msg["content"]))

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc["function"]
            content.append(ToolUseBlock(
                id=str(uuid.uuid4()),
                name=fn["name"],
                input=fn.get("arguments", {}),
            ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(stop_reason=stop_reason, content=content)

    def response_to_message(self, response: LLMResponse) -> dict:
        tool_calls = []
        text_parts = []
        for b in response.content:
            if b.type == "text":
                text_parts.append(b.text)
            elif b.type == "tool_use":
                tool_calls.append({"function": {"name": b.name, "arguments": b.input}})

        msg: dict = {"role": "assistant", "content": " ".join(text_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg


class OllamaVisionProvider(VisionProvider):
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def analyze(self, image_bytes: bytes, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [base64.standard_b64encode(image_bytes).decode()],
            }],
            "stream": False,
        }
        resp = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        for block in content:
            btype = block.get("type")
            if btype == "text":
                result.append({"role": role, "content": block["text"]})
            elif btype == "tool_result":
                result.append({
                    "role": "tool",
                    "content": block.get("content", ""),
                    "tool_call_id": block.get("tool_use_id", ""),
                })
            elif btype == "tool_use":
                # Reconstruct Ollama-style assistant message with tool_calls
                result.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": block["name"], "arguments": block["input"]}}],
                })
    return result


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schema to Ollama/OpenAI tool schema."""
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return converted
