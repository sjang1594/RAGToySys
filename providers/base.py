from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

"""Base classes for LLM and Vision providers."""
@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""

@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    stop_reason: str
    content: list[Any]

""" Abstract Class: LLMProvider defines the interface for language model 
providers, with a method to generate completions based on input messages, 
system instructions, and tools. 

It also includes a helper method to convert the LLM response 
into a standardized message format for history tracking."""
class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        pass

    def response_to_message(self, response: LLMResponse) -> dict:
        """Convert LLMResponse to canonical message dict for history accumulation."""
        blocks = []
        for b in response.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return {"role": "assistant", "content": blocks}


class VisionProvider(ABC):
    @abstractmethod
    def analyze(self, image_bytes: bytes, prompt: str) -> str:
        pass
