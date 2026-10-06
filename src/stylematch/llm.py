"""Chat model adapters behind one interface: ``complete(messages) -> str``.

``OpenAICompatibleClient`` calls any OpenAI-compatible ``/chat/completions``
endpoint with the standard library, in JSON mode, at temperature 0. The key
comes from the settings (the environment), never from source code.
``ScriptedChatClient`` returns fixed replies for the tests and the offline demo.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable


class ChatClient:
    name = "base"

    def complete(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(ChatClient):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, timeout_s: float = 30.0, retries: int = 2):
        if not api_key:
            raise ValueError("the LLM client needs STYLEMATCH_LLM_API_KEY or OPENAI_API_KEY")
        self._key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries

    def __repr__(self) -> str:
        return f"OpenAICompatibleClient(model={self.model!r}, base_url={self.base_url!r})"

    def complete(self, messages):
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0, "response_format": {"type": "json_object"}}
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - configured URL
                    payload = json.loads(response.read())
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503) and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError("unreachable")


class ScriptedChatClient(ChatClient):
    """Returns a fixed reply, or the result of ``reply(messages)``. No network."""

    name = "scripted"

    def __init__(self, reply: str | Callable[[list[dict]], str]):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply(messages) if callable(self.reply) else self.reply


def make_client(settings) -> ChatClient | None:
    if not settings.has_llm:
        return None
    return OpenAICompatibleClient(settings.llm_api_key, settings.llm_model, settings.llm_base_url, settings.llm_timeout_s)
