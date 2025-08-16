"""
OpenRouter LLM Service (Python)

Minimal, dependency-light client around OpenRouter Chat Completions.

- Endpoint: https://openrouter.ai/api/v1/chat/completions
- Auth:   Authorization: Bearer <OPENROUTER_API_KEY>
- Optional headers: HTTP-Referer, X-Title

Usage:
    # pip install requests python-dotenv (optional)
    from openrouter_llm import OpenRouterLLMService

    svc = OpenRouterLLMService(api_key="...", referer="https://your.app", title="Your App")
    text = svc.chat([{"role": "user", "content": "Say hello"}], model="openai/gpt-4o-mini")
    print(text)

    # Streaming
    for chunk in svc.stream_chat([{"role": "user", "content": "Stream a short poem"}]):
        print(chunk, end="")

Notes:
- If `api_key` is omitted, the client will try to read OPENROUTER_API_KEY from env.
- Message format aligns with OpenAI-style chat messages.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Generator, Iterable, List, Optional, Union

import requests


BASE_URL = "https://openrouter.ai/api/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"


Message = Dict[str, Union[str, Dict]]  # {"role": "user|system|assistant", "content": "..."}


class OpenRouterError(Exception):
    """Raised when the OpenRouter API returns an error response."""

    def __init__(self, status_code: int, message: str, payload: Optional[dict] = None) -> None:
        super().__init__(f"OpenRouter API error {status_code}: {message}")
        self.status_code = status_code
        self.payload = payload or {}


class OpenRouterLLMService:
    """Thin sync client for OpenRouter Chat Completions.

    Provides:
    - chat(): single-shot completion, returns the assistant content as str
    - stream_chat(): generator yielding streamed content chunks
    - list_models(): fetch available models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        timeout: float = 60.0,
        base_url: str = BASE_URL,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing OpenRouter API key. Pass api_key or set OPENROUTER_API_KEY in env."
            )
        self.referer = referer
        self.title = title
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ----- public API -----
    def chat(
        self,
        messages: Union[str, Message, List[Message]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """Perform a non-streaming chat completion and return assistant text content.

        messages can be:
        - str: treated as a single user message
        - dict: a single message {role, content}
        - list[dict]: full message array
        """
        body = self._build_body(messages, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=False, extra=extra)
        url = f"{self.base_url}{CHAT_COMPLETIONS_PATH}"
        resp = self._session.post(url, headers=self._headers(), json=body, timeout=self.timeout)
        if resp.status_code >= 400:
            self._raise_api_error(resp)
        data = resp.json()
        # OpenAI-style schema: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            raise OpenRouterError(resp.status_code, "Unexpected response format", data)

    def stream_chat(
        self,
        messages: Union[str, Message, List[Message]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """Stream a chat completion, yielding content deltas as they arrive."""
        body = self._build_body(messages, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True, extra=extra)
        url = f"{self.base_url}{CHAT_COMPLETIONS_PATH}"
        with self._session.post(
            url,
            headers=self._headers(),
            json=body,
            timeout=self.timeout,
            stream=True,
        ) as resp:
            if resp.status_code >= 400:
                # Read text for better error visibility even in stream mode
                text = resp.text
                raise OpenRouterError(resp.status_code, text)

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # SSE payload lines typically start with "data: "
                if line.startswith("data: "):
                    payload = line[len("data: "):].strip()
                else:
                    # Occasionally keep-alive/comment lines may appear; skip
                    continue

                if payload in ("[DONE]", "done", "\"[DONE]\""):
                    break

                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    # ignore non-JSON comment/heartbeat
                    continue

                # OpenAI-style streaming deltas: choices[].delta.content
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        yield content

    def list_models(self) -> List[dict]:
        """Return list of available models (as dicts) from OpenRouter."""
        url = f"{self.base_url}{MODELS_PATH}"
        resp = self._session.get(url, headers=self._headers(include_json=False), timeout=self.timeout)
        if resp.status_code >= 400:
            self._raise_api_error(resp)
        data = resp.json()
        # The models response is typically { data: [ { id: "...", ... }, ... ] }
        return data.get("data", []) if isinstance(data, dict) else data

    # ----- internals -----
    def _headers(self, *, include_json: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if include_json:
            headers["Content-Type"] = "application/json"
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    def _build_body(
        self,
        messages: Union[str, Message, List[Message]],
        *,
        model: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        stream: bool,
        extra: Optional[dict],
    ) -> dict:
        msg_array = self._normalize_messages(messages)
        body: dict = {
            "messages": msg_array,
            # If model is None, OpenRouter may use the user's default; it's recommended to set explicitly
        }
        if model:
            body["model"] = model
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stream:
            body["stream"] = True
        if extra:
            body.update(extra)
        return body

    @staticmethod
    def _normalize_messages(messages: Union[str, Message, List[Message]]) -> List[Message]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, dict):
            # trust caller
            return [messages]  # type: ignore[list-item]
        if isinstance(messages, Iterable):
            return list(messages)  # type: ignore[list-item]
        raise TypeError("messages must be str, dict, or list[dict]")

    @staticmethod
    def _raise_api_error(resp: requests.Response) -> None:
        try:
            payload = resp.json()
            message = (
                payload.get("error", {}).get("message")
                or payload.get("message")
                or resp.text
            )
        except Exception:
            payload = None
            message = resp.text
        raise OpenRouterError(resp.status_code, message, payload)


if __name__ == "__main__":
    # Simple manual test when running this file directly
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("Set OPENROUTER_API_KEY in env to run the demo.")
    else:
        client = OpenRouterLLMService(api_key=key, referer="https://example.com", title="LLM-01 Demo")
        print("Non-streaming demo:\n-------------------")
        out = client.chat("Briefly say hi.", model="openai/gpt-4o-mini", temperature=0.2, max_tokens=64)
        print(out)

        print("\nStreaming demo:\n---------------")
        for chunk in client.stream_chat("Write one short rhyming line.", model="openai/gpt-4o-mini", temperature=0.7, max_tokens=64):
            print(chunk, end="")
        print()
