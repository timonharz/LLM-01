"""
Prompt Manager for managing prompts across different AI agents.

- Stores prompts as JSON files under `src/prompts/<agent>/<prompt_name>.json`.
- Each prompt file contains a list of chat messages: [{"role": "system|user|assistant", "content": "..."}].
- Supports simple variable interpolation using Python's str.format_map with a SafeDict.
- No extra dependencies required.

Example file: src/prompts/assistant/general.json
{
  "name": "general",
  "description": "General assistant system instructions",
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "{user_query}"}
  ]
}

Usage:
    from prompt_manager import PromptManager
    pm = PromptManager()
    msgs = pm.get_prompt("assistant", "general", {"user_query": "Say hello"})

    # Use with OpenRouterLLMService
    # svc.chat(msgs, model="openai/gpt-4o-mini")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


class SafeDict(dict):
    """dict that returns placeholders unchanged when key missing."""

    def __missing__(self, key):  # type: ignore[override]
        return "{" + key + "}"


@dataclass
class Prompt:
    name: str
    agent: str
    description: str
    messages: List[Message]
    path: Path


class PromptManager:
    def __init__(self, prompts_dir: Optional[str] = None) -> None:
        # Default to `src/prompts` relative to this file if not provided
        if prompts_dir is None:
            prompts_dir = str(Path(__file__).parent / "prompts")
        self.prompts_dir = Path(prompts_dir)
        self._index: Dict[str, Dict[str, Prompt]] = {}
        self._ensure_dirs()
        self.reload()

    # ----- public API -----
    def list_agents(self) -> List[str]:
        return sorted(self._index.keys())

    def list_prompts(self, agent: str) -> List[str]:
        return sorted((self._index.get(agent) or {}).keys())

    def has_prompt(self, agent: str, name: str) -> bool:
        return name in (self._index.get(agent) or {})

    def get_prompt(self, agent: str, name: str, variables: Optional[Dict[str, str]] = None, *, assistant_prefill: Optional[str] = None) -> List[Message]:
        prompt = self._require_prompt(agent, name)
        variables = variables or {}
        msgs: List[Message] = []
        for m in prompt.messages:
            content = str(m.get("content", ""))
            rendered = content.format_map(SafeDict(variables))
            msgs.append({"role": m.get("role", "user"), "content": rendered})
        if assistant_prefill:
            msgs.append({"role": "assistant", "content": assistant_prefill})
        return msgs

    def save_prompt(self, agent: str, name: str, messages: Iterable[Message], description: str = "") -> Path:
        """Create or overwrite a prompt file and update the index."""
        agent_dir = self.prompts_dir / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        path = agent_dir / f"{name}.json"
        payload = {
            "name": name,
            "description": description,
            "messages": list(messages),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._add_to_index_from_file(path)
        return path

    def reload(self) -> None:
        self._index.clear()
        if not self.prompts_dir.exists():
            return
        for path in self.prompts_dir.rglob("*.json"):
            self._add_to_index_from_file(path)

    # ----- internals -----
    def _ensure_dirs(self) -> None:
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def _require_prompt(self, agent: str, name: str) -> Prompt:
        agent_map = self._index.get(agent)
        if not agent_map or name not in agent_map:
            raise FileNotFoundError(f"Prompt not found: agent='{agent}', name='{name}' in {self.prompts_dir}")
        return agent_map[name]

    def _add_to_index_from_file(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            # Skip malformed files but keep going
            return
        name = str(data.get("name") or path.stem)
        description = str(data.get("description") or "")
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            return
        agent = path.parent.name
        prompt = Prompt(name=name, agent=agent, description=description, messages=messages, path=path)
        self._index.setdefault(agent, {})[name] = prompt


if __name__ == "__main__":
    pm = PromptManager()
    if not pm.list_agents():
        # Seed a sample prompt for quick start
        pm.save_prompt(
            "assistant",
            "general",
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "{user_query}"},
            ],
            description="General assistant prompt",
        )
        print("Seeded sample prompt: assistant/general.json")

    print("Agents:", pm.list_agents())
    for agent in pm.list_agents():
        print(f"Prompts for {agent}:", pm.list_prompts(agent))
    msgs = pm.get_prompt("assistant", "general", {"user_query": "Say hello succinctly."})
    print("Rendered messages:", msgs)
