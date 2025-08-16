"""
Single-LLM refinement pipeline: generate -> feedback -> refine

Uses the same underlying LLM for all three phases. Prompts are stored under
`src/prompts/` and loaded via PromptManager.

Prompts expected:
- p_gen:     agent "assistant", name "generate"
- p_fb:      agent "refiner",  name "feedback"
- p_refine:  agent "refiner",  name "apply_feedback"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Any

from .openrouter_llm import OpenRouterLLMService
from .prompt_manager import PromptManager


@dataclass
class RefinementResult:
    question: str
    initial_answer: str
    feedback: str
    refined_answer: str


class SingleLLMRefinementPipeline:
    def __init__(
        self,
        llm: OpenRouterLLMService,
        *,
        prompt_manager: Optional[PromptManager] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra = extra or {}

    # ----- multimodal helpers -----
    @staticmethod
    def _attach_multimodal_parts(
        msgs: List[Dict[str, Any]],
        *,
        image_urls: Optional[List[str]] = None,
        extra_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Attach image_url parts and arbitrary extra content parts to the last user message.

        - Preserves existing text by converting it into a {"type":"text"} part.
        - image_urls: list of http(s) or data URLs. Each becomes {"type":"image_url","image_url":{"url":...}}.
        - extra_parts: advanced usage for models that support additional content types; these parts are appended as-is.
        """
        image_urls = image_urls or []
        extra_parts = extra_parts or []
        if not image_urls and not extra_parts:
            return msgs

        # Find last user message
        user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                user_idx = i
                break
        if user_idx is None:
            # If none, append a new user message to carry parts
            msgs = list(msgs) + [{"role": "user", "content": ""}]
            user_idx = len(msgs) - 1

        user_msg = dict(msgs[user_idx])
        content = user_msg.get("content", "")

        # Start parts with existing text (if str), else if already list assume caller handled it
        parts: List[Dict[str, Any]] = []
        if isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            # Use existing list as starting point
            parts = list(content)

        # Add images
        for url in image_urls:
            parts.append({"type": "image_url", "image_url": {"url": url}})

        # Add any extra parts (e.g., video or other model-specific parts)
        parts.extend(extra_parts)

        user_msg["content"] = parts
        new_msgs = list(msgs)
        new_msgs[user_idx] = user_msg
        return new_msgs

    # ----- steps -----
    def generate(
        self,
        question: str,
        *,
        image_urls: Optional[List[str]] = None,
        extra_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        msgs = self.pm.get_prompt("assistant", "generate", {"question": question})
        msgs = self._attach_multimodal_parts(msgs, image_urls=image_urls, extra_parts=extra_parts)
        return self.llm.chat(
            msgs,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra=self.extra,
        )

    def feedback(
        self,
        question: str,
        answer: str,
        *,
        image_urls: Optional[List[str]] = None,
        extra_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        msgs = self.pm.get_prompt("refiner", "feedback", {"question": question, "answer": answer})
        msgs = self._attach_multimodal_parts(msgs, image_urls=image_urls, extra_parts=extra_parts)
        return self.llm.chat(
            msgs,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra=self.extra,
        )

    def refine(
        self,
        question: str,
        prior_answer: str,
        feedback: str,
        *,
        image_urls: Optional[List[str]] = None,
        extra_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        msgs = self.pm.get_prompt(
            "refiner",
            "apply_feedback",
            {"question": question, "answer": prior_answer, "feedback": feedback},
        )
        msgs = self._attach_multimodal_parts(msgs, image_urls=image_urls, extra_parts=extra_parts)
        return self.llm.chat(
            msgs,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra=self.extra,
        )

    def run(
        self,
        question: str,
        *,
        # Optional multimodal inputs per step
        gen_image_urls: Optional[List[str]] = None,
        gen_extra_parts: Optional[List[Dict[str, Any]]] = None,
        fb_image_urls: Optional[List[str]] = None,
        fb_extra_parts: Optional[List[Dict[str, Any]]] = None,
        ref_image_urls: Optional[List[str]] = None,
        ref_extra_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> RefinementResult:
        initial = self.generate(
            question,
            image_urls=gen_image_urls,
            extra_parts=gen_extra_parts,
        )
        fb = self.feedback(
            question,
            initial,
            image_urls=fb_image_urls,
            extra_parts=fb_extra_parts,
        )
        refined = self.refine(
            question,
            initial,
            fb,
            image_urls=ref_image_urls,
            extra_parts=ref_extra_parts,
        )
        return RefinementResult(
            question=question,
            initial_answer=initial,
            feedback=fb,
            refined_answer=refined,
        )
