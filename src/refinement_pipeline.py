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
from typing import Dict, Optional

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

    def generate(self, question: str) -> str:
        msgs = self.pm.get_prompt("assistant", "generate", {"question": question})
        return self.llm.chat(
            msgs,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra=self.extra,
        )

    def feedback(self, question: str, answer: str) -> str:
        msgs = self.pm.get_prompt("refiner", "feedback", {"question": question, "answer": answer})
        return self.llm.chat(
            msgs,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra=self.extra,
        )

    def refine(self, question: str, prior_answer: str, feedback: str) -> str:
        msgs = self.pm.get_prompt(
            "refiner",
            "apply_feedback",
            {"question": question, "answer": prior_answer, "feedback": feedback},
        )
        return self.llm.chat(
            msgs,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra=self.extra,
        )

    def run(self, question: str) -> RefinementResult:
        initial = self.generate(question)
        fb = self.feedback(question, initial)
        refined = self.refine(question, initial, fb)
        return RefinementResult(
            question=question,
            initial_answer=initial,
            feedback=fb,
            refined_answer=refined,
        )
