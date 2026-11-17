"""Agents: each one is an LLM call with its own system prompt, returning a Message."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src import prompts
from src.llm import LLMClient, derive_seed
from src.parser import parse_final_answer


@dataclass
class Message:
    agent_id: str  # "solver1", "aggregator", ...
    role: str  # "solver" | "aggregator" | "verifier"
    content: str
    parsed_answer: str | None
    is_faulty: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    finish_reason: str | None = None
    seed: int | None = None
    temperature: float | None = None

    @property
    def is_llm_call(self) -> bool:
        return not self.is_faulty

    def to_dict(self) -> dict:
        return asdict(self)


def run_solver(
    llm: LLMClient,
    question: str,
    *,
    agent_id: str,
    question_id: str,
    base_seed: int,
    temperature: float | None = None,
) -> Message:
    # The seed depends on (question, agent) only, not on the experiment config, so that
    # the same agent draws the same sample across configs (paired comparisons).
    seed = derive_seed(question_id, agent_id, base=base_seed)
    temp = llm.config.temperature if temperature is None else temperature
    out = llm.chat(
        [
            {"role": "system", "content": prompts.SOLVER_SYSTEM},
            {"role": "user", "content": prompts.solver_user(question)},
        ],
        temperature=temp,
        seed=seed,
    )
    return Message(
        agent_id=agent_id,
        role="solver",
        content=out.text,
        parsed_answer=parse_final_answer(out.text),
        prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens,
        latency=round(out.latency, 3),
        finish_reason=out.finish_reason,
        seed=seed,
        temperature=temp,
    )
