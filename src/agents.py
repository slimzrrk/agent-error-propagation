"""Agents: each one is an LLM call with its own system prompt, returning a Message."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src import prompts
from src.data import Question
from src.llm import LLMClient, derive_seed
from src.parser import parse_final_answer


@dataclass
class Message:
    agent_id: str  # "solver1", "aggregator", ...
    role: str  # "solver" | "aggregator" | "verifier"
    content: str
    parsed_answer: str | None
    is_faulty: bool = False
    sees: list[str] = field(default_factory=list)  # agents whose messages were in the prompt
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


def _llm_message(
    llm: LLMClient,
    q: Question,
    *,
    agent_id: str,
    role: str,
    system: str,
    user: str,
    sees: list[str],
    base_seed: int,
    temperature: float | None,
) -> Message:
    # The seed depends on (question, agent) only, not on the experiment config, so that
    # the same agent draws the same sample across configs (paired comparisons).
    seed = derive_seed(q.id, agent_id, base=base_seed)
    temp = llm.config.temperature if temperature is None else temperature
    out = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temp,
        seed=seed,
    )
    return Message(
        agent_id=agent_id,
        role=role,
        content=out.text,
        parsed_answer=parse_final_answer(out.text),
        sees=sees,
        prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens,
        latency=round(out.latency, 3),
        finish_reason=out.finish_reason,
        seed=seed,
        temperature=temp,
    )


def run_solver(
    llm: LLMClient,
    q: Question,
    *,
    agent_id: str,
    base_seed: int,
    temperature: float | None = None,
    previous: Message | None = None,
    skeptical: bool = False,
) -> Message:
    return _llm_message(
        llm, q,
        agent_id=agent_id,
        role="solver",
        system=prompts.SOLVER_SYSTEM,
        user=prompts.solver_user(q.question, previous.content if previous else None, skeptical),
        sees=[previous.agent_id] if previous else [],
        base_seed=base_seed,
        temperature=temperature,
    )


def run_aggregator(
    llm: LLMClient,
    q: Question,
    solutions: list[Message],
    *,
    base_seed: int,
    temperature: float | None = None,
    skeptical: bool = False,
) -> Message:
    """`solutions` are shown in the given order, labelled 1..n (no agent names).
    With no solution at all (every one rejected by the verifier) it solves the problem alone."""
    if not solutions:
        system, user = prompts.SOLVER_SYSTEM, prompts.solver_user(q.question)
    else:
        system = prompts.AGGREGATOR_SYSTEM
        user = prompts.aggregator_user(q.question, [m.content for m in solutions], skeptical)
    return _llm_message(
        llm, q,
        agent_id="aggregator",
        role="aggregator",
        system=system,
        user=user,
        sees=[m.agent_id for m in solutions],
        base_seed=base_seed,
        temperature=temperature,
    )


def run_verifier(
    llm: LLMClient,
    q: Question,
    target: Message,
    *,
    base_seed: int,
    temperature: float | None = None,
) -> Message:
    """Checks one solver message; verifier i checks solver i (its seed follows the checked agent)."""
    return _llm_message(
        llm, q,
        agent_id=target.agent_id.replace("solver", "verifier"),
        role="verifier",
        system=prompts.VERIFIER_SYSTEM,
        user=prompts.verifier_user(q.question, target.content),
        sees=[target.agent_id],
        base_seed=base_seed,
        temperature=temperature,
    )
