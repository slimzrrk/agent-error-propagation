"""Defenses against a faulty agent.

skeptical: every agent that reads other agents' work (chained solvers and the aggregator)
is told that it may contain mistakes and to redo the calculations (src/prompts.py).

verifier: a verifier agent checks each solver message as soon as it is written, by
solving the problem itself; the message is VALID if both final answers are equal. A
message judged INVALID is removed from the conversation: later solvers and the
aggregator never see it (the next chain solver reads the last valid solution, or solves
alone). If no solution is left, the aggregator solves the problem alone. If the
verifier gives no valid answer, there is no verdict and the message is kept; a message
without a final answer of its own cannot be checked and is INVALID.
"""

from __future__ import annotations

from src.agents import Message, run_verifier
from src.data import Question
from src.llm import LLMClient
from src.parser import answers_match


def verdict(own_answer: str | None, checked_answer: str | None) -> str | None:
    if own_answer is None:
        return None
    if checked_answer is None:
        return "INVALID"
    return "VALID" if answers_match(own_answer, checked_answer) else "INVALID"


def verify(llm: LLMClient, q: Question, target: Message, *, base_seed: int,
           temperature: float | None = None) -> dict:
    """Verdict record: the verifier's message plus the checked agent and the verdict."""
    msg = run_verifier(llm, q, target, base_seed=base_seed, temperature=temperature)
    return {**msg.to_dict(), "target": target.agent_id, "verdict": verdict(msg.parsed_answer, target.parsed_answer)}


def passes(v: dict | None) -> bool:
    return v is None or v["verdict"] != "INVALID"
