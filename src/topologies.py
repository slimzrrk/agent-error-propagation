"""Team topologies. Each runner takes one question and returns a result dict (one JSONL line)."""

from __future__ import annotations

from src.agents import Message, run_solver
from src.config import ExperimentConfig
from src.data import Question
from src.llm import LLMClient
from src.parser import answers_match


def run_single(llm: LLMClient, q: Question, exp: ExperimentConfig, base_seed: int) -> dict:
    # Same agent id and seed as solver1 in the teams, so the baseline is directly comparable.
    msg = run_solver(llm, q.question, agent_id="solver1", question_id=q.id, base_seed=base_seed, temperature=exp.temperature)
    return build_result(q, [msg], final_answer=msg.parsed_answer)


RUNNERS = {"single": run_single}


def expected_llm_calls(exp: ExperimentConfig) -> int:
    """LLM calls per question: 3 solvers (minus the scripted faulty one) + aggregator,
    plus one verifier call per solver message with the verifier defense."""
    if exp.topology == "single":
        return 1
    solvers = 3 - (exp.fault != "none")
    verifier = 3 if exp.defense == "verifier" else 0
    return solvers + 1 + verifier


def run_question(llm: LLMClient, q: Question, exp: ExperimentConfig, base_seed: int) -> dict:
    try:
        runner = RUNNERS[exp.topology]
    except KeyError:
        raise NotImplementedError(f"topology {exp.topology!r} not implemented yet") from None
    return runner(llm, q, exp, base_seed)


def build_result(q: Question, messages: list[Message], final_answer: str | None, verdicts: list | None = None) -> dict:
    llm_msgs = [m for m in messages if m.is_llm_call]
    return {
        "question_id": q.id,
        "gold": q.gold,
        "messages": [m.to_dict() for m in messages],
        "verdicts": verdicts or [],
        "final_answer": final_answer,
        "correct": answers_match(final_answer, q.gold),
        "n_llm_calls": len(llm_msgs),
        "prompt_tokens": sum(m.prompt_tokens for m in llm_msgs),
        "completion_tokens": sum(m.completion_tokens for m in llm_msgs),
    }
