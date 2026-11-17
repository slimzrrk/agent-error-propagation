"""Team topologies. Each runner takes one question and returns a result dict (one JSONL line)."""

from __future__ import annotations

import random
from collections import Counter

from src.agents import Message, run_aggregator, run_solver
from src.config import ExperimentConfig
from src.data import Question
from src.llm import LLMClient, derive_seed
from src.parser import answers_match

N_SOLVERS = 3


def run_single(llm: LLMClient, q: Question, exp: ExperimentConfig, base_seed: int) -> dict:
    # Same agent id and seed as solver1 in the teams, so the baseline is directly comparable.
    msg = run_solver(llm, q, agent_id="solver1", base_seed=base_seed, temperature=exp.temperature)
    return build_result(q, [msg], final_answer=msg.parsed_answer)


def run_chain(llm: LLMClient, q: Question, exp: ExperimentConfig, base_seed: int) -> dict:
    """Solver1 -> Solver2 -> Solver3, each sees the previous solution; the aggregator sees all three.

    The aggregator gets them in chain order: in a chain the order carries meaning
    (each solution is a revision of the previous one).
    """
    solvers: list[Message] = []
    previous = None
    for i in range(1, N_SOLVERS + 1):
        msg = run_solver(llm, q, agent_id=f"solver{i}", base_seed=base_seed,
                         temperature=exp.temperature, previous=previous)
        solvers.append(msg)
        previous = msg
    agg = run_aggregator(llm, q, solvers, base_seed=base_seed, temperature=exp.temperature)
    return build_result(q, solvers + [agg], final_answer=agg.parsed_answer)


def presentation_order(question_id: str, agent_ids: list[str], base_seed: int) -> list[str]:
    """Deterministic per-question shuffle, so no solver is always shown first."""
    order = list(agent_ids)
    random.Random(derive_seed(question_id, "presentation", base=base_seed)).shuffle(order)
    return order


def run_star(llm: LLMClient, q: Question, exp: ExperimentConfig, base_seed: int) -> dict:
    """Three independent solvers; the aggregator sees them in a shuffled order."""
    solvers = [
        run_solver(llm, q, agent_id=f"solver{i}", base_seed=base_seed, temperature=exp.temperature)
        for i in range(1, N_SOLVERS + 1)
    ]
    by_id = {m.agent_id: m for m in solvers}
    shown = [by_id[a] for a in presentation_order(q.id, list(by_id), base_seed)]
    agg = run_aggregator(llm, q, shown, base_seed=base_seed, temperature=exp.temperature)
    return build_result(q, solvers + [agg], final_answer=agg.parsed_answer)


RUNNERS = {"single": run_single, "chain": run_chain, "star": run_star}


def run_question(llm: LLMClient, q: Question, exp: ExperimentConfig, base_seed: int) -> dict:
    if exp.fault != "none" or exp.defense != "none":
        raise NotImplementedError("faults and defenses are not implemented yet")
    return RUNNERS[exp.topology](llm, q, exp, base_seed)


def call_kinds(exp: ExperimentConfig) -> Counter:
    """LLM calls per question by kind, used for cost estimates.

    solo = solver without a previous solution, chained = solver that reads one,
    verifier calls are counted separately. The scripted faulty solver costs nothing.
    """
    if exp.topology == "single":
        return Counter(solo=1)
    faulty = exp.position if exp.fault != "none" else None
    kinds = Counter(aggregator=1)
    for i in range(1, N_SOLVERS + 1):
        if i == faulty:
            continue
        kinds["chained" if exp.topology == "chain" and i > 1 else "solo"] += 1
    if exp.defense == "verifier":
        kinds["verifier"] = N_SOLVERS
    return kinds


def expected_llm_calls(exp: ExperimentConfig) -> int:
    return sum(call_kinds(exp).values())


def build_result(q: Question, messages: list[Message], final_answer: str | None, verdicts: list | None = None) -> dict:
    llm_msgs = [m for m in messages if m.is_llm_call]
    return {
        "question_id": q.id,
        "question": q.question,
        "gold": q.gold,
        "messages": [m.to_dict() for m in messages],
        "verdicts": verdicts or [],
        "final_answer": final_answer,
        "correct": answers_match(final_answer, q.gold),
        "n_llm_calls": len(llm_msgs),
        "prompt_tokens": sum(m.prompt_tokens for m in llm_msgs),
        "completion_tokens": sum(m.completion_tokens for m in llm_msgs),
    }
