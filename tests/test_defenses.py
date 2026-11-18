import pytest

from src import prompts
from src.config import ExperimentConfig
from src.defenses import verdict
from src.topologies import expected_llm_calls, run_question
from tests.test_topologies import Q, FakeLLM, fake_faults  # noqa: F401


@pytest.mark.parametrize("own, checked, expected", [
    ("5", "5", "VALID"), ("5", "5.0", "VALID"), ("5", "6", "INVALID"),
    ("5", None, "INVALID"),  # nothing to check against
    (None, "5", None),  # no verdict: the message is kept
])
def test_verdict_compares_answers(own, checked, expected):
    assert verdict(own, checked) == expected


def test_no_defense_prompts_are_unchanged():
    # Exactly the phase 3 prompts, so runs without defense stay comparable across phases.
    assert prompts.solver_user("Q?", "S") == (
        "Problem: Q?\n\nAnother agent proposed this solution:\n<solution>\nS\n</solution>\n\n"
        "You may keep this answer or revise it. Write your own step-by-step solution.\n"
        'Remember to end with the line "Final answer: <number>".')
    assert prompts.aggregator_user("Q?", ["A", "B"]) == (
        'Problem: Q?\n\n<solution id="1">\nA\n</solution>\n\n<solution id="2">\nB\n</solution>\n\n'
        "Compare the solutions briefly, then give your decision.\n"
        'Remember to end with the line "Final answer: <number>".')


def test_skeptical_only_changes_agents_that_read_others():
    plain, skeptical = FakeLLM(), FakeLLM()
    run_question(plain, Q, ExperimentConfig("chain"), base_seed=0)
    run_question(skeptical, Q, ExperimentConfig("chain", defense="skeptical"), base_seed=0)
    has = [prompts.SKEPTICAL_INSTRUCTION in c["user"] for c in skeptical.calls]
    assert has == [False, True, True, True]  # solver1 reads nobody
    assert skeptical.calls[0] == plain.calls[0]
    star = FakeLLM()
    run_question(star, Q, ExperimentConfig("star", defense="skeptical"), base_seed=0)
    assert [prompts.SKEPTICAL_INSTRUCTION in c["user"] for c in star.calls] == [False, False, False, True]


def verifier_llm(own_answer="5"):
    """Solvers answer 5 (right), the scripted fault says 6; the verifier finds own_answer."""
    return FakeLLM(lambda system, user: own_answer if system == prompts.VERIFIER_SYSTEM else "5")


def test_chain_verifier_drops_rejected_solution():
    llm = verifier_llm()
    exp = ExperimentConfig("chain", "neutral", 1, "verifier")
    r = run_question(llm, Q, exp, base_seed=0)
    verdicts = {v["target"]: v["verdict"] for v in r["verdicts"]}
    assert verdicts == {"solver1": "INVALID", "solver2": "VALID", "solver3": "VALID"}
    msgs = {m["agent_id"]: m for m in r["messages"]}
    assert msgs["solver2"]["sees"] == []  # solves alone, the faulty solution was dropped
    assert msgs["solver3"]["sees"] == ["solver2"]
    assert msgs["aggregator"]["sees"] == ["solver2", "solver3"]
    assert r["fault"]["shown_at"] is None
    assert r["n_llm_calls"] == expected_llm_calls(exp) == 6
    assert r["prompt_tokens"] == 60  # verifier calls are counted in the cost


def test_chain_verifier_skips_to_last_accepted_solution():
    llm = verifier_llm()
    r = run_question(llm, Q, ExperimentConfig("chain", "neutral", 2, "verifier"), base_seed=0)
    msgs = {m["agent_id"]: m for m in r["messages"]}
    assert msgs["solver3"]["sees"] == ["solver1"]
    assert msgs["aggregator"]["sees"] == ["solver1", "solver3"]


def test_star_aggregator_solves_alone_when_everything_is_rejected():
    llm = verifier_llm(own_answer="7")
    r = run_question(llm, Q, ExperimentConfig("star", defense="verifier"), base_seed=0)
    agg = r["messages"][-1]
    assert agg["role"] == "aggregator" and agg["sees"] == []
    assert llm.calls[-1]["system"] == prompts.SOLVER_SYSTEM
    assert len(r["verdicts"]) == 3 and all(v["verdict"] == "INVALID" for v in r["verdicts"])


def test_unparsed_verdict_keeps_the_message():
    r = run_question(verifier_llm(own_answer=""), Q, ExperimentConfig("star", "neutral", 1, "verifier"), base_seed=0)
    assert all(v["verdict"] is None for v in r["verdicts"])
    assert len(r["messages"][-1]["sees"]) == 3


def test_verifier_seed_follows_the_checked_agent():
    a, b = verifier_llm(), verifier_llm()
    run_question(a, Q, ExperimentConfig("star", defense="verifier"), base_seed=0)
    run_question(b, Q, ExperimentConfig("star", "persuasive", 1, "verifier"), base_seed=0)
    seeds = lambda llm: [c["seed"] for c in llm.calls if c["system"] == prompts.VERIFIER_SYSTEM]  # noqa: E731
    assert seeds(a) == seeds(b) and len(set(seeds(a))) == 3
