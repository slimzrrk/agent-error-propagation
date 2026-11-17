import json
from collections import Counter

import pytest

import src.faults
from src.config import ExperimentConfig
from src.data import Question
from src.faults import PERSUASIVE_SENTENCE
from src.llm import Completion, LLMConfig
from src.topologies import call_kinds, expected_llm_calls, presentation_order, run_question

FAULT_TEXT = "The sum is 2 + 3 = 6.\n\nFinal answer: 6"


@pytest.fixture(autouse=True)
def fake_faults(tmp_path, monkeypatch):
    path = tmp_path / "faults.json"
    path.write_text(json.dumps({"faults": {"test-7": {"text": FAULT_TEXT, "wrong_answer": "6"}}}))
    monkeypatch.setattr(src.faults, "FAULTS_PATH", path)
    src.faults._cache.clear()
    yield
    src.faults._cache.clear()


class FakeLLM:
    """Answers with a number chosen by `answer_fn(system, user)` and records every prompt."""

    def __init__(self, answer_fn=lambda system, user: "5"):
        self.config = LLMConfig()
        self.answer_fn = answer_fn
        self.calls = []

    def chat(self, messages, temperature=None, seed=None, max_tokens=None):
        system, user = messages[0]["content"], messages[1]["content"]
        self.calls.append({"system": system, "user": user, "seed": seed, "temperature": temperature})
        text = f"reply #{len(self.calls)}\nFinal answer: {self.answer_fn(system, user)}"
        return Completion(text=text, prompt_tokens=10, completion_tokens=3, latency=0.01, finish_reason="stop")


Q = Question(id="test-7", question="What is 2 + 3?", rationale="2 + 3 = 5", gold="5")


def test_single_matches_solver1_of_teams():
    llm = FakeLLM()
    single = run_question(llm, Q, ExperimentConfig("single"), base_seed=42)
    star = run_question(llm, Q, ExperimentConfig("star"), base_seed=42)
    chain = run_question(llm, Q, ExperimentConfig("chain"), base_seed=42)
    s1 = [c for c in llm.calls if c["seed"] == single["messages"][0]["seed"]]
    # same seed and same prompt for the single agent and solver1 of both teams
    assert len(s1) == 3 and len({(c["system"], c["user"]) for c in s1}) == 1
    assert star["messages"][0]["seed"] == chain["messages"][0]["seed"]


def test_chain_passes_previous_solution():
    llm = FakeLLM()
    r = run_question(llm, Q, ExperimentConfig("chain"), base_seed=42)
    msgs = r["messages"]
    assert [m["agent_id"] for m in msgs] == ["solver1", "solver2", "solver3", "aggregator"]
    assert [m["sees"] for m in msgs] == [[], ["solver1"], ["solver2"], ["solver1", "solver2", "solver3"]]
    assert "reply #1" not in llm.calls[0]["user"]
    assert "reply #1" in llm.calls[1]["user"] and "reply #2" in llm.calls[2]["user"]
    assert "reply #1" not in llm.calls[2]["user"]  # solver3 only sees solver2
    agg_prompt = llm.calls[3]["user"]
    assert agg_prompt.index("reply #1") < agg_prompt.index("reply #2") < agg_prompt.index("reply #3")
    assert r["n_llm_calls"] == 4 and r["correct"] and r["final_answer"] == "5"
    assert r["prompt_tokens"] == 40 and r["completion_tokens"] == 12


def test_star_solvers_are_independent_and_order_is_shuffled():
    llm = FakeLLM()
    r = run_question(llm, Q, ExperimentConfig("star"), base_seed=42)
    assert all("<solution" not in c["user"] for c in llm.calls[:3])
    agg = r["messages"][-1]
    assert sorted(agg["sees"]) == ["solver1", "solver2", "solver3"]
    assert agg["sees"] == presentation_order(Q, ["solver1", "solver2", "solver3"], 42)
    # the aggregator prompt follows the logged order
    prompt = llm.calls[3]["user"]
    shown = sorted(range(3), key=lambda i: prompt.index(f"reply #{i + 1}"))
    assert [f"solver{i + 1}" for i in shown] == agg["sees"]


def test_presentation_order_rotates_all_six_orders():
    ids = ["solver1", "solver2", "solver3"]
    orders = [tuple(presentation_order(Question("q", "", "", "1", rank=r), ids, 42)) for r in range(6)]
    assert len(set(orders)) == 6
    slots = Counter(presentation_order(Question("q", "", "", "1", rank=r), ids, 42).index("solver1")
                    for r in range(200))
    assert sorted(slots.values()) == [66, 67, 67]


@pytest.mark.parametrize("position", [1, 2, 3])
def test_chain_fault_replaces_solver(position):
    llm = FakeLLM()
    exp = ExperimentConfig("chain", "neutral", position)
    r = run_question(llm, Q, exp, base_seed=42)
    msgs = {m["agent_id"]: m for m in r["messages"]}
    faulty = msgs[f"solver{position}"]
    assert faulty["is_faulty"] and faulty["content"] == FAULT_TEXT and faulty["sees"] == []
    assert faulty["parsed_answer"] == "6" and faulty["prompt_tokens"] == 0
    assert r["n_llm_calls"] == 3 == len(llm.calls)
    if position < 3:  # the next solver reads the faulty solution
        assert FAULT_TEXT in llm.calls[position - 1]["user"]
    assert r["fault"]["wrong_answer"] == "6" and r["fault"]["shown_at"] == position
    assert r["fault"]["faults_sha"]


def test_star_persuasive_fault():
    llm = FakeLLM()
    r = run_question(llm, Q, ExperimentConfig("star", "persuasive", 1), base_seed=42)
    faulty = r["messages"][0]
    assert faulty["content"] == FAULT_TEXT.replace("\n\nFinal", f"\n\n{PERSUASIVE_SENTENCE}\nFinal")
    agg = r["messages"][-1]
    assert r["fault"]["shown_at"] == agg["sees"].index("solver1") + 1
    assert PERSUASIVE_SENTENCE in llm.calls[-1]["user"]


def test_wrong_aggregator_makes_team_wrong():
    llm = FakeLLM(lambda system, user: "9" if "coordinate" in system else "5")
    r = run_question(llm, Q, ExperimentConfig("star"), base_seed=42)
    assert r["final_answer"] == "9" and not r["correct"]


def test_call_kinds():
    assert call_kinds(ExperimentConfig("chain")) == Counter(solo=1, chained=2, aggregator=1)
    assert call_kinds(ExperimentConfig("star")) == Counter(solo=3, aggregator=1)
    assert call_kinds(ExperimentConfig("chain", "neutral", 1)) == Counter(chained=2, aggregator=1)
    assert call_kinds(ExperimentConfig("chain", "neutral", 3)) == Counter(solo=1, chained=1, aggregator=1)
    assert expected_llm_calls(ExperimentConfig("chain", "none", None, "verifier")) == 7
