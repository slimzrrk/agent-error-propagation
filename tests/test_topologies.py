from collections import Counter

from src.config import ExperimentConfig
from src.data import Question
from src.llm import Completion, LLMConfig
from src.topologies import call_kinds, expected_llm_calls, presentation_order, run_question


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
    assert agg["sees"] == presentation_order(Q.id, ["solver1", "solver2", "solver3"], 42)
    # the aggregator prompt follows the logged order
    prompt = llm.calls[3]["user"]
    shown = sorted(range(3), key=lambda i: prompt.index(f"reply #{i + 1}"))
    assert [f"solver{i + 1}" for i in shown] == agg["sees"]


def test_presentation_order_is_deterministic_and_balanced():
    ids = ["solver1", "solver2", "solver3"]
    assert presentation_order("q1", ids, 42) == presentation_order("q1", ids, 42)
    first = Counter(presentation_order(f"test-{i}", ids, 42).index("solver1") for i in range(300))
    assert set(first) == {0, 1, 2} and min(first.values()) > 70


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
