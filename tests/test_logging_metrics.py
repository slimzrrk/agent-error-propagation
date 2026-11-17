import numpy as np

from src.logging_utils import JsonlLog, read_records
from src.metrics import (
    adoption,
    bootstrap_ci,
    bootstrap_ratio_ci,
    downstream,
    invalid_rate,
    majority_vote,
    mcnemar,
    upstream,
)


def msg(agent_id, answer, sees=(), role=None, faulty=False):
    return {"agent_id": agent_id, "role": role or ("aggregator" if agent_id == "aggregator" else "solver"),
            "parsed_answer": answer, "sees": list(sees), "is_faulty": faulty}


def chain(a1, a2, a3, agg, gold="5"):
    return {"gold": gold, "messages": [
        msg("solver1", a1), msg("solver2", a2, ["solver1"]), msg("solver3", a3, ["solver2"]),
        msg("aggregator", agg, ["solver1", "solver2", "solver3"]),
    ]}


def star(a1, a2, a3, agg, gold="5"):
    return {"gold": gold, "messages": [
        msg("solver1", a1), msg("solver2", a2), msg("solver3", a3),
        msg("aggregator", agg, ["solver2", "solver1", "solver3"]),
    ]}


def test_downstream_is_transitive():
    msgs = chain("1", "1", "1", "1")["messages"]
    assert downstream(msgs, "solver1") == {"solver2", "solver3", "aggregator"}
    assert downstream(msgs, "solver3") == {"aggregator"}
    assert downstream(star("1", "1", "1", "1")["messages"], "solver1") == {"aggregator"}


def test_adoption_chain():
    records = [
        chain("7", "7", "5", "5"),  # solver1 wrong, solver2 adopts, then corrected
        chain("7", "7", "7", "7"),  # full propagation
        chain("5", "5", "5", "5"),  # solver1 right: not a case
        chain(None, "5", "5", "5"),  # invalid source: not a case
    ]
    res = adoption(records)
    assert res["n_cases"] == 2
    assert res["adoption_rate"] == 4 / 6
    assert np.isnan(res["control_rate"])  # no independent agent in a chain
    s2 = next(a for a in res["agents"] if a["agent_id"] == "solver2")
    assert s2["relation"] == "downstream" and s2["adopted"] == 1.0


def test_adoption_excludes_upstream_agents():
    # solver2 copied solver1's wrong answer: solver1 is upstream, not a coincidence
    res = adoption([chain("7", "7", "5", "7")], source="solver2")
    s1 = next(a for a in res["agents"] if a["agent_id"] == "solver1")
    assert s1["relation"] == "upstream"
    assert np.isnan(res["control_rate"])
    assert res["adoption_rate"] == 0.5  # solver3 no, aggregator yes
    assert upstream(chain("1", "1", "1", "1")["messages"], "solver3") == {"solver1", "solver2"}


def test_adoption_star_has_coincidence_control():
    res = adoption([star("7", "7", "5", "7"), star("7", "5", "5", "5")])
    assert res["adoption_rate"] == 0.5  # aggregator: 1 of 2
    assert res["control_rate"] == 0.25  # solver2/solver3: 1 of 4


def test_bootstrap_ratio_ci():
    rate, lo, hi = bootstrap_ratio_ci([1, 0, 2, 3], [3, 3, 3, 3])
    assert rate == 0.5 and lo <= rate <= hi
    assert np.isnan(bootstrap_ratio_ci([0], [0])[0])


def test_majority_vote():
    assert majority_vote(["5", "5", "7"]) == "5"
    assert majority_vote(["5", "6", "7"]) is None
    assert majority_vote(["5", None, None]) is None
    assert majority_vote(["5", "5", None]) == "5"
    assert majority_vote([None, None, None]) is None


def test_jsonl_resume(tmp_path):
    path = tmp_path / "runs.jsonl"
    jl = JsonlLog(path)
    jl.write_meta({"model": "m"})
    jl.write_run({"config_id": "a", "question_id": "q1", "correct": True})
    jl.write_run({"config_id": "b", "question_id": "q1", "correct": False})
    with path.open("a") as f:
        f.write('{"record_type": "run", "config_id": "c"')  # interrupted write

    assert JsonlLog(path).done_keys() == {("a", "q1"), ("b", "q1")}
    assert len(read_records(path, "meta")) == 1
    assert len(read_records(path, None)) == 3


def test_bootstrap_ci():
    mean, lo, hi = bootstrap_ci([1] * 70 + [0] * 30)
    assert mean == 0.7
    assert 0.6 < lo < 0.7 < hi < 0.8
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)
    assert np.isnan(bootstrap_ci([])[0])


def test_mcnemar():
    a = [1] * 20 + [0] * 10
    assert mcnemar(a, a)["p_value"] == 1.0
    b = [0] * 15 + [1] * 5 + [0] * 10  # a wins on 15 discordant pairs, b on none
    res = mcnemar(a, b)
    assert (res["only_a"], res["only_b"]) == (15, 0)
    assert res["p_value"] < 0.001


def test_invalid_rate_ignores_faulty_messages():
    records = [{"messages": [
        {"role": "solver", "is_faulty": True, "parsed_answer": None},
        {"role": "solver", "is_faulty": False, "parsed_answer": "3"},
        {"role": "aggregator", "is_faulty": False, "parsed_answer": None},
    ]}]
    assert invalid_rate(records) == 0.5
    assert invalid_rate(records, role="solver") == 0.0


def fault_run(qid, s2, s2_text, agg, gold="160", wrong="80", shown_at=1):
    return {
        "question_id": qid, "gold": gold, "correct": agg == gold,
        "fault": {"agent_id": "solver1", "wrong_answer": wrong, "shown_at": shown_at},
        "messages": [
            {**msg("solver1", wrong, faulty=True), "content": f"Final answer: {wrong}"},
            {**msg("solver2", s2, ["solver1"]), "content": s2_text},
            {**msg("aggregator", agg, ["solver1", "solver2"]), "content": f"Final answer: {agg}"},
        ],
    }


def test_lenient_counts_invalid_outputs_that_copy_the_error():
    # solver2 copies the wrong answer but forgets the "Final answer" line (like train-204)
    r = fault_run("q1", None, "Therefore, each book has on average 80 pages.", "160")
    strict = adoption([r], source="solver1")
    lenient = adoption([r], source="solver1", lenient=True)
    assert strict["adoption_rate"] == 0.0
    assert lenient["adoption_rate"] == 0.5  # solver2 yes, aggregator no


def test_lenient_accuracy_and_team_adoption():
    from src.metrics import is_correct, team_adopted

    r = fault_run("q1", "80", "Final answer: 80", None)
    r["messages"][-1]["content"] = "So the answer is 80 pages."
    assert not is_correct(r) and not team_adopted(r)
    assert team_adopted(r, lenient=True)


def test_fault_summary_and_slots():
    from src.metrics import by_slot, fault_summary

    records = [fault_run("q1", "80", "", "80", shown_at=1), fault_run("q2", "160", "", "160", shown_at=2)]
    s = fault_summary(records)
    assert s["recovery"][0] == 0.5 and s["team_adopted"][0] == 0.5
    assert s["downstream_adoption"][0] == 0.5
    slots = by_slot(records)
    assert [(x["group"], x["team_adopted"][0]) for x in slots] == [(1, 1.0), (2, 0.0)]


def test_paired_outcomes():
    from src.metrics import paired_outcomes

    a = [{"question_id": f"q{i}", "correct": i < 8} for i in range(10)]
    b = [{"question_id": f"q{i}", "correct": i < 3} for i in range(10)]
    res = paired_outcomes(a, b, lambda r: r["correct"])
    assert res["n"] == 10 and (res["only_a"], res["only_b"]) == (5, 0)
    assert res["rate_a"] == 0.8 and res["rate_b"] == 0.3


def test_protocol_fingerprint_changes_with_protocol_files(tmp_path, monkeypatch):
    import src.logging_utils as lu

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "prompts.py").write_text("A = 1\n")
    monkeypatch.setattr(lu, "ROOT", tmp_path)
    before = lu.protocol_fingerprint()
    assert before == lu.protocol_fingerprint()
    (tmp_path / "src" / "prompts.py").write_text("A = 2\n")
    assert lu.protocol_fingerprint() != before


def test_breakdown_by_slip_position():
    from src.metrics import by_slip_position

    early = fault_run("q1", "80", "", "80")
    late = fault_run("q2", "160", "", "160")
    early["fault"]["slip_position"] = {"relative": 0.25, "category": "early"}
    late["fault"]["slip_position"] = {"relative": 1.0, "category": "late"}
    rows = {x["group"]: x for x in by_slip_position([early, late])}
    assert rows["early"]["downstream_adoption"][0] == 1.0
    assert rows["late"]["downstream_adoption"][0] == 0.0 and rows["late"]["recovery"][0] == 1.0


def test_majority_vs_aggregator():
    from src.metrics import majority_vs_aggregator

    # faulty solver1 says 80, both others say 160: majority is right, the aggregator adopted 80
    r = fault_run("q1", "160", "", "80")
    r["messages"].insert(2, {**msg("solver3", "160", ["solver2"]), "content": "Final answer: 160"})
    res = majority_vs_aggregator([r])
    assert res["majority_acc"][0] == 1.0 and res["aggregator_acc"][0] == 0.0
    assert res["majority_adopted"][0] == 0.0 and res["aggregator_adopted"][0] == 1.0


def test_paired_permutation_test():
    from src.metrics import paired_permutation_test

    assert paired_permutation_test([1.0] * 20, n_perm=5000) < 0.001
    assert paired_permutation_test([-1.0] * 20, n_perm=5000) > 0.99
    assert paired_permutation_test([0.0] * 20, n_perm=5000) == 1.0
    p = paired_permutation_test([1, -1] * 10, alternative="two-sided", n_perm=5000)
    assert p > 0.5


def test_holm():
    from src.metrics import holm

    adj = holm({"H1": 0.01, "H2": 0.04, "H3": 0.03, "H4": 0.5})
    assert adj == {"H1": 0.04, "H3": 0.09, "H2": 0.09, "H4": 0.5}
