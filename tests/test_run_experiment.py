import json
from types import SimpleNamespace

import pytest

from scripts.run_experiment import check_test_run, same_outputs, select_reusable
from src.config import ExperimentConfig, RunConfig, full_grid
from src.data import Question


def old_run(cid, qid, faulty_text=None, wrong="7"):
    msgs = [{"agent_id": "solver1", "content": "a", "is_faulty": False}]
    if faulty_text is not None:
        msgs.append({"agent_id": "solver2", "content": faulty_text, "is_faulty": True})
    return {"config_id": cid, "question_id": qid, "messages": msgs, "final_answer": "5",
            "fault": {"wrong_answer": wrong} if faulty_text is not None else None}


def test_select_reusable_only_unchanged_inputs():
    nofault = ExperimentConfig("star")
    fault = ExperimentConfig("chain", "neutral", 2)
    q1, q2, q3 = (Question(f"q{i}", "?", "", "5") for i in (1, 2, 3))
    old = [old_run(nofault.id, "q1"), old_run(fault.id, "q1", "same text"), old_run(fault.id, "q2", "old text"),
           old_run(fault.id, "q3", "same text", wrong="8")]
    tasks = [(nofault, q1), (nofault, q2), (fault, q1), (fault, q2), (fault, q3)]
    texts = {"q1": ("same text", "7"), "q2": ("new text", "7"), "q3": ("same text", "7")}
    reusable = select_reusable(old, tasks, lambda qid, style: texts[qid])
    # q2 never ran without fault; its fault text changed; q3's injected answer changed.
    assert set(reusable) == {(nofault.id, "q1"), (fault.id, "q1")}


def test_same_outputs_compares_every_message_and_verdict():
    a = old_run("c", "q1")
    assert same_outputs(a, {**a})
    assert not same_outputs(a, {**a, "messages": [{**a["messages"][0], "content": "b"}]})
    assert not same_outputs({**a, "verdicts": [{"content": "VALID"}]}, {**a, "verdicts": [{"content": "INVALID"}]})


@pytest.fixture
def prereg(tmp_path):
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps({"protocol_sha": "abc123", "amendment": 3, "config": "configs/grid.yaml"}))
    return path


def test_test_run_needs_the_preregistered_protocol(prereg):
    cfg = RunConfig(name="grid", experiments=full_grid())
    args = SimpleNamespace(limit=None, reuse=None, config="configs/grid.yaml")
    assert check_test_run("abc123", cfg, args, prereg)["amendment"] == 3
    with pytest.raises(SystemExit, match="differs from the pre-registered"):
        check_test_run("abc124", cfg, args, prereg)
    with pytest.raises(SystemExit, match="no --limit"):
        check_test_run("abc123", cfg, SimpleNamespace(limit=5, reuse=None, config="x"), prereg)
    with pytest.raises(SystemExit, match="full grid"):
        check_test_run("abc123", RunConfig(name="x", experiments=full_grid()[:-1]), args, prereg)
