from scripts.run_experiment import same_outputs, select_reusable
from src.config import ExperimentConfig
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
