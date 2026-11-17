import numpy as np

from src.logging_utils import JsonlLog, read_records
from src.metrics import bootstrap_ci, invalid_rate, mcnemar


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
