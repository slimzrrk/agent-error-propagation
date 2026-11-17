import pytest

from src.data import load_questions, make_splits, parse_gold, sample_indices, save_splits

ROWS = {
    "train": [{"question": f"train q{i}", "answer": f"step {i}\n#### {i * 10:,}"} for i in range(100)],
    "test": [{"question": f"test q{i}", "answer": f"step {i}\n#### {i}"} for i in range(60)],
}


def fake_loader(split):
    return ROWS[split]


def test_parse_gold():
    rationale, gold = parse_gold("She pays 2*3=<<2*3=6>>6 dollars.\n#### 6")
    assert gold == "6"
    assert rationale == "She pays 2*3=<<2*3=6>>6 dollars."
    assert parse_gold("x\n#### 1,000")[1] == "1000"


def test_parse_gold_requires_marker():
    with pytest.raises(ValueError):
        parse_gold("no final marker 12")


def test_sample_indices_deterministic_and_unique():
    a = sample_indices(1000, 50, seed=42)
    assert a == sample_indices(1000, 50, seed=42)
    assert a != sample_indices(1000, 50, seed=43)
    assert len(set(a)) == 50 and a == sorted(a)


def test_make_and_load_splits(tmp_path):
    splits = make_splits(seed=42, sizes={"test": 20, "dev": 5}, loader=fake_loader)
    assert len(splits["splits"]["test"]["ids"]) == 20
    assert all(i.startswith("train-") for i in splits["splits"]["dev"]["ids"])

    path = tmp_path / "splits.json"
    save_splits(splits, path)
    dev = load_questions("dev", path=path, loader=fake_loader)
    assert [q.id for q in dev] == splits["splits"]["dev"]["ids"]
    q = dev[0]
    idx = int(q.id.split("-")[1])
    assert q.question == f"train q{idx}"
    assert q.gold == str(idx * 10)  # commas removed
    assert len(load_questions("test", limit=3, path=path, loader=fake_loader)) == 3
