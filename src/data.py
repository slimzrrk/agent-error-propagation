"""GSM8K loading and the fixed dev/test splits used in all experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from src.parser import normalize_number

DATASET = "openai/gsm8k"
CONFIG = "main"
SPLITS_PATH = Path(__file__).resolve().parent.parent / "data" / "splits.json"

# Our split name -> (HF split it is drawn from, size)
SPLIT_SPECS = {
    "test": ("test", 200),
    "dev": ("train", 50),
}


@dataclass(frozen=True)
class Question:
    id: str  # e.g. "test-1093": HF split + row index
    question: str
    rationale: str  # gold step-by-step solution, without the "#### <answer>" line
    gold: str  # canonical number
    rank: int | None = None  # position in a seeded shuffle of the whole split (balanced designs)


def split_ranks(ids: list[str], seed: int = 42) -> dict[str, int]:
    """Rank of each question in a seeded shuffle of its full split, independent of --limit."""
    order = list(ids)
    random.Random(seed).shuffle(order)
    return {qid: i for i, qid in enumerate(order)}


def parse_gold(answer_field: str) -> tuple[str, str]:
    """Split a GSM8K answer field into (rationale, canonical gold number)."""
    rationale, sep, final = answer_field.rpartition("####")
    if not sep:
        raise ValueError("GSM8K answer without '####'")
    gold = normalize_number(final)
    if gold is None:
        raise ValueError(f"Unparsable gold answer: {final!r}")
    return rationale.strip(), gold


def load_hf_split(hf_split: str):
    from datasets import load_dataset  # heavy import, only when needed

    return load_dataset(DATASET, CONFIG, split=hf_split)


def sample_indices(n_total: int, k: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_total), k))


def make_splits(seed: int = 42, sizes: dict[str, int] | None = None, loader=load_hf_split) -> dict:
    out = {"dataset": DATASET, "config": CONFIG, "seed": seed, "splits": {}}
    for name, (hf_split, default_size) in SPLIT_SPECS.items():
        k = (sizes or {}).get(name, default_size)
        n_total = len(loader(hf_split))
        idx = sample_indices(n_total, k, seed)
        out["splits"][name] = {
            "source_split": hf_split,
            "source_size": n_total,
            "ids": [f"{hf_split}-{i}" for i in idx],
        }
    return out


def save_splits(splits: dict, path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n")


def load_splits(path: Path = SPLITS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `python -m src.data` first")
    return json.loads(path.read_text())


def load_questions(split: str, limit: int | None = None, path: Path = SPLITS_PATH, loader=load_hf_split) -> list[Question]:
    """Questions of our 'dev' or 'test' split, in the order stored in splits.json."""
    spec = load_splits(path)["splits"][split]
    ranks = split_ranks(spec["ids"])
    ids = spec["ids"][:limit] if limit else spec["ids"]
    rows = loader(spec["source_split"])
    questions = []
    for qid in ids:
        row = rows[int(qid.rsplit("-", 1)[1])]
        rationale, gold = parse_gold(row["answer"])
        questions.append(Question(id=qid, question=row["question"], rationale=rationale, gold=gold, rank=ranks[qid]))
    return questions


if __name__ == "__main__":
    splits = make_splits()
    save_splits(splits)
    for name, s in splits["splits"].items():
        print(f"{name}: {len(s['ids'])} questions from HF '{s['source_split']}' ({s['source_size']} rows)")
    print(f"saved to {SPLITS_PATH}")
