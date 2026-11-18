"""Pre-registered confirmatory tests (report/hypotheses.md), with Holm correction.

    python -m scripts.confirmatory results/test/grid.jsonl

Outcome: team adoption = the team's final answer is the injected wrong answer (strict
parser). Unit of analysis: the question. Each hypothesis is a sign-flip permutation test
on per-question differences, averaged over the two fault styles; one-sided or two-sided
as pre-registered (H2 and H4 two-sided since amendment 1).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_utils import read_records
from src.metrics import bootstrap_ci, holm, paired_permutation_test, team_adopted

STYLES = ("neutral", "persuasive")
ALPHA = 0.05
HYPOTHESES = {
    "H1": "chain: team adoption higher with the faulty solver at p1 than at p2/p3",
    "H2": "persuasive vs neutral team adoption differ (chain p1-p3 and star)",
    "H3": "primacy: team adoption higher when the faulty solution is shown first (chain_rotated)",
    "H4": "chain (chronological) vs star team adoption differ",
}
# "greater": the difference is positive; "two-sided": it is not zero.
ALTERNATIVES = {"H1": "greater", "H2": "two-sided", "H3": "greater", "H4": "two-sided"}


def adoption_by_question(records: list[dict]) -> tuple[dict, dict]:
    """{config_id: {question_id: 0/1 team adoption}} and {config_id: {question_id: shown rank}}."""
    adopted, rank = defaultdict(dict), defaultdict(dict)
    for r in records:
        if r.get("fault"):
            adopted[r["config_id"]][r["question_id"]] = float(team_adopted(r))
            rank[r["config_id"]][r["question_id"]] = r["fault"]["shown_at"]
    return adopted, rank


def diffs_h1(a: dict, qids: list[str]) -> np.ndarray:
    cfg = lambda s, p: f"chain-{s}-p{p}-none"  # noqa: E731
    return np.array([np.mean([a[cfg(s, 1)][q] for s in STYLES])
                     - np.mean([a[cfg(s, p)][q] for s in STYLES for p in (2, 3)]) for q in qids])


def diffs_h2(a: dict, qids: list[str]) -> np.ndarray:
    cells = [("chain", p) for p in (1, 2, 3)] + [("star", 1)]
    return np.array([np.mean([a[f"{t}-persuasive-p{p}-none"][q] - a[f"{t}-neutral-p{p}-none"][q] for t, p in cells])
                     for q in qids])


def diffs_h3(a: dict, rank: dict, qids: list[str]) -> np.ndarray:
    out = []
    for q in qids:
        per_style = []
        for s in STYLES:
            by_rank = {rank[f"chain_rotated-{s}-p{p}-none"][q]: a[f"chain_rotated-{s}-p{p}-none"][q] for p in (1, 2, 3)}
            per_style.append(by_rank[1] - (by_rank[2] + by_rank[3]) / 2)
        out.append(np.mean(per_style))
    return np.array(out)


def diffs_h4(a: dict, qids: list[str]) -> np.ndarray:
    return np.array([np.mean([a[f"chain-{s}-p{p}-none"][q] for s in STYLES for p in (1, 2, 3)])
                     - np.mean([a[f"star-{s}-p1-none"][q] for s in STYLES]) for q in qids])


def run(records: list[dict], n_perm: int = 100_000) -> pd.DataFrame:
    a, rank = adoption_by_question(records)
    needed = ([f"chain-{s}-p{p}-none" for s in STYLES for p in (1, 2, 3)]
              + [f"chain_rotated-{s}-p{p}-none" for s in STYLES for p in (1, 2, 3)]
              + [f"star-{s}-p1-none" for s in STYLES])
    missing = [c for c in needed if c not in a]
    if missing:
        raise SystemExit(f"missing configs: {missing}")
    qids = sorted(set.intersection(*(set(a[c]) for c in needed)))
    diffs = {"H1": diffs_h1(a, qids), "H2": diffs_h2(a, qids), "H3": diffs_h3(a, rank, qids), "H4": diffs_h4(a, qids)}
    pvals = {h: paired_permutation_test(d, ALTERNATIVES[h], n_perm=n_perm, seed=0) for h, d in diffs.items()}
    adjusted = holm(pvals)
    rows = []
    for h, d in diffs.items():
        mean, lo, hi = bootstrap_ci(d)
        rows.append({"hypothesis": h, "description": HYPOTHESES[h], "n_questions": len(d),
                     "mean_difference": mean, "ci_low": lo, "ci_high": hi,
                     "alternative": ALTERNATIVES[h], "p_value": pvals[h], "p_holm": adjusted[h],
                     "supported": adjusted[h] < ALPHA})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--save", action="store_true", help="write <log>_confirmatory.csv")
    args = p.parse_args(argv)
    df = run(read_records(args.log))
    for r in df.itertuples():
        print(f"{r.hypothesis}  {r.description}\n    n={r.n_questions}  difference {r.mean_difference:+.3f} "
              f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]  p ({r.alternative})={r.p_value:.4f}  Holm p={r.p_holm:.4f}  "
              f"{'supported' if r.supported else 'not supported'} at alpha={ALPHA}")
    if args.save:
        out = f"{Path(args.log).with_suffix('')}_confirmatory.csv"
        df.to_csv(out, index=False, float_format="%.4f")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
