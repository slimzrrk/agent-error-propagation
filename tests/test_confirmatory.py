from scripts.confirmatory import run

STYLES = ("neutral", "persuasive")


def record(config_id, qid, adopted, shown_at, majority_adopts=False):
    final = "99" if adopted else "5"
    honest = "99" if majority_adopts else "5"
    solvers = [{"agent_id": f"solver{i}", "role": "solver", "parsed_answer": a, "content": f"Final answer: {a}",
                "is_faulty": i == 1, "sees": []} for i, a in enumerate(("99", honest, "5"), 1)]
    return {
        "record_type": "run", "config_id": config_id, "question_id": qid, "gold": "5",
        "fault": {"wrong_answer": "99", "shown_at": shown_at},
        "messages": solvers + [{"agent_id": "aggregator", "role": "aggregator", "parsed_answer": final,
                                "content": f"Final answer: {final}", "is_faulty": False, "sees": []}],
    }


def synthetic(n=40):
    """Adoption only when the faulty solution is shown first (chain shows p1 first); the
    verifier removes every adoption, the skeptical instruction changes nothing, and the
    majority vote never adopts."""
    recs = []
    for i in range(n):
        q = f"q{i}"
        for s in STYLES:
            for p in (1, 2, 3):
                for d in ("none", "skeptical", "verifier"):
                    recs.append(record(f"chain-{s}-p{p}-{d}", q, adopted=p == 1 and d != "verifier", shown_at=p))
                rank = (i + p) % 3 + 1  # Latin square
                recs.append(record(f"chain_rotated-{s}-p{p}-none", q, adopted=rank == 1, shown_at=rank))
            for d in ("none", "skeptical", "verifier"):
                recs.append(record(f"star-{s}-p1-{d}", q, adopted=False, shown_at=i % 3 + 1))
    return recs


def test_confirmatory_on_synthetic_data():
    df = run(synthetic(), n_perm=2000).set_index("hypothesis")
    assert list(df.index) == [f"H{k}" for k in range(1, 9)]
    assert df.loc["H1", "mean_difference"] == 1.0 and df.loc["H1", "supported"]
    assert df.loc["H3", "mean_difference"] == 1.0 and df.loc["H3", "supported"]
    assert df.loc["H2", "mean_difference"] == 0.0 and not df.loc["H2", "supported"]
    assert abs(df.loc["H4", "mean_difference"] - 1 / 3) < 1e-9 and df.loc["H4", "supported"]
    # Chain p1 adopts in 2 of the 8 cells: none - verifier = skeptical - verifier = agg - majority = 1/4.
    for h in ("H5", "H6", "H7"):
        assert df.loc[h, "mean_difference"] == 0.25 and df.loc[h, "supported"], h
    # The two honest solvers always agree on the gold answer, so the hybrid vote is always
    # right; the aggregator is wrong exactly when it adopts the error.
    assert df.loc["H8", "mean_difference"] == 0.25 and df.loc["H8", "supported"]
    assert (df["p_holm"] >= df["p_value"]).all()


def test_two_sided_hypotheses_detect_reversed_effects():
    # Star adopts more than chain here: a one-sided H4 would miss it, the two-sided test does not.
    recs = []
    for r in synthetic():
        if r["config_id"].startswith(("chain-", "star-")):
            adopt = r["config_id"].startswith("star")
            r = {**r, "messages": r["messages"][:-1] + [{**r["messages"][-1], "parsed_answer": "99" if adopt else "5"}]}
        recs.append(r)
    df = run(recs, n_perm=2000).set_index("hypothesis")
    assert df.loc["H4", "alternative"] == "two-sided"
    assert df.loc["H4", "mean_difference"] == -1.0 and df.loc["H4", "supported"]


def test_majority_adoption_counts_for_h7():
    recs = [record(r["config_id"], r["question_id"], adopted=False, shown_at=r["fault"]["shown_at"],
                   majority_adopts=True) for r in synthetic()]
    df = run(recs, n_perm=2000).set_index("hypothesis")
    assert df.loc["H7", "mean_difference"] == -1.0 and not df.loc["H7", "supported"]  # one-sided


def test_hybrid_falls_back_to_the_aggregator_without_majority():
    # Three different solver answers: the hybrid vote is the aggregator's answer, so H8 is 0.
    recs = []
    for r in synthetic():
        msgs = [{**m, "parsed_answer": a} if m["role"] == "solver" else m
                for m, a in zip(r["messages"], ("99", "7", "5", None))]
        recs.append({**r, "messages": msgs})
    df = run(recs, n_perm=2000).set_index("hypothesis")
    assert df.loc["H8", "mean_difference"] == 0.0 and not df.loc["H8", "supported"]
