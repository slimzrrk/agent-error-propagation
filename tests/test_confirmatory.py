from scripts.confirmatory import run

STYLES = ("neutral", "persuasive")


def record(config_id, qid, adopted, shown_at):
    final = "99" if adopted else "5"
    return {
        "record_type": "run", "config_id": config_id, "question_id": qid, "gold": "5",
        "fault": {"wrong_answer": "99", "shown_at": shown_at},
        "messages": [{"agent_id": "aggregator", "role": "aggregator", "parsed_answer": final,
                      "content": f"Final answer: {final}", "is_faulty": False, "sees": []}],
    }


def synthetic(n=40):
    """Adoption only when the faulty solution is shown first; chain shows p1 first."""
    recs = []
    for i in range(n):
        q = f"q{i}"
        for s in STYLES:
            for p in (1, 2, 3):
                recs.append(record(f"chain-{s}-p{p}-none", q, adopted=p == 1, shown_at=p))
                rank = (i + p) % 3 + 1  # Latin square
                recs.append(record(f"chain_rotated-{s}-p{p}-none", q, adopted=rank == 1, shown_at=rank))
            recs.append(record(f"star-{s}-p1-none", q, adopted=False, shown_at=i % 3 + 1))
    return recs


def test_confirmatory_on_synthetic_data():
    df = run(synthetic(), n_perm=2000).set_index("hypothesis")
    assert df.loc["H1", "mean_difference"] == 1.0 and df.loc["H1", "supported"]
    assert df.loc["H3", "mean_difference"] == 1.0 and df.loc["H3", "supported"]
    assert df.loc["H2", "mean_difference"] == 0.0 and not df.loc["H2", "supported"]
    assert abs(df.loc["H4", "mean_difference"] - 1 / 3) < 1e-9 and df.loc["H4", "supported"]
    assert (df["p_holm"] >= df["p_value"]).all()


def test_two_sided_hypotheses_detect_reversed_effects():
    # Star adopts more than chain here: a one-sided H4 would miss it, the two-sided test does not.
    recs = [{**r, "messages": [{**r["messages"][0], "parsed_answer": "99" if r["config_id"].startswith("star") else "5"}]}
            if not r["config_id"].startswith("chain_rotated") else r for r in synthetic()]
    df = run(recs, n_perm=2000).set_index("hypothesis")
    assert df.loc["H4", "alternative"] == "two-sided"
    assert df.loc["H4", "mean_difference"] == -1.0 and df.loc["H4", "supported"]
