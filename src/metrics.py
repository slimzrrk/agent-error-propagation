"""Metrics over run records, bootstrap confidence intervals and McNemar test."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from src.parser import answers_match, last_number


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI of a 0/1 or numeric array."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_ratio_ci(num, den, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """sum(num) / sum(den) with a CI from resampling questions (clusters), not agents:
    agents answering the same question are not independent."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    ratios = ratios[~np.isnan(ratios)]
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def mcnemar(a, b) -> dict:
    """Exact McNemar test on paired 0/1 outcomes (same questions, two configs)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else binomtest(only_a, n, 0.5).pvalue
    return {"only_a": only_a, "only_b": only_b, "p_value": float(p)}


def invalid_rate(records: list[dict], role: str | None = None) -> float:
    """Share of LLM messages (optionally of one role) with no parsable final answer."""
    msgs = [
        m for r in records for m in r["messages"]
        if not m["is_faulty"] and (role is None or m["role"] == role)
    ]
    if not msgs:
        return float("nan")
    return sum(m["parsed_answer"] is None for m in msgs) / len(msgs)


def downstream(messages: list[dict], source: str) -> set[str]:
    """Agents that saw `source`'s message, directly or through another agent.
    Messages are logged in execution order, so one pass is enough."""
    exposed, out = {source}, set()
    for m in messages:
        if exposed & set(m.get("sees", [])):
            exposed.add(m["agent_id"])
            out.add(m["agent_id"])
    return out


def upstream(messages: list[dict], source: str) -> set[str]:
    """Agents whose messages `source` saw, directly or through another agent."""
    sees = {m["agent_id"]: set(m.get("sees", [])) for m in messages}
    out, todo = set(), list(sees.get(source, ()))
    while todo:
        a = todo.pop()
        if a not in out:
            out.add(a)
            todo.extend(sees.get(a, ()))
    return out


def relation(messages: list[dict], source: str, agent_id: str) -> str:
    if agent_id in downstream(messages, source):
        return "downstream"
    if agent_id in upstream(messages, source):
        return "upstream"
    return "independent"


def answer_of(m: dict, lenient: bool = False) -> str | None:
    """Parsed answer of a message. Lenient (sensitivity analysis only): an output without
    a valid "Final answer" line falls back to the last number in its text."""
    if m["parsed_answer"] is None and lenient:
        return last_number(m["content"])
    return m["parsed_answer"]


def final_message(r: dict) -> dict:
    """The message that holds the team answer: the aggregator, or the single solver."""
    return next((m for m in r["messages"] if m["role"] == "aggregator"), r["messages"][0])


def is_correct(r: dict, lenient: bool = False) -> bool:
    return answers_match(answer_of(final_message(r), lenient), r["gold"])


def adoption(records: list[dict], source: str = "solver1", lenient: bool = False) -> dict:
    """How often other agents end with the same wrong answer as `source`.

    Only runs where `source` gives a valid wrong answer count (for an injected fault this
    is every run). Adoption is measured on downstream agents (they saw the source, directly
    or not). Independent agents (neither downstream nor upstream) give the control: how
    often the same wrong answer appears by coincidence. Upstream agents are excluded from
    both, since the source may have copied them.
    """
    per_agent = defaultdict(lambda: {"n": 0, "same": 0, "gold": 0})
    pooled = {k: ([], []) for k in ("downstream", "independent")}
    for r in records:
        msgs = r["messages"]
        src = next((m for m in msgs if m["agent_id"] == source), None)
        wrong = answer_of(src, lenient) if src else None
        if wrong is None or answers_match(wrong, r["gold"]):
            continue
        counts = {k: [0, 0] for k in pooled}
        for m in msgs:
            if m["agent_id"] == source:
                continue
            rel = relation(msgs, source, m["agent_id"])
            ans = answer_of(m, lenient)
            same = answers_match(ans, wrong)
            a = per_agent[(m["agent_id"], rel)]
            a["n"] += 1
            a["same"] += same
            a["gold"] += answers_match(ans, r["gold"])
            if rel in counts:
                counts[rel][0] += same
                counts[rel][1] += 1
        for k, (num, den) in pooled.items():
            num.append(counts[k][0])
            den.append(counts[k][1])

    agents = [
        {"agent_id": aid, "relation": rel, "n": v["n"], "adopted": v["same"] / v["n"], "correct": v["gold"] / v["n"]}
        for (aid, rel), v in sorted(per_agent.items())
    ]
    rate, lo, hi = bootstrap_ratio_ci(*pooled["downstream"])
    ctrl, clo, chi = bootstrap_ratio_ci(*pooled["independent"])
    return {
        "source": source,
        "lenient": lenient,
        "n_cases": len(pooled["downstream"][0]),
        "adoption_rate": rate, "adoption_ci": (lo, hi),
        "control_rate": ctrl, "control_ci": (clo, chi),
        "agents": agents,
    }


def team_adopted(r: dict, lenient: bool = False) -> bool:
    """The team's final answer is the injected wrong answer."""
    return bool(r.get("fault")) and answers_match(answer_of(final_message(r), lenient), r["fault"]["wrong_answer"])


def fault_summary(records: list[dict], lenient: bool = False) -> dict:
    """Effect of an injected fault on one config: recovery (final answer still correct),
    final adoption (team answer = injected answer) and adoption by the downstream agents."""
    source = records[0]["fault"]["agent_id"]
    ad = adoption(records, source=source, lenient=lenient)
    return {
        "source": source,
        "n": len(records),
        "recovery": bootstrap_ci([is_correct(r, lenient) for r in records]),
        "team_adopted": bootstrap_ci([team_adopted(r, lenient) for r in records]),
        "downstream_adoption": (ad["adoption_rate"], *ad["adoption_ci"]),
        "coincidence": (ad["control_rate"], *ad["control_ci"]),
        "agents": ad["agents"],
    }


def breakdown(records: list[dict], key, lenient: bool = False) -> list[dict]:
    """Fault effects within groups of runs, e.g. by presentation slot or slip position."""
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    rows = []
    for value, rs in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        ad = adoption(rs, source=rs[0]["fault"]["agent_id"], lenient=lenient)
        rows.append({
            "group": value, "n": len(rs),
            "team_adopted": bootstrap_ci([team_adopted(r, lenient) for r in rs]),
            "recovery": bootstrap_ci([is_correct(r, lenient) for r in rs]),
            "downstream_adoption": (ad["adoption_rate"], *ad["adoption_ci"]),
        })
    return rows


def by_slot(records: list[dict], lenient: bool = False) -> list[dict]:
    """By the slot where the faulty solution was shown to the aggregator."""
    return breakdown(records, lambda r: r["fault"]["shown_at"], lenient)


def by_obvious(records: list[dict], lenient: bool = False) -> list[dict]:
    """By whether the fault had to be kept although it breaks the plausibility rule."""
    return breakdown(records, lambda r: bool(r["fault"].get("obvious")), lenient)


def by_slip_position(records: list[dict], lenient: bool = False) -> list[dict]:
    """By whether the slip is early or late in the faulty solution."""
    return breakdown(records, lambda r: (r["fault"].get("slip_position") or {}).get("category"), lenient)


def paired_outcomes(a: list[dict], b: list[dict], outcome) -> dict:
    """McNemar test of a per-question 0/1 outcome between two configs, on shared questions."""
    rb = {r["question_id"]: r for r in b}
    pairs = [(outcome(r), outcome(rb[r["question_id"]])) for r in a if r["question_id"] in rb]
    if not pairs:
        return {"n": 0}
    xa, xb = zip(*pairs)
    return {"n": len(pairs), "rate_a": float(np.mean(xa)), "rate_b": float(np.mean(xb)), **mcnemar(xa, xb)}


def majority_vote(answers: list[str | None]) -> str | None:
    """Strict majority among valid answers; None if there is none (e.g. three different answers)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    (answer, count), = Counter(valid).most_common(1)
    return answer if count > len(answers) / 2 else None


def majority_answer(r: dict, lenient: bool = False) -> str | None:
    """Counterfactual team answer: strict majority of the solvers' answers (faulty one
    included), no LLM call. Three different answers give no majority (None, i.e. wrong)."""
    return majority_vote([answer_of(m, lenient) for m in r["messages"] if m["role"] == "solver"])


def majority_vs_aggregator(records: list[dict], lenient: bool = False) -> dict:
    """Accuracy (and, with a fault, adoption of the injected answer) of the LLM aggregator
    and of a majority vote over the same solver answers, with paired McNemar tests."""
    agg_ok = [is_correct(r, lenient) for r in records]
    maj_ok = [answers_match(majority_answer(r, lenient), r["gold"]) for r in records]
    out = {"n": len(records), "aggregator_acc": bootstrap_ci(agg_ok), "majority_acc": bootstrap_ci(maj_ok),
           "acc_test": mcnemar(agg_ok, maj_ok)}
    if records[0].get("fault"):
        agg_ad = [team_adopted(r, lenient) for r in records]
        maj_ad = [answers_match(majority_answer(r, lenient), r["fault"]["wrong_answer"]) for r in records]
        out.update(aggregator_adopted=bootstrap_ci(agg_ad), majority_adopted=bootstrap_ci(maj_ad),
                   adopt_test=mcnemar(agg_ad, maj_ad))
    return out


def paired_permutation_test(diffs, alternative: str = "greater", n_perm: int = 100_000, seed: int = 0) -> float:
    """Sign-flip permutation test on per-question paired differences (H0: mean 0).

    Under H0 each question's difference is equally likely to have either sign.
    """
    d = np.asarray(diffs, dtype=float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_perm:
        k = min(10_000, n_perm - done)
        null = (rng.choice([-1.0, 1.0], size=(k, d.size)) * d).mean(axis=1)
        if alternative == "greater":
            hits += int(np.sum(null >= obs - 1e-12))
        elif alternative == "less":
            hits += int(np.sum(null <= obs + 1e-12))
        else:
            hits += int(np.sum(np.abs(null) >= abs(obs) - 1e-12))
        done += k
    return (hits + 1) / (n_perm + 1)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, adjusted = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def solver_references(records: list[dict]) -> dict:
    """Non-LLM references for the aggregator: majority vote and oracle (any solver right)."""
    maj, oracle = [], []
    for r in records:
        solvers = [m for m in r["messages"] if m["role"] == "solver"]
        maj.append(answers_match(majority_vote([m["parsed_answer"] for m in solvers]), r["gold"]))
        oracle.append(any(answers_match(m["parsed_answer"], r["gold"]) for m in solvers))
    return {"majority_acc": bootstrap_ci(maj), "oracle_acc": bootstrap_ci(oracle)}


def per_agent(records: list[dict]) -> list[dict]:
    """Accuracy and invalid rate of each agent position (solver1, ..., aggregator)."""
    groups = defaultdict(list)
    for r in records:
        for m in r["messages"]:
            groups[m["agent_id"]].append((m, r["gold"]))
    rows = []
    for aid, items in groups.items():
        rows.append({
            "agent_id": aid,
            "n": len(items),
            "accuracy": float(np.mean([answers_match(m["parsed_answer"], g) for m, g in items])),
            "invalid_rate": float(np.mean([m["parsed_answer"] is None for m, _ in items])),
            "prompt_tokens": float(np.mean([m["prompt_tokens"] for m, _ in items])),
            "completion_tokens": float(np.mean([m["completion_tokens"] for m, _ in items])),
        })
    order = {"aggregator": 99}
    return sorted(rows, key=lambda x: (order.get(x["agent_id"], 0), x["agent_id"]))


def summarize(records: list[dict]) -> dict:
    """Headline numbers for one config (strict protocol, plus lenient accuracy)."""
    acc, lo, hi = bootstrap_ci([r["correct"] for r in records])
    msgs = [m for r in records for m in r["messages"] if not m["is_faulty"]]
    return {
        "n": len(records),
        "accuracy": acc,
        "acc_ci_low": lo,
        "acc_ci_high": hi,
        "accuracy_lenient": float(np.mean([is_correct(r, lenient=True) for r in records])),
        "invalid_rate": invalid_rate(records),
        "truncated_rate": (sum(m["finish_reason"] == "length" for m in msgs) / len(msgs)) if msgs else float("nan"),
        "calls_per_q": float(np.mean([r["n_llm_calls"] for r in records])),
        "prompt_tokens_per_q": float(np.mean([r["prompt_tokens"] for r in records])),
        "completion_tokens_per_q": float(np.mean([r["completion_tokens"] for r in records])),
        "latency_per_call": float(np.mean([m["latency"] for m in msgs])) if msgs else float("nan"),
    }
