"""Metrics over run records, bootstrap confidence intervals and McNemar test."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from src.parser import answers_match


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


def adoption(records: list[dict], source: str = "solver1") -> dict:
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
        if src is None or src["parsed_answer"] is None or answers_match(src["parsed_answer"], r["gold"]):
            continue
        wrong = src["parsed_answer"]
        counts = {k: [0, 0] for k in pooled}
        for m in msgs:
            if m["agent_id"] == source:
                continue
            rel = relation(msgs, source, m["agent_id"])
            same = answers_match(m["parsed_answer"], wrong)
            a = per_agent[(m["agent_id"], rel)]
            a["n"] += 1
            a["same"] += same
            a["gold"] += answers_match(m["parsed_answer"], r["gold"])
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
        "n_cases": len(pooled["downstream"][0]),
        "adoption_rate": rate, "adoption_ci": (lo, hi),
        "control_rate": ctrl, "control_ci": (clo, chi),
        "agents": agents,
    }


def majority_vote(answers: list[str | None]) -> str | None:
    """Strict majority among valid answers; None if there is none (e.g. three different answers)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    # Group equal numbers ("18" and "18.0" are already canonicalised by the parser).
    (answer, count), = Counter(valid).most_common(1)
    return answer if count > len(answers) / 2 else None


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
    """Headline numbers for one config."""
    correct = [r["correct"] for r in records]
    acc, lo, hi = bootstrap_ci(correct)
    msgs = [m for r in records for m in r["messages"] if not m["is_faulty"]]
    return {
        "n": len(records),
        "accuracy": acc,
        "acc_ci_low": lo,
        "acc_ci_high": hi,
        "invalid_rate": invalid_rate(records),
        "truncated_rate": (sum(m["finish_reason"] == "length" for m in msgs) / len(msgs)) if msgs else float("nan"),
        "calls_per_q": float(np.mean([r["n_llm_calls"] for r in records])),
        "prompt_tokens_per_q": float(np.mean([r["prompt_tokens"] for r in records])),
        "completion_tokens_per_q": float(np.mean([r["completion_tokens"] for r in records])),
        "latency_per_call": float(np.mean([m["latency"] for m in msgs])) if msgs else float("nan"),
    }
