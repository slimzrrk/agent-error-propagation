"""Metrics over run records, bootstrap confidence intervals and McNemar test."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI of a 0/1 or numeric array."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


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
