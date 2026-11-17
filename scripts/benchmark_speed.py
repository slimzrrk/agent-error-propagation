"""Measure generation throughput at several concurrency levels and estimate the grid cost.

    python -m scripts.benchmark_speed --config configs/baseline.yaml --levels 1 4 8

Every level runs the same solver requests (same questions and seeds), so the levels
differ only by concurrency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src import prompts
from src.config import RunConfig, full_grid
from src.data import load_questions
from src.llm import LLMClient, derive_seed
from src.logging_utils import environment_metadata, now
from src.topologies import expected_llm_calls

SPLIT_SIZES = {"dev": 50, "test": 200}


def one_request(llm: LLMClient, q, base_seed: int):
    return llm.chat(
        [{"role": "system", "content": prompts.SOLVER_SYSTEM},
         {"role": "user", "content": prompts.solver_user(q.question)}],
        seed=derive_seed(q.id, "bench", base=base_seed),
    )


def run_level(llm, questions, concurrency: int, base_seed: int) -> dict:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outs = list(pool.map(lambda q: one_request(llm, q, base_seed), questions))
    wall = time.perf_counter() - start
    completion = sum(o.completion_tokens for o in outs)
    lat = [o.latency for o in outs]
    return {
        "concurrency": concurrency,
        "requests": len(outs),
        "wall_s": round(wall, 2),
        "completion_tokens": completion,
        "prompt_tokens": sum(o.prompt_tokens for o in outs),
        "throughput_tok_s": round(completion / wall, 1),
        "requests_per_s": round(len(outs) / wall, 3),
        "latency_mean_s": round(statistics.mean(lat), 2),
        "latency_p90_s": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2),
        "per_stream_tok_s": round(statistics.mean(o.completion_tokens / o.latency for o in outs), 1),
    }


def grid_estimate(sec_per_call: float) -> dict:
    grid = full_grid(include_single=True)
    calls_per_q = sum(expected_llm_calls(e) for e in grid)
    est = {"n_configs": len(grid), "llm_calls_per_question": calls_per_q}
    for split, n in SPLIT_SIZES.items():
        calls = calls_per_q * n
        est[split] = {"questions": n, "llm_calls": calls, "hours": round(calls * sec_per_call / 3600, 2)}
    return est


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--requests", type=int, default=24, help="requests per level")
    p.add_argument("--out", default="results/benchmark_speed.json")
    args = p.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config)
    llm = LLMClient(cfg.llm)
    questions = load_questions("dev")[: args.requests]

    print(f"model {cfg.llm.model} @ {cfg.llm.base_url}, {len(questions)} requests per level")
    one_request(llm, questions[0], cfg.seed + 1)  # warm-up: load the model

    rows = []
    for c in args.levels:
        row = run_level(llm, questions, c, cfg.seed)
        rows.append(row)
        print(f"concurrency {c:>2}: {row['throughput_tok_s']:>7.1f} tok/s total, "
              f"{row['per_stream_tok_s']:>5.1f} tok/s per stream, {row['requests_per_s']:.2f} req/s, "
              f"latency {row['latency_mean_s']:.1f}s (p90 {row['latency_p90_s']:.1f}s)")

    # Experiment runs keep `max(levels)` requests in flight, so use that level's rate.
    best = rows[-1]
    sec_per_call = best["wall_s"] / best["requests"]
    est = grid_estimate(sec_per_call)
    print(f"\nfull grid: {est['n_configs']} configs, {est['llm_calls_per_question']} LLM calls per question")
    for split in SPLIT_SIZES:
        e = est[split]
        print(f"  {split:>4} ({e['questions']} q): {e['llm_calls']:>6} calls ~ {e['hours']:.1f} h "
              f"at {sec_per_call:.2f} s/call (solver-sized calls; longer aggregator prompts add some)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": now(),
        "llm": cfg.llm.to_dict(),
        "server": llm.server_info(),
        "levels": rows,
        "sec_per_call_at_max_level": round(sec_per_call, 3),
        "grid_estimate": est,
        **environment_metadata(),
    }, indent=2) + "\n")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
