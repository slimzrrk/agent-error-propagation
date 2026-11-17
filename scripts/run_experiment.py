"""Run every experiment of a config file on a split, in parallel, with resumable logging.

    python -m scripts.run_experiment --config configs/baseline.yaml --split dev --limit 20
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src import prompts
from src.config import RunConfig
from src.data import load_questions
from src.llm import LLMClient
from src.logging_utils import JsonlLog, environment_metadata, read_records
from src.metrics import summarize
from src.topologies import run_question

log = logging.getLogger("run_experiment")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--limit", type=int, default=None, help="first N questions of the split")
    p.add_argument("--workers", type=int, default=None, help="overrides the config value")
    p.add_argument("--out", default=None, help="default: results/<split>/<config name>.jsonl")
    return p.parse_args(argv)


def print_summary(records: list[dict], config_order: list[str]) -> None:
    by_cfg = defaultdict(list)
    for r in records:
        by_cfg[r["config_id"]].append(r)
    header = f"{'config':<36} {'n':>4} {'acc':>6} {'95% CI':>15} {'invalid':>8} {'trunc':>6} {'calls':>6} {'tok/q':>7} {'s/call':>7}"
    print("\n" + header + "\n" + "-" * len(header))
    for cid in config_order:
        if cid not in by_cfg:
            continue
        s = summarize(by_cfg[cid])
        print(
            f"{cid:<36} {s['n']:>4} {s['accuracy']:>6.1%} [{s['acc_ci_low']:>5.1%}, {s['acc_ci_high']:>5.1%}]"
            f" {s['invalid_rate']:>8.1%} {s['truncated_rate']:>6.1%} {s['calls_per_q']:>6.1f}"
            f" {s['prompt_tokens_per_q'] + s['completion_tokens_per_q']:>7.0f} {s['latency_per_call']:>7.2f}"
        )


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args(argv)

    cfg = RunConfig.from_yaml(args.config)
    workers = args.workers or cfg.workers
    out = Path(args.out or f"results/{args.split}/{cfg.name}.jsonl")

    questions = load_questions(args.split, limit=args.limit)
    llm = LLMClient(cfg.llm)
    jl = JsonlLog(out)

    previous = {r.get("prompts_sha") for r in read_records(out, "meta")} - {None}
    if previous and previous != {prompts.fingerprint()}:
        raise SystemExit(f"{out} was produced with different prompts; use another --out")

    done = jl.done_keys()
    tasks = [(exp, q) for exp in cfg.experiments for q in questions if (exp.id, q.id) not in done]
    print(f"{len(cfg.experiments)} experiments x {len(questions)} questions; "
          f"{len(tasks)} to run, {len(done)} already in {out}")

    if tasks:
        jl.write_meta({
            "config_name": cfg.name,
            "config_file": str(args.config),
            "split": args.split,
            "limit": args.limit,
            "workers": workers,
            "base_seed": cfg.seed,
            "prompts_sha": prompts.fingerprint(),
            "llm": cfg.llm.to_dict(),
            "server": llm.server_info(),
            "experiments": [{"id": e.id, **e.to_dict()} for e in cfg.experiments],
            **environment_metadata(),
        })

    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_question, llm, q, exp, cfg.seed): (exp, q) for exp, q in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="runs"):
            exp, q = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # keep going; the pair is retried on the next run
                failures += 1
                log.error("%s / %s failed: %r", exp.id, q.id, exc)
                continue
            jl.write_run({
                "config_id": exp.id,
                "config": exp.to_dict(),
                "split": args.split,
                "model": cfg.llm.model,
                "base_seed": cfg.seed,
                **result,
            })

    if failures:
        print(f"{failures} runs failed, rerun the same command to retry them")
    qids = {q.id for q in questions}
    records = [r for r in read_records(out) if r["question_id"] in qids]
    print_summary(records, [e.id for e in cfg.experiments])


if __name__ == "__main__":
    main()
