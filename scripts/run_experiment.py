"""Run every experiment of a config file on a split, in parallel, with resumable logging.

    python -m scripts.run_experiment --config configs/baseline.yaml --split dev --limit 20
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from scripts.summarize import print_main_table
from src import prompts
from src.config import RunConfig
from src.data import load_questions
from src.faults import faults_sha
from src.llm import LLMClient
from src.logging_utils import JsonlLog, environment_metadata, protocol_fingerprint, read_records
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

    protocol = protocol_fingerprint(faults_sha())
    previous = {r.get("protocol_sha") for r in read_records(out, "meta")}
    if previous and previous != {protocol}:
        raise SystemExit(f"{out} was produced with a different protocol (prompts, code or faults); "
                         "use another --out")

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
            "protocol_sha": protocol,
            "prompts_sha": prompts.fingerprint(),
            "faults_sha": faults_sha(),
            "llm": cfg.llm.to_dict(),
            "server": llm.server_info(),
            "experiments": [{"id": e.id, **e.to_dict()} for e in cfg.experiments],
            **environment_metadata(),
        })

        failures = 0
        start = time.perf_counter()
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
        wall = time.perf_counter() - start
        # Wall time of the session, used to turn per-call latencies into cost estimates.
        jl.write({"record_type": "session_end", "session_id": jl.session_id,
                  "wall_s": round(wall, 1), "runs": len(tasks) - failures, "failures": failures, "workers": workers})
        print(f"{len(tasks) - failures} runs in {wall / 60:.1f} min")
        if failures:
            print(f"{failures} runs failed, rerun the same command to retry them")

    qids = {q.id for q in questions}
    records = [r for r in read_records(out) if r["question_id"] in qids]
    print_main_table(records, [e.id for e in cfg.experiments])


if __name__ == "__main__":
    main()
