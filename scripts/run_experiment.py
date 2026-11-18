"""Run every experiment of a config file on a split, in parallel, with resumable logging.

    python -m scripts.run_experiment --config configs/baseline.yaml --split dev --limit 20

On dev, --reuse OLD_LOG copies the runs of an older log whose inputs have not changed
(same config, question and faulty message; same model, sampling settings and seed), so
that only the affected runs are redone. Sampling is seeded per (question, agent), so a
rerun would give the same messages; this is checked by rerunning a random sample of the
copied runs, and nothing is reused if any of them differs.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from scripts.summarize import print_main_table
from src import prompts
from src.config import RunConfig
from src.data import load_questions
from src.faults import fault_text, faults_sha
from src.llm import LLMClient
from src.logging_utils import JsonlLog, environment_metadata, protocol_fingerprint, read_records
from src.topologies import fault_record, run_question

log = logging.getLogger("run_experiment")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--limit", type=int, default=None, help="first N questions of the split")
    p.add_argument("--workers", type=int, default=None, help="overrides the config value")
    p.add_argument("--out", default=None, help="default: results/<split>/<config name>.jsonl")
    p.add_argument("--reuse", default=None, help="dev only: older log whose unchanged runs are copied")
    p.add_argument("--reuse-check", type=int, default=24, help="copied runs that are rerun as a check")
    return p.parse_args(argv)


def run_tasks(llm: LLMClient, tasks: list, base_seed: int, workers: int, desc: str = "runs"):
    """Yield (exp, question, result or exception) as runs complete."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_question, llm, q, exp, base_seed): (exp, q) for exp, q in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            exp, q = futures[fut]
            try:
                yield exp, q, fut.result()
            except Exception as exc:  # keep going; the pair is retried on the next run
                yield exp, q, exc


def select_reusable(old_runs: list[dict], tasks: list, fault_text_of=fault_text) -> dict:
    """{(config_id, question_id): old run} for the tasks whose inputs are unchanged: same
    config and question and, with a fault, the same faulty message and injected answer."""
    by_key = {(r["config_id"], r["question_id"]): r for r in old_runs}
    out = {}
    for exp, q in tasks:
        r = by_key.get((exp.id, q.id))
        if r is None:
            continue
        if exp.fault != "none":
            faulty = next((m for m in r["messages"] if m["is_faulty"]), None)
            text, wrong = fault_text_of(q.id, exp.fault)
            if faulty is None or faulty["content"] != text or r["fault"]["wrong_answer"] != wrong:
                continue
        out[(exp.id, q.id)] = r
    return out


def same_outputs(a: dict, b: dict) -> bool:
    texts = lambda r: [m["content"] for m in r["messages"]] + [v["content"] for v in r.get("verdicts", [])]  # noqa: E731
    return texts(a) == texts(b) and a["final_answer"] == b["final_answer"]


def check_old_log(path: str, cfg: RunConfig, split: str, server: dict) -> list[str]:
    metas = read_records(path, "meta")
    if not metas:
        raise SystemExit(f"{path}: no runs to reuse")
    for m in metas:
        same = (m["split"] == split and m["base_seed"] == cfg.seed and m["llm"] == cfg.llm.to_dict()
                and all(m["server"].get(k) == server.get(k) for k in ("model_digest", "ollama_version")))
        if not same:
            raise SystemExit(f"{path} used another split, seed, sampling setting or model build; cannot reuse it")
    return sorted({m["session_id"] for m in metas})


def prepare_reuse(args, cfg: RunConfig, llm: LLMClient, tasks: list, workers: int):
    """(meta info, rerun check results, copied records)."""
    if args.split == "test":
        raise SystemExit("--reuse is for dev only: the test split is run once, from scratch")
    sessions = check_old_log(args.reuse, cfg, args.split, llm.server_info())
    reusable = select_reusable(read_records(args.reuse), tasks)
    sample = set(random.Random(0).sample(sorted(reusable), min(args.reuse_check, len(reusable))))
    to_check = [(exp, q) for exp, q in tasks if (exp.id, q.id) in sample]

    fresh, mismatches = [], []
    for exp, q, result in run_tasks(llm, to_check, cfg.seed, workers, desc="reuse check"):
        if isinstance(result, Exception):
            raise SystemExit(f"reuse check failed on {exp.id} / {q.id}: {result!r}")
        fresh.append((exp, result))
        if not same_outputs(result, reusable[(exp.id, q.id)]):
            mismatches.append(f"{exp.id} / {q.id}")
    if mismatches:
        raise SystemExit(f"rerun differs from {args.reuse} for {mismatches}; run without --reuse")

    questions = {q.id: q for _, q in tasks}
    exps = {exp.id: exp for exp, _ in tasks}
    copies = []
    for (cid, qid), old in reusable.items():
        if (cid, qid) in sample:
            continue
        rec = {k: v for k, v in old.items() if k not in ("record_type", "session_id", "timestamp", "reused_from")}
        if exps[cid].fault != "none":
            rec["fault"] = fault_record(questions[qid].id, exps[cid], old["fault"]["shown_at"])
        rec["reused_from"] = {"log": str(args.reuse), "session_id": old["session_id"]}
        copies.append(rec)
    info = {"from": str(args.reuse), "sessions": sessions, "copied": len(copies),
            "check": {"rerun": len(fresh), "identical": len(fresh) - len(mismatches)}}
    print(f"reuse: {len(copies)} runs copied from {args.reuse}; "
          f"{len(fresh)} others rerun as a check, all identical")
    return info, fresh, copies


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
    reuse_info, fresh, copies = None, [], []
    if args.reuse and tasks:
        reuse_info, fresh, copies = prepare_reuse(args, cfg, llm, tasks, workers)
        handled = {(r["config_id"], r["question_id"]) for r in copies} | {(e.id, r["question_id"]) for e, r in fresh}
        tasks = [(exp, q) for exp, q in tasks if (exp.id, q.id) not in handled]
    print(f"{len(cfg.experiments)} experiments x {len(questions)} questions; "
          f"{len(tasks)} to run, {len(done)} already in {out}")

    def run_record(exp, result):
        return {"config_id": exp.id, "config": exp.to_dict(), "split": args.split,
                "model": cfg.llm.model, "base_seed": cfg.seed, **result}

    if tasks or fresh or copies:
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
            "reused": reuse_info,
            **environment_metadata(),
        })
        for exp, result in fresh:  # outside the timed loop below, hence flagged
            jl.write_run({**run_record(exp, result), "reuse_check": True})
        for rec in copies:
            jl.write_run(rec)

        failures = 0
        start = time.perf_counter()
        for exp, q, result in run_tasks(llm, tasks, cfg.seed, workers):
            if isinstance(result, Exception):
                failures += 1
                log.error("%s / %s failed: %r", exp.id, q.id, result)
                continue
            jl.write_run(run_record(exp, result))
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
