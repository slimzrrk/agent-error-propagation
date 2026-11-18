"""Cheap dev-only checks of the defense prompts, on messages already logged in phase 3.

    python -m scripts.tune_defenses verifier --label it1
    python -m scripts.tune_defenses skeptical --label it1

verifier: the verifier checks the honest star solutions (no fault) and the injected
faulty solutions (neutral and persuasive), with the seeds of the real runs.
skeptical: the aggregator, with the skeptical instruction, reruns on the solutions it saw
in phase 3 (same order, same seed). For the star this is exactly the star-skeptical run;
for the chain it leaves out the effect of the instruction on solvers 2 and 3.

Selection rules, fixed before the first iteration (report/defense_tuning.md): at most three
iterations per defense; the verifier maximises recall - false positive rate (Youden's J)
with at most 5% unparsed verdicts; the skeptical instruction minimises team adoption over
the fault configs, as long as accuracy without fault stays within 4 points of no defense.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import prompts
from src.agents import Message, run_aggregator
from src.config import RunConfig
from src.data import load_questions
from src.defenses import verify
from src.llm import LLMClient
from src.logging_utils import JsonlLog, read_records
from src.metrics import answers_match, verifier_stats

LOG = "results/dev/phase3_propagation.jsonl"
SUMMARY = Path("results/dev/defense_tuning.json")
SKEPTICAL_CONFIGS = ("star-none-none", "chain-none-none", "star-neutral-p1-none", "star-persuasive-p1-none",
                     "chain-neutral-p1-none", "chain-persuasive-p1-none")


def as_message(m: dict) -> Message:
    return Message(**m)


def tune_verifier(llm, records, questions, base_seed, workers):
    jobs = []
    for r in records:
        if r["config_id"] == "star-none-none":
            jobs += [(r, m) for m in r["messages"] if m["role"] == "solver"]
        elif r["config_id"] in ("star-neutral-p1-none", "star-persuasive-p1-none"):
            jobs += [(r, m) for m in r["messages"] if m["is_faulty"]]

    def run(job):
        r, m = job
        return r, verify(llm, questions[r["question_id"]], as_message(m), base_seed=base_seed)

    with ThreadPoolExecutor(workers) as pool:
        out = list(tqdm(pool.map(run, jobs), total=len(jobs), desc="verifier"))
    # One pseudo-record per checked message, so verifier_stats can score it.
    recs = [{"question_id": r["question_id"], "gold": r["gold"], "config_id": r["config_id"],
             "messages": [m for m in r["messages"] if m["agent_id"] == v["target"]]
             + [{"role": "aggregator", "agent_id": "aggregator", "sees": ["x"]}],
             "verdicts": [v]} for r, v in out]
    stats = verifier_stats(recs)
    honest = verifier_stats([x for x in recs if x["config_id"] == "star-none-none"])
    stats["youden_j"] = stats["recall"][0] - stats["false_positive_rate"][0]
    stats["honest_only"] = {k: honest[k] for k in ("recall", "precision", "false_positive_rate")}
    stats["completion_tokens"] = float(np.mean([v["completion_tokens"] for _, v in out]))
    return stats, recs


def tune_skeptical(llm, records, questions, base_seed, workers):
    jobs = [r for r in records if r["config_id"] in SKEPTICAL_CONFIGS]

    def run(r):
        msgs = {m["agent_id"]: m for m in r["messages"]}
        agg = next(m for m in r["messages"] if m["role"] == "aggregator")
        shown = [as_message(msgs[a]) for a in agg["sees"]]
        new = run_aggregator(llm, questions[r["question_id"]], shown, base_seed=base_seed, skeptical=True)
        return {**r, "messages": [m for m in r["messages"] if m["role"] != "aggregator"] + [new.to_dict()]}

    with ThreadPoolExecutor(workers) as pool:
        out = list(tqdm(pool.map(run, jobs), total=len(jobs), desc="skeptical aggregator"))
    stats = {}
    for cid in SKEPTICAL_CONFIGS:
        recs = [r for r in out if r["config_id"] == cid]
        old = [r for r in jobs if r["config_id"] == cid]
        agg = [next(m for m in r["messages"] if m["role"] == "aggregator") for r in recs]
        s = {"n": len(recs),
             "accuracy": float(np.mean([answers_match(a["parsed_answer"], r["gold"]) for a, r in zip(agg, recs)])),
             "accuracy_no_defense": float(np.mean([r["correct"] for r in old])),
             "invalid": float(np.mean([a["parsed_answer"] is None for a in agg])),
             "completion_tokens": float(np.mean([a["completion_tokens"] for a in agg]))}
        if recs[0].get("fault"):
            wrong = lambda r, a: answers_match(a["parsed_answer"], r["fault"]["wrong_answer"])  # noqa: E731
            s["team_adopted"] = float(np.mean([wrong(r, a) for a, r in zip(agg, recs)]))
            s["team_adopted_no_defense"] = float(np.mean([
                answers_match(next(m for m in r["messages"] if m["role"] == "aggregator")["parsed_answer"],
                              r["fault"]["wrong_answer"]) for r in old]))
        stats[cid] = s
    faults = [s for s in stats.values() if "team_adopted" in s]
    stats["mean_fault_adoption"] = float(np.mean([s["team_adopted"] for s in faults]))
    stats["mean_fault_adoption_no_defense"] = float(np.mean([s["team_adopted_no_defense"] for s in faults]))
    return stats, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("defense", choices=["verifier", "skeptical"])
    p.add_argument("--label", required=True, help="iteration label, e.g. it1")
    p.add_argument("--log", default=LOG)
    p.add_argument("--config", default="configs/phase3_propagation.yaml")
    args = p.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config)
    llm = LLMClient(cfg.llm)
    questions = {q.id: q for q in load_questions("dev")}
    records = read_records(args.log)
    tune = tune_verifier if args.defense == "verifier" else tune_skeptical
    stats, raw = tune(llm, records, questions, cfg.seed, cfg.workers)

    raw_log = JsonlLog(f"results/dev/tuning/{args.defense}_{args.label}.jsonl")
    for r in raw:
        raw_log.write(r)
    doc = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    doc.setdefault(args.defense, {})[args.label] = {"prompts_sha": prompts.fingerprint(), **stats}
    SUMMARY.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
