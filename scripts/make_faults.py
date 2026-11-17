"""Generate data/faults.json: one wrong solution per dev and test question.

    python -m scripts.make_faults --config configs/baseline.yaml

    python -m scripts.make_faults --fix-implausible   # revision 2, see README

For each question, plan a calculation slip (src/faults.py), let the model write the
solution with that slip, and keep the first text that passes every check. Error plans
are tried in this order: from the gold rationale with the error in the first half of the
steps, then from a correct solution written by the model (useful when the gold rationale
gives its calculations in words), and only then with the error at any step.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src import prompts
from src.config import RunConfig
from src.data import Question, load_questions
from src.faults import (
    FAULTS_PATH, PERSUASIVE_SENTENCE, REWRITE_SYSTEM, check_fault_text, covariates, finalize,
    messages_sha, plan_errors, plan_to_dict, rewrite_messages, slip_is_plausible, slip_values,
)
from src.llm import LLMClient, derive_seed
from src.logging_utils import environment_metadata, now
from src.parser import parse_final_answer

BASE_SEED = 42
MANUAL_STEPS_PATH = FAULTS_PATH.parent / "fault_steps_manual.json"
BASE_TRIES = 6  # correct model solutions tried as a source of steps
REWRITE_TRIES = 12  # generations per error plan
MAX_PLANS = 4
REASONS = ("wrong equations", "is not the injected one", "missing calculation", "exactly once",
           "forbidden wording", "correct answer is computed", "final answer")


def reason(problem: str) -> str:
    return next((r for r in REASONS if r in problem), problem)


def correct_model_solutions(llm: LLMClient, q: Question) -> list[str]:
    texts = []
    for t in range(BASE_TRIES):
        out = llm.chat(
            [{"role": "system", "content": prompts.SOLVER_SYSTEM},
             {"role": "user", "content": prompts.solver_user(q.question)}],
            seed=derive_seed(q.id, "fault-base", t, base=BASE_SEED),
        )
        if parse_final_answer(out.text) == q.gold:
            texts.append(out.text)
    return texts


def manual_steps(qid: str) -> str | None:
    steps = json.loads(MANUAL_STEPS_PATH.read_text())["steps"].get(qid)
    return "\n".join(steps) if steps else None


def candidate_plans(llm: LLMClient, q: Question):
    """Yield (source, early_only, plans) in order of preference."""
    seed = derive_seed(q.id, "fault", base=BASE_SEED)
    model_texts = None
    manual = manual_steps(q.id)
    for early_only in (True, False):
        yield "gold", early_only, plan_errors(q.question, q.rationale, q.gold, seed, MAX_PLANS, early_only)
        if manual:
            yield "manual", early_only, plan_errors(q.question, manual, q.gold, seed, MAX_PLANS, early_only)
        if model_texts is None:
            model_texts = correct_model_solutions(llm, q)
        for text in model_texts:
            plans = plan_errors(q.question, text, q.gold, seed, MAX_PLANS, early_only)
            if plans:
                yield "model", early_only, plans
                break


def make_fault(llm: LLMClient, q: Question) -> dict:
    attempts, last_problems, rejections, n_plans = 0, [], Counter(), 0
    for source, early_only, plans in candidate_plans(llm, q):
        n_plans += len(plans)
        for p_idx, plan in enumerate(plans):
            for t in range(REWRITE_TRIES):
                attempts += 1
                seed = derive_seed(q.id, "fault-rewrite", source, early_only, p_idx, t, base=BASE_SEED)
                out = llm.chat(rewrite_messages(q.question, plan), seed=seed)
                if out.finish_reason != "stop":
                    last_problems = ["truncated"]
                    rejections["truncated"] += 1
                    continue
                # The model's own reasoning must reach the target; the final line is then
                # rewritten canonically and the checks run on the text actually used.
                if parse_final_answer(out.text) != plan.wrong_answer:
                    last_problems = [f"model reached {parse_final_answer(out.text)}"]
                    rejections["other final answer"] += 1
                    continue
                text = finalize(out.text, plan.wrong_answer)
                last_problems = check_fault_text(text, plan, q.gold)
                if last_problems:
                    rejections[reason(last_problems[0])] += 1
                    continue
                return {
                    "question_id": q.id, "gold": q.gold, "wrong_answer": plan.wrong_answer,
                    "text": text, "source": source, "early": early_only, "plan": plan_to_dict(plan),
                    **covariates(plan_to_dict(plan), q.gold),
                    "generation": {"seed": seed, "attempts": attempts, "rejections": dict(rejections)},
                }
    return {"question_id": q.id, "gold": q.gold, "failed": True, "n_plans": n_plans,
            "attempts": attempts, "last_problems": last_problems, "rejections": dict(rejections)}


def add_covariates(path: Path) -> None:
    doc = json.loads(path.read_text())
    before = messages_sha(doc)
    for f in doc["faults"].values():
        f.update(covariates(f["plan"], f["gold"]))
    if messages_sha(doc) != before:
        raise SystemExit("fault messages changed, not writing")
    doc["messages_sha"] = before
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"covariates added to {len(doc['faults'])} faults; messages hash unchanged ({before})")


def fix_implausible(llm: LLMClient, path: Path, workers: int) -> None:
    """Regenerate only the faults whose slip breaks the plausibility rule; every other
    text is left untouched. A fault that cannot be regenerated is kept and marked obvious."""
    doc = json.loads(path.read_text())
    questions = {q.id: q for s in ("dev", "test") for q in load_questions(s)}
    concerned = [qid for qid, f in doc["faults"].items() if not slip_is_plausible(*slip_values(f["plan"]))]
    untouched = {qid: f["text"] for qid, f in doc["faults"].items() if qid not in concerned}

    with ThreadPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(lambda qid: make_fault(llm, questions[qid]), concerned), total=len(concerned)))

    replaced, obvious = [], []
    for qid, r in zip(concerned, results):
        old = doc["faults"][qid]
        if r.get("failed"):
            old["obvious"] = True
            old["obvious_reason"] = "no plausible slip" if r["n_plans"] == 0 else "generation failed"
            obvious.append(qid)
        else:
            r["obvious"] = False
            r["replaces"] = {"wrong_answer": old["wrong_answer"], "original": old["plan"]["original"],
                             "corrupted": old["plan"]["corrupted"]}
            doc["faults"][qid] = r
            replaced.append(qid)
    for qid, f in doc["faults"].items():
        f.setdefault("obvious", False)
        f.update(covariates(f["plan"], f["gold"]))
    assert all(doc["faults"][qid]["text"] == text for qid, text in untouched.items())

    doc["messages_sha"] = messages_sha(doc)
    doc.setdefault("revisions", []).append({
        "date": now(), "rule": "slip on a step with true result >= 10, relative error 5-30%, trailing zeros kept",
        "replaced": replaced, "kept_as_obvious": obvious,
    })
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(concerned)} faults broke the rule: {len(replaced)} regenerated, {len(obvious)} kept as obvious")
    print(f"  {len(untouched)} other texts unchanged; new messages hash {doc['messages_sha']}")
    print(f"  regenerated, source: {dict(Counter(doc['faults'][q]['source'] for q in replaced))}, "
          f"corrupted step: {dict(sorted(Counter(doc['faults'][q]['plan']['step'] + 1 for q in replaced).items()))}")
    for qid in obvious:
        print(f"  OBVIOUS {qid}: {doc['faults'][qid]['obvious_reason']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/baseline.yaml", help="llm settings")
    p.add_argument("--splits", nargs="+", default=["dev", "test"])
    p.add_argument("--out", default=str(FAULTS_PATH))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--add-covariates", action="store_true",
                   help="only add the covariates to an existing file; texts are left untouched")
    p.add_argument("--fix-implausible", action="store_true",
                   help="regenerate only the faults that break the plausibility rule")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.add_covariates:
        add_covariates(Path(args.out))
        return

    cfg = RunConfig.from_yaml(args.config)
    llm = LLMClient(cfg.llm)
    if args.fix_implausible:
        fix_implausible(llm, Path(args.out), args.workers)
        return
    questions = [q for s in args.splits for q in load_questions(s)]

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(tqdm(pool.map(lambda q: make_fault(llm, q), questions), total=len(questions)))

    faults = {r["question_id"]: r for r in results if not r.get("failed")}
    failed = [r for r in results if r.get("failed")]
    doc = {
        "created": now(),
        "model": cfg.llm.model,
        "temperature": cfg.llm.temperature,
        "base_seed": BASE_SEED,
        "persuasive_sentence": PERSUASIVE_SENTENCE,
        "rewrite_system_prompt": REWRITE_SYSTEM,
        "server": llm.server_info(),
        "packages": environment_metadata()["packages"],
        "messages_sha": messages_sha({"faults": faults}),
        "faults": dict(sorted(faults.items())),
        "failed": failed,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    print(f"{len(faults)}/{len(questions)} faults written to {args.out}")
    print(f"  source: {dict(Counter(r['source'] for r in faults.values()))}, "
          f"error in first half: {sum(r['early'] for r in faults.values())}")
    print(f"  corrupted step: {dict(sorted(Counter(r['plan']['step'] + 1 for r in faults.values()).items()))}")
    attempts = [r["generation"]["attempts"] for r in faults.values()]
    if attempts:
        print(f"  generations per accepted fault: mean {sum(attempts) / len(attempts):.1f}, max {max(attempts)}")
    rejected = sum((Counter(r.get("generation", r).get("rejections", {})) for r in results), Counter())
    print(f"  rejected generations: {dict(rejected.most_common())}")
    for r in failed:
        print(f"  FAILED {r['question_id']}: {r['n_plans']} plans, {r['attempts']} attempts, {r['last_problems']}")


if __name__ == "__main__":
    main()
