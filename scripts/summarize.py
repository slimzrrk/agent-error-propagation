"""Tables for one results file: accuracy and cost per config, per-agent accuracy, paired
tests, natural propagation, and the effect of injected faults (strict and lenient).

    python -m scripts.summarize results/dev/phase3_propagation.jsonl --save
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.logging_utils import read_records
from src.metrics import (
    adoption, answers_match, bootstrap_ci, by_obvious, by_slip_position, by_slot, fault_summary, hybrid_adopted,
    hybrid_correct, hybrid_cost, is_correct, majority_adopted, majority_answer, majority_correct,
    majority_vs_aggregator, mcnemar, paired_outcomes, per_agent, pooled_answer, solver_cost, solver_references,
    summarize, team_adopted, verifier_anchoring, verifier_stats,
)


def group(records: list[dict]) -> dict[str, list[dict]]:
    by_cfg = defaultdict(list)
    for r in records:
        by_cfg[r["config_id"]].append(r)
    return by_cfg


def ci(t) -> str:
    return f"{t[0]:5.0%} [{t[1]:4.0%}, {t[2]:4.0%}]" if not math.isnan(t[0]) else "    -"


def print_main_table(records: list[dict], config_order: list[str] | None = None) -> pd.DataFrame:
    by_cfg = group(records)
    order = [c for c in (config_order or sorted(by_cfg)) if c in by_cfg]
    rows = [{"config": cid, **summarize(by_cfg[cid])} for cid in order]
    header = (f"{'config':<30} {'n':>4} {'acc':>6} {'95% CI':>15} {'lenient':>8} {'invalid':>8}"
              f" {'calls':>6} {'tok/q':>7} {'s/call':>7}")
    print("\n" + header + "\n" + "-" * len(header))
    for s in rows:
        print(
            f"{s['config']:<30} {s['n']:>4} {s['accuracy']:>6.1%} [{s['acc_ci_low']:>5.1%}, {s['acc_ci_high']:>5.1%}]"
            f" {s['accuracy_lenient']:>8.1%} {s['invalid_rate']:>8.1%} {s['calls_per_q']:>6.1f}"
            f" {s['prompt_tokens_per_q'] + s['completion_tokens_per_q']:>7.0f} {s['latency_per_call']:>7.2f}"
        )
    return pd.DataFrame(rows)


def paired_vs_reference(by_cfg: dict, reference: str) -> None:
    print(f"\nPaired McNemar tests vs {reference} (final answer correct)")
    for cid, recs in sorted(by_cfg.items()):
        if cid == reference:
            continue
        t = paired_outcomes(recs, by_cfg[reference], lambda r: r["correct"])
        if t["n"]:
            print(f"  {cid:<28} n={t['n']:>3}  {t['rate_a']:.0%} vs {t['rate_b']:.0%}  "
                  f"wins {t['only_a']:>2} / losses {t['only_b']:>2}  p={t['p_value']:.3f}")


def natural_propagation(teams: dict, source: str) -> dict:
    print(f"\nNatural propagation: other agents ending with {source}'s own wrong answer (no fault injected)")
    out = {}
    for cid, recs in sorted(teams.items()):
        for lenient in (False, True):
            res = adoption(recs, source=source, lenient=lenient)
            out[f"{cid}{'|lenient' if lenient else ''}"] = res
            label = "lenient" if lenient else "strict"
            line = (f"  {cid:<24} {label:<8} cases={res['n_cases']:>3}  downstream "
                    f"{ci((res['adoption_rate'], *res['adoption_ci']))}")
            if not math.isnan(res["control_rate"]):
                line += f"  independent {ci((res['control_rate'], *res['control_ci']))}"
            print(line)
    return out


def fault_tables(faulty: dict) -> tuple[dict, list[dict]]:
    print("\nInjected faults: recovery = final answer still correct; team adopted = final answer is "
          "the injected one;\ndownstream = agents that saw the faulty solution and give its answer; "
          "independent = same answer by coincidence")
    head = f"  {'config':<30} {'':<8} {'recovery':>17} {'team adopted':>17} {'downstream':>17} {'independent':>17}"
    print(head)
    out = {}
    for cid, recs in sorted(faulty.items()):
        for lenient in (False, True):
            s = fault_summary(recs, lenient=lenient)
            out[f"{cid}{'|lenient' if lenient else ''}"] = s
            print(f"  {cid if not lenient else '':<30} {'lenient' if lenient else 'strict':<8} "
                  f"{ci(s['recovery']):>17} {ci(s['team_adopted']):>17} "
                  f"{ci(s['downstream_adoption']):>17} {ci(s['coincidence']):>17}")
        cells = [f"{a['agent_id']} {a['adopted']:.0%}" for a in out[cid]["agents"] if a["relation"] == "downstream"]
        print(f"  {'':<30} {'':<8} per downstream agent (strict): " + ", ".join(cells))

    print("\nNeutral vs persuasive (paired by question, exact McNemar)")
    tests = []
    pairs = sorted({cid.replace("persuasive", "STYLE").replace("neutral", "STYLE") for cid in faulty})
    pooled = defaultdict(lambda: ([], []))
    for pattern in pairs:
        neu, per = faulty.get(pattern.replace("STYLE", "neutral")), faulty.get(pattern.replace("STYLE", "persuasive"))
        if not (neu and per):
            continue
        topo = pattern.split("-")[0]
        for r in neu:
            pooled[topo][0].append({**r, "question_id": f"{r['question_id']}|{pattern}"})
        for r in per:
            pooled[topo][1].append({**r, "question_id": f"{r['question_id']}|{pattern}"})
        tests += compare_styles(pattern.replace("STYLE", "*"), neu, per)
    for topo, (neu, per) in sorted(pooled.items()):
        tests += compare_styles(f"{topo} (all positions)", neu, per)
    return out, tests


def compare_styles(label: str, neutral: list[dict], persuasive: list[dict]) -> list[dict]:
    rows = []
    for name, outcome in (("team adopted", team_adopted), ("recovery", is_correct)):
        t = paired_outcomes(persuasive, neutral, outcome)
        rows.append({"comparison": label, "outcome": name, **t})
        print(f"  {label:<30} {name:<13} persuasive {t['rate_a']:.0%} vs neutral {t['rate_b']:.0%}  "
              f"(n={t['n']}, discordant {t['only_a']}/{t['only_b']}, p={t['p_value']:.3f})")
    return rows


def position_rank_table(faulty: dict) -> dict:
    """chain_rotated: team adoption by chain position x presentation rank (both styles)."""
    rotated = [r for cid, recs in faulty.items() if cid.startswith("chain_rotated") for r in recs]
    if not rotated:
        return {}
    print("\nchain_rotated: team adoption by position in the chain (rows) and presentation rank (columns), both styles")
    cells = defaultdict(list)
    for r in rotated:
        cells[(r["fault"]["position"], r["fault"]["shown_at"])].append(team_adopted(r))
    out = {}
    print(f"  {'':<10}" + "".join(f"{'rank ' + str(k):>16}" for k in (1, 2, 3)) + f"{'all ranks':>16}")
    for p in (1, 2, 3):
        row = [cells[(p, k)] for k in (1, 2, 3)]
        allv = [v for c in row for v in c]
        out[f"p{p}"] = {f"rank{k}": (sum(c) / len(c), len(c)) for k, c in zip((1, 2, 3), row) if c}
        print(f"  {'p' + str(p):<10}" + "".join(f"{sum(c) / len(c):>10.0%} (n={len(c):>2})" for c in row)
              + f"{sum(allv) / len(allv):>10.0%} (n={len(allv)})")
    for k in (1, 2, 3):
        col = [v for p in (1, 2, 3) for v in cells[(p, k)]]
        out[f"rank{k}"] = (sum(col) / len(col), len(col))
    print(f"  {'all pos.':<10}" + "".join(f"{out[f'rank{k}'][0]:>10.0%} (n={out[f'rank{k}'][1]})" for k in (1, 2, 3)))
    return out


def slot_table(faulty: dict) -> dict:
    out = {}
    star = {cid: recs for cid, recs in faulty.items() if cid.startswith("star")}
    if not star:
        return out
    print("\nStar: effect of the slot where the faulty solution is shown to the aggregator")
    for cid, recs in sorted(star.items()):
        out[cid] = by_slot(recs)
        for row in out[cid]:
            print(f"  {cid:<30} slot {row['group']}  n={row['n']:>3}  team adopted {ci(row['team_adopted'])}"
                  f"  recovery {ci(row['recovery'])}")
    all_recs = [r for recs in star.values() for r in recs]
    out["star (both styles)"] = by_slot(all_recs)
    for row in out["star (both styles)"]:
        print(f"  {'star (both styles)':<30} slot {row['group']}  n={row['n']:>3}  team adopted {ci(row['team_adopted'])}")
    return out


def covariate_table(faulty: dict, title: str, split) -> dict:
    print(f"\nExploratory: fault effects by {title}")
    out = {}
    groups = {**faulty, "all fault configs": [r for recs in faulty.values() for r in recs]}
    for cid, recs in sorted(groups.items()):
        out[cid] = split(recs)
        for row in out[cid]:
            print(f"  {cid:<30} {str(row['group']):<5} n={row['n']:>3}  downstream {ci(row['downstream_adoption'])}"
                  f"  team adopted {ci(row['team_adopted'])}  recovery {ci(row['recovery'])}")
    return out


def majority_table(team_cfgs: dict) -> dict:
    print("\nLLM aggregator vs counterfactual votes over the same solver answers (no extra LLM call): majority"
          "\n(three different answers = no majority = wrong) and hybrid (majority if any, else the aggregator)."
          "\nMcNemar p: aggregator vs majority, aggregator vs hybrid.")
    out = {}
    for cid, recs in sorted(team_cfgs.items()):
        res = majority_vs_aggregator(recs)
        agg_ok = [is_correct(r) for r in recs]
        hyb_ok = [hybrid_correct(r) for r in recs]
        res.update(hybrid_acc=bootstrap_ci(hyb_ok), hybrid_acc_test=mcnemar(hyb_ok, agg_ok))
        line = (f"  {cid:<28} accuracy agg {ci(res['aggregator_acc'])} maj {ci(res['majority_acc'])}"
                f" p={res['acc_test']['p_value']:.3f} hyb {ci(res['hybrid_acc'])} p={res['hybrid_acc_test']['p_value']:.3f}")
        if "adopt_test" in res:
            hyb_ad = [hybrid_adopted(r) for r in recs]
            res.update(hybrid_adopted=bootstrap_ci(hyb_ad),
                       hybrid_adopt_test=mcnemar(hyb_ad, [team_adopted(r) for r in recs]))
            line += (f" | adopted agg {res['aggregator_adopted'][0]:4.0%} maj {res['majority_adopted'][0]:4.0%}"
                     f" p={res['adopt_test']['p_value']:.3f} hyb {res['hybrid_adopted'][0]:4.0%}"
                     f" p={res['hybrid_adopt_test']['p_value']:.3f}")
        else:
            line += f" | oracle {ci(solver_references(recs)['oracle_acc'])}"
        out[cid] = res
        print(line)
    return out


DEFENSES = ("none", "skeptical", "verifier")


def cell_of(r: dict) -> str:
    c = r["config"]
    return f"{c['topology']}-{c.get('fault', 'none')}" + (f"-p{c['position']}" if c.get("position") else "")


RECOVERY = {"majority vote": majority_correct, "hybrid vote": hybrid_correct}


def method_rows(by_def: dict) -> list[dict]:
    """One row per method on a cell: the no-defense team, a majority vote over its solvers
    (the zero-cost defense), and each defense; paired tests against both references."""
    base = by_def["none"]
    faulty = bool(base[0].get("fault"))
    qids = sorted(set.intersection(*(set(r["question_id"] for r in recs) for recs in by_def.values())))

    def pick(recs):
        by_q = {r["question_id"]: r for r in recs}
        return [by_q[q] for q in qids]

    outcome = (lambda r: team_adopted(r)) if faulty else (lambda r: r["correct"])
    ref_none = [outcome(r) for r in pick(base)]
    ref_maj = [(majority_adopted(r) if faulty else majority_correct(r)) for r in pick(base)]
    methods = [("none", pick(base), outcome, lambda r: (r["n_llm_calls"], r["prompt_tokens"] + r["completion_tokens"])),
               ("majority vote", pick(base), majority_adopted if faulty else majority_correct, solver_cost),
               ("hybrid vote", pick(base), hybrid_adopted if faulty else hybrid_correct, hybrid_cost)]
    methods += [(d, pick(by_def[d]), outcome, lambda r: (r["n_llm_calls"], r["prompt_tokens"] + r["completion_tokens"]))
                for d in DEFENSES[1:] if d in by_def]
    rows = []
    for name, recs, out, cost in methods:
        x = [out(r) for r in recs]
        costs = [cost(r) for r in recs]
        rows.append({
            "method": name, "n": len(recs), "outcome": "team adopted" if faulty else "accuracy",
            "rate": bootstrap_ci(x),
            "recovery": bootstrap_ci([RECOVERY.get(name, is_correct)(r) for r in recs]) if faulty else None,
            "calls_per_q": sum(c[0] for c in costs) / len(costs), "tokens_per_q": sum(c[1] for c in costs) / len(costs),
            "vs_none": mcnemar(x, ref_none) if name != "none" else None,
            "vs_majority": mcnemar(x, ref_maj) if name != "majority vote" else None,
        })
    return rows


def defense_table(by_cfg: dict) -> dict:
    """Defenses against no defense and against a majority vote, cell by cell and pooled."""
    cells = defaultdict(dict)
    for cid, recs in by_cfg.items():
        c = recs[0]["config"]
        if c["topology"] in ("chain", "star"):
            cells[cell_of(recs[0])][c.get("defense", "none")] = recs
    cells = {k: v for k, v in cells.items() if "none" in v and len(v) > 1}
    if not cells:
        return {}
    print("\nDefenses: team adoption with a fault (accuracy without), recovery, cost; McNemar vs no defense and"
          "\nvs a majority vote over the no-defense solvers (the zero-cost defense). Hybrid vote: majority if"
          "\nany, else the aggregator; its cost assumes the aggregator is only called without majority.")
    out = {}
    pooled = defaultdict(lambda: defaultdict(list))
    for cell, by_def in sorted(cells.items()):
        topo, fault = cell.split("-")[:2]
        if fault != "none":
            for d, recs in by_def.items():
                pooled[f"{topo}-all faults"][d] += [{**r, "question_id": f"{r['question_id']}|{cell}"} for r in recs]
    for cell, by_def in [*sorted(cells.items()), *sorted(pooled.items())]:
        out[cell] = method_rows(by_def)
        for row in out[cell]:
            tests = "".join(f"  {k.replace('vs_', 'vs ')} p={row[k]['p_value']:.3f}" for k in ("vs_none", "vs_majority")
                            if row[k])
            rec = f" recovery {ci(row['recovery'])}" if row["recovery"] else ""
            print(f"  {cell:<20} {row['method']:<14} {row['outcome']:<12} {ci(row['rate'])}{rec}"
                  f"  calls {row['calls_per_q']:.1f} tok {row['tokens_per_q']:6.0f}{tests}")
    return out


def verifier_table(by_cfg: dict) -> dict:
    ver = {cid: recs for cid, recs in by_cfg.items() if any(r.get("verdicts") for r in recs)}
    if not ver:
        return {}
    print("\nVerifier (positive = checked message wrong): recall, precision, false positives = correct messages"
          "\ndropped, recall on the injected faulty message, verdicts not parsed (kept), runs with every solution dropped")
    out = {}
    groups = {**ver, "all verifier configs": [r for recs in ver.values() for r in recs],
              "no-fault configs": [r for recs in ver.values() for r in recs if not r.get("fault")]}
    for cid, recs in sorted(groups.items()):
        out[cid] = v = verifier_stats(recs)
        print(f"  {cid:<30} n={v['n_verdicts']:>4} wrong={v['n_wrong']:>3}  recall {ci(v['recall'])}"
              f"  precision {ci(v['precision'])}  false pos. {ci(v['false_positive_rate'])}"
              f"  fault recall {ci(v['fault_recall'])}  unparsed {v['unparsed'][0]:.1%}"
              f" (kept, {v['unparsed_wrong'][0]:.0%} of them wrong)  all dropped {v['all_dropped'][0]:.0%}")
    return out


def anchoring_table(by_cfg: dict, threshold: float = 0.9) -> dict:
    recs = [r for rs in by_cfg.values() for r in rs if r.get("verdicts")]
    if not recs:
        return {}
    print(f"\nExploratory: verifier anchoring, difflib ratio between the verifier's text and the checked message"
          f"\n(near copy = ratio >= {threshold}); recall on wrong targets for near copies and for the others")
    out = verifier_anchoring(recs, threshold)
    for kind, row in out.items():
        print(f"  {kind:<8} n={row['n']:>4}  mean ratio {row['mean_ratio']:.2f}  near copies {row['near_copy']:.0%}"
              f"  recall: near copies {row['recall_near_copy']:.0%} (n={row['n_wrong_near_copy']}),"
              f" others {row['recall_not_near_copy']:.0%} (n={row['n_wrong_not_near_copy']})")
    return out


def pooled_table(by_cfg: dict) -> dict:
    ver = {cid: rs for cid, rs in by_cfg.items() if any(r.get("verdicts") for r in rs)}
    if not ver:
        return {}
    print("\nExploratory: pooled vote, strict majority over the 3 solvers and the 3 verifiers (4 of 6 answers),"
          "\nagainst the team (LLM aggregator) and the solvers' majority, in the verifier configs")
    out = {}
    groups = {**ver, "verifier configs with a fault": [r for rs in ver.values() for r in rs if r.get("fault")]}
    for cid, recs in sorted(groups.items()):
        row = {"n": len(recs)}
        for name, answer in (("team", lambda r: r["final_answer"]), ("majority", majority_answer),
                             ("pooled", pooled_answer)):
            row[f"{name}_acc"] = bootstrap_ci([answers_match(answer(r), r["gold"]) for r in recs])
            if recs[0].get("fault"):
                row[f"{name}_adopted"] = bootstrap_ci([answers_match(answer(r), r["fault"]["wrong_answer"]) for r in recs])
        out[cid] = row
        line = "  ".join(f"{n} {ci(row[f'{n}_acc'])}" for n in ("team", "majority", "pooled"))
        if recs[0].get("fault"):
            line += "  | adopted " + " ".join(f"{n} {row[f'{n}_adopted'][0]:.0%}" for n in ("team", "majority", "pooled"))
        print(f"  {cid:<32} accuracy {line}")
    return out


def json_safe(obj):
    """NaN -> null, tuples -> lists, so the file is strict JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--reference", default="single", help="config for the paired McNemar tests")
    p.add_argument("--source", default="solver1", help="agent whose own wrong answers we follow")
    p.add_argument("--save", action="store_true", help="write <log>_summary.csv, _propagation.json, _tests.csv")
    args = p.parse_args(argv)

    records = read_records(args.log)
    by_cfg = group(records)
    main_df = print_main_table(records)

    print("\nPer agent (accuracy / invalid / prompt tok / completion tok)")
    for cid, recs in sorted(by_cfg.items()):
        cells = [f"{a['agent_id']} {a['accuracy']:.0%}/{a['invalid_rate']:.0%}/{a['prompt_tokens']:.0f}/{a['completion_tokens']:.0f}"
                 for a in per_agent(recs)]
        print(f"  {cid:<28} " + " | ".join(cells))

    teams = {cid: recs for cid, recs in by_cfg.items()
             if not recs[0].get("fault") and any(m["role"] == "aggregator" for m in recs[0]["messages"])}
    faulty = {cid: recs for cid, recs in by_cfg.items() if recs[0].get("fault")}

    saved = {}
    if teams or faulty:
        saved["majority"] = majority_table({**teams, **faulty})
    saved["defenses"] = defense_table(by_cfg)
    saved["verifier"] = verifier_table(by_cfg)
    saved["anchoring"] = anchoring_table(by_cfg)
    saved["pooled"] = pooled_table(by_cfg)
    if args.reference in by_cfg:
        paired_vs_reference(by_cfg, args.reference)

    if teams:
        saved["natural"] = natural_propagation(teams, args.source)
    tests = []
    if faulty:
        saved["faults"], tests = fault_tables(faulty)
        saved["slots"] = slot_table(faulty)
        saved["position_rank"] = position_rank_table(faulty)
        saved["slip_position"] = covariate_table(
            faulty, "position of the slip in the faulty solution (early <= 50% of the steps)", by_slip_position)
        saved["obvious"] = covariate_table(faulty, "obvious faults (True = breaks the plausibility rule)", by_obvious)

    if args.save:
        stem = Path(args.log).with_suffix("")
        main_df.to_csv(f"{stem}_summary.csv", index=False, float_format="%.4f")
        Path(f"{stem}_propagation.json").write_text(json.dumps(json_safe(saved), indent=1, allow_nan=False) + "\n")
        if tests:
            pd.DataFrame(tests).to_csv(f"{stem}_tests.csv", index=False, float_format="%.4f")
        print(f"\nsaved {stem}_summary.csv, {stem}_propagation.json" + (f", {stem}_tests.csv" if tests else ""))


if __name__ == "__main__":
    main()
