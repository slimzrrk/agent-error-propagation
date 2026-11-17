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
    adoption, by_slip_position, by_slot, fault_summary, is_correct, paired_outcomes, per_agent,
    solver_references, summarize, team_adopted,
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


def slip_position_table(faulty: dict) -> dict:
    print("\nExploratory: fault effects by position of the slip in the faulty solution (early <= 50% of the steps)")
    out = {}
    groups = {**faulty, "all fault configs": [r for recs in faulty.values() for r in recs]}
    for cid, recs in sorted(groups.items()):
        out[cid] = by_slip_position(recs)
        for row in out[cid]:
            print(f"  {cid:<30} {str(row['group']):<5} n={row['n']:>3}  downstream {ci(row['downstream_adoption'])}"
                  f"  team adopted {ci(row['team_adopted'])}  recovery {ci(row['recovery'])}")
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

    if teams:
        print("\nAggregator vs non-LLM references (no fault)")
        for cid, recs in sorted(teams.items()):
            s, ref = summarize(recs), solver_references(recs)
            print(f"  {cid:<28} aggregator {ci((s['accuracy'], s['acc_ci_low'], s['acc_ci_high']))}"
                  f"  majority {ci(ref['majority_acc'])}  oracle {ci(ref['oracle_acc'])}")
    if args.reference in by_cfg:
        paired_vs_reference(by_cfg, args.reference)

    saved = {}
    if teams:
        saved["natural"] = natural_propagation(teams, args.source)
    tests = []
    if faulty:
        saved["faults"], tests = fault_tables(faulty)
        saved["slots"] = slot_table(faulty)
        saved["slip_position"] = slip_position_table(faulty)

    if args.save:
        stem = Path(args.log).with_suffix("")
        main_df.to_csv(f"{stem}_summary.csv", index=False, float_format="%.4f")
        Path(f"{stem}_propagation.json").write_text(json.dumps(json_safe(saved), indent=1, allow_nan=False) + "\n")
        if tests:
            pd.DataFrame(tests).to_csv(f"{stem}_tests.csv", index=False, float_format="%.4f")
        print(f"\nsaved {stem}_summary.csv, {stem}_propagation.json" + (f", {stem}_tests.csv" if tests else ""))


if __name__ == "__main__":
    main()
