"""Tables for one results file: accuracy and cost per config, per-agent accuracy,
paired tests against a reference config, and error propagation.

    python -m scripts.summarize results/dev/phase2.jsonl --reference single
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.metrics import adoption, mcnemar, per_agent, solver_references, summarize
from src.logging_utils import read_records


def group(records: list[dict]) -> dict[str, list[dict]]:
    by_cfg = defaultdict(list)
    for r in records:
        by_cfg[r["config_id"]].append(r)
    return by_cfg


def print_main_table(records: list[dict], config_order: list[str] | None = None) -> pd.DataFrame:
    by_cfg = group(records)
    order = [c for c in (config_order or sorted(by_cfg)) if c in by_cfg]
    rows = [{"config": cid, **summarize(by_cfg[cid])} for cid in order]
    header = (f"{'config':<30} {'n':>4} {'acc':>6} {'95% CI':>15} {'invalid':>8} {'trunc':>6}"
              f" {'calls':>6} {'tok/q':>7} {'s/call':>7}")
    print("\n" + header + "\n" + "-" * len(header))
    for s in rows:
        print(
            f"{s['config']:<30} {s['n']:>4} {s['accuracy']:>6.1%} [{s['acc_ci_low']:>5.1%}, {s['acc_ci_high']:>5.1%}]"
            f" {s['invalid_rate']:>8.1%} {s['truncated_rate']:>6.1%} {s['calls_per_q']:>6.1f}"
            f" {s['prompt_tokens_per_q'] + s['completion_tokens_per_q']:>7.0f} {s['latency_per_call']:>7.2f}"
        )
    return pd.DataFrame(rows)


def paired_tests(by_cfg: dict, reference: str) -> list[dict]:
    ref = {r["question_id"]: r["correct"] for r in by_cfg[reference]}
    rows = []
    for cid, recs in by_cfg.items():
        if cid == reference:
            continue
        pairs = [(r["correct"], ref[r["question_id"]]) for r in recs if r["question_id"] in ref]
        if not pairs:
            continue
        a, b = zip(*pairs)
        res = mcnemar(a, b)
        rows.append({"config": cid, "vs": reference, "n": len(pairs),
                     "delta_acc": sum(a) / len(a) - sum(b) / len(b), **res})
    return rows


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
    p.add_argument("--source", default="solver1", help="agent whose wrong answers we follow")
    p.add_argument("--save", action="store_true", help="write <log>_summary.csv and <log>_propagation.json")
    args = p.parse_args(argv)

    records = read_records(args.log)
    by_cfg = group(records)
    main_df = print_main_table(records)

    print("\nPer agent (accuracy / invalid / prompt tok / completion tok)")
    for cid, recs in sorted(by_cfg.items()):
        cells = [f"{a['agent_id']} {a['accuracy']:.0%}/{a['invalid_rate']:.0%}/{a['prompt_tokens']:.0f}/{a['completion_tokens']:.0f}"
                 for a in per_agent(recs)]
        print(f"  {cid:<28} " + " | ".join(cells))

    teams = {cid: recs for cid, recs in by_cfg.items() if any(m["role"] == "aggregator" for m in recs[0]["messages"])}
    if teams:
        print("\nAggregator vs non-LLM references (accuracy with 95% CI)")
        for cid, recs in sorted(teams.items()):
            s = summarize(recs)
            ref = solver_references(recs)
            fmt = lambda t: f"{t[0]:.0%} [{t[1]:.0%}, {t[2]:.0%}]"  # noqa: E731
            print(f"  {cid:<28} aggregator {fmt((s['accuracy'], s['acc_ci_low'], s['acc_ci_high']))}"
                  f"  majority {fmt(ref['majority_acc'])}  oracle {fmt(ref['oracle_acc'])}")

    if args.reference in by_cfg:
        print(f"\nPaired McNemar tests vs {args.reference}")
        for t in paired_tests(by_cfg, args.reference):
            print(f"  {t['config']:<28} n={t['n']:>3}  delta {t['delta_acc']:+.1%}  "
                  f"wins {t['only_a']:>2} / losses {t['only_b']:>2}  p={t['p_value']:.3f}")

    propagation = {}
    if teams:
        print(f"\nPropagation of {args.source}'s own wrong answers")
        for cid, recs in sorted(teams.items()):
            res = adoption(recs, source=args.source)
            propagation[cid] = res
            ci = res["adoption_ci"]
            line = f"  {cid:<28} cases={res['n_cases']:>3}  downstream adoption {res['adoption_rate']:.0%} [{ci[0]:.0%}, {ci[1]:.0%}]"
            if not math.isnan(res["control_rate"]):
                cci = res["control_ci"]
                line += f"  | independent (coincidence) {res['control_rate']:.0%} [{cci[0]:.0%}, {cci[1]:.0%}]"
            print(line)
            for a in res["agents"]:
                print(f"      {a['agent_id']:<11} {a['relation']:<11} n={a['n']:>3}"
                      f"  same wrong answer {a['adopted']:.0%}  correct {a['correct']:.0%}")

    if args.save:
        stem = Path(args.log).with_suffix("")
        main_df.to_csv(f"{stem}_summary.csv", index=False, float_format="%.4f")
        Path(f"{stem}_propagation.json").write_text(json.dumps(json_safe(propagation), indent=2, allow_nan=False) + "\n")
        print(f"\nsaved {stem}_summary.csv and {stem}_propagation.json")


if __name__ == "__main__":
    main()
