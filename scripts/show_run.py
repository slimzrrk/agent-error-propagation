"""Print one logged conversation as Markdown.

    python -m scripts.show_run results/dev/phase2.jsonl --config chain-none-none --question train-53
"""

from __future__ import annotations

import argparse

from src.logging_utils import read_records
from src.parser import answers_match


def render(r: dict) -> str:
    lines = [f"### `{r['config_id']}` — question `{r['question_id']}`", ""]
    question = r.get("question")
    if question:
        lines += [f"**Problem.** {question}", ""]
    lines += [f"**Gold answer:** {r['gold']}  ", f"**Team answer:** {r['final_answer']} "
              f"({'correct' if r['correct'] else 'wrong'})", ""]
    for m in r["messages"]:
        mark = "✓" if answers_match(m["parsed_answer"], r["gold"]) else "✗"
        sees = f", sees {', '.join(m['sees'])}" if m["sees"] else ""
        faulty = " — FAULTY (scripted)" if m["is_faulty"] else ""
        lines += [
            f"#### {m['agent_id']} ({m['role']}{sees}){faulty} → {m['parsed_answer']} {mark}",
            "",
            "```text",
            m["content"].strip(),
            "```",
            "",
        ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--config", required=True)
    p.add_argument("--question", required=True)
    args = p.parse_args(argv)
    for r in read_records(args.log):
        if r["config_id"] == args.config and r["question_id"] == args.question:
            print(render(r))
            return
    raise SystemExit("no such run in this log")


if __name__ == "__main__":
    main()
