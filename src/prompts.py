"""System prompts and message templates. Tuned on the dev split only.

The no-defense prompts are deliberately neutral: they neither encourage nor discourage
trusting other agents, so that the skeptical defense is a clean intervention.
"""

import hashlib
from pathlib import Path

FINAL_ANSWER_RULE = (
    "End your response with a last line of the form:\n"
    "Final answer: <number>\n"
    "where <number> is a single number, without units."
)

# The 1.5B model often drops the format line when it is only in the system prompt,
# so it is repeated at the end of the user message.
FORMAT_REMINDER = 'Remember to end with the line "Final answer: <number>".'

SOLVER_SYSTEM = (
    "You are a careful math problem solver. Solve the problem step by step, "
    "showing each calculation.\n" + FINAL_ANSWER_RULE
)

AGGREGATOR_SYSTEM = (
    "You coordinate a team of math problem solvers. You are given a math problem and the "
    "solutions written by the team members. Compare the solutions in a few sentences, "
    "then decide on the correct final answer to the problem.\n" + FINAL_ANSWER_RULE
)

# Without this line the aggregator only copies an answer, with no comparison at all.
AGGREGATOR_INSTRUCTION = "Compare the solutions briefly, then give your decision."


def solver_user(question: str, previous: str | None = None) -> str:
    """First solver (or star solver) sees the problem only; chain solvers also see the previous solution."""
    if previous is None:
        return f"Problem: {question}\n\n{FORMAT_REMINDER}"
    return (
        f"Problem: {question}\n\n"
        f"Another agent proposed this solution:\n<solution>\n{previous.strip()}\n</solution>\n\n"
        "You may keep this answer or revise it. Write your own step-by-step solution.\n"
        f"{FORMAT_REMINDER}"
    )


def aggregator_user(question: str, solutions: list[str]) -> str:
    parts = [f"Problem: {question}"]
    for i, text in enumerate(solutions, 1):
        parts.append(f'<solution id="{i}">\n{text.strip()}\n</solution>')
    parts.append(f"{AGGREGATOR_INSTRUCTION}\n{FORMAT_REMINDER}")
    return "\n\n".join(parts)


def fingerprint() -> str:
    """Short hash of this file, stored with results so runs with different prompts are never mixed."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
