"""System prompts and message templates. Tuned on the dev split only."""

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


def solver_user(question: str) -> str:
    return f"Problem: {question}\n\n{FORMAT_REMINDER}"


def fingerprint() -> str:
    """Short hash of this file, stored with results so runs with different prompts are never mixed."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
