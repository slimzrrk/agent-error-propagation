"""Scripted faulty agent: pre-computed wrong solutions with exactly one calculation error.

Building a wrong solution for a question:
1. Extract the calculation steps of the gold rationale ("a op b = c").
2. Change the result of the earliest possible step by a small amount (a calculation
   slip) and recompute every later step that uses it, so the rest stays consistent.
3. Ask the model to write a solution in its usual style that uses exactly these
   calculations (scripts/make_faults.py), and keep it only if automatic checks pass.

The persuasive style is the neutral text plus one confidence sentence, nothing else.
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import random
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from src.agents import Message
from src.parser import parse_final_answer

FAULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "faults.json"
PERSUASIVE_SENTENCE = "I am completely sure, I double-checked every step."

# ---------------------------------------------------------------- arithmetic

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval(expr: str) -> Fraction:
    """Exact evaluation of + - * / and parentheses; raises ValueError on anything else."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = ev(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("division by zero")
            return _OPS[type(node.op)](ev(node.left), right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        raise ValueError(f"unsupported expression: {expr!r}")

    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except SyntaxError as exc:
        raise ValueError(str(exc)) from None


def fmt(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- equations in text

_NUM = r"-?\d+(?:\.\d+)?"
_OPERAND = rf"\(*\s*{_NUM}\s*\)*"
# The result may be written as a fraction ("= 6/6").
_EQUATION = re.compile(
    rf"(?P<lhs>{_OPERAND}(?:\s*[-+*/]\s*{_OPERAND})+)\s*=\s*(?P<rhs>{_NUM}(?:\s*/\s*\d+)?)(?![\d.]*\d)"
)
_NUM_TOKEN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")


def normalize_math(text: str) -> str:
    """Turn the many ways of writing arithmetic (LaTeX, x, ×, $, units...) into plain a * b = c."""
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    t = re.sub(r"\\text\{[^{}]*\}", " ", t)
    for a, b in [("\\times", "*"), ("\\cdot", "*"), ("×", "*"), ("·", "*"), ("\\div", "/"), ("÷", "/"),
                 ("−", "-"), ("–", "-"), ("\\left", ""), ("\\right", ""), ("\\$", ""), ("$", ""),
                 ("€", ""), ("£", ""), ("\\(", " "), ("\\)", " "), ("\\[", " "), ("\\]", " "),
                 ("\\,", " "), ("\\%", "%"), ("​", "")]:
        t = t.replace(a, b)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # thousands separators
    t = re.sub(r"(?<![\d.])\.(?=\d)", "0.", t)  # .3 -> 0.3
    t = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", " * ", t)  # 12 x 80
    t = re.sub(r"(?<=[\d)])\s*\((?=\s*\d)", " * (", t)  # 2(40 + 9)
    # "12/20 * 100% = 60%" is a conversion to percent: both sides are in percent units.
    t = re.sub(r"\*\s*100\s*%(\s*=\s*\d+(?:\.\d+)?)\s*%", r"* 100\1", t)
    t = re.sub(r"(\*\s*100\s*=\s*\d+(?:\.\d+)?)\s*%", r"\1", t)
    t = t.replace("%", "/100")
    t = re.sub(r"/\s*[A-Za-z]+\b", " ", t)  # units such as pages/inch or 3/game
    t = re.sub(r"[A-Za-z_{}\\]+", " ", t)  # remaining words
    t = re.sub(r"(?<=\d)\.(?!\d)", " ", t)  # sentence-final dots
    return re.sub(r"[ \t]+", " ", t)


@dataclass
class Equation:
    lhs: str
    rhs: str
    lhs_value: Fraction = field(repr=False)
    rhs_value: Fraction = field(repr=False)

    @property
    def correct(self) -> bool:
        # Allow rounding of the written result (10 / 3 = 3.33), but not a slip of one unit.
        decimals = len(self.rhs.split(".")[1]) if "." in self.rhs else 0
        return abs(self.lhs_value - self.rhs_value) <= Fraction(1, 2 * 10**decimals)

    def __str__(self) -> str:
        return f"{self.lhs} = {self.rhs}"


def extract_equations(text: str) -> list[Equation]:
    out = []
    for m in _EQUATION.finditer(normalize_math(text)):
        lhs = re.sub(r"\s*([-+*/])\s*", r" \1 ", m.group("lhs").strip())
        lhs = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", lhs))
        try:
            value = safe_eval(lhs)
        except (ValueError, ZeroDivisionError):
            continue
        if lhs.count("(") != lhs.count(")"):
            continue
        rhs = re.sub(r"\s+", "", m.group("rhs"))
        try:
            rhs_value = Fraction(rhs)
        except ZeroDivisionError:
            continue
        out.append(Equation(lhs, rhs, value, rhs_value))
    return out


_ANNOTATION = re.compile(r"<<([^<>=]*)=([^<>]*)>>")


def rationale_steps(rationale: str) -> list[Equation]:
    """Correct calculation steps of a GSM8K rationale, in order.

    Annotations <<expr=value>> usually duplicate a visible "expr = value"; when the
    expression is not visible ("he earns <<12*5=60>>60"), it is made visible.
    """
    def expand(m):
        before = rationale[: m.start()].rstrip(" $")
        return "" if before.endswith("=") else f" {m.group(1)} = "

    text = _ANNOTATION.sub(expand, rationale)
    return [e for e in extract_equations(text) if e.correct]


# ---------------------------------------------------------------- error injection

@dataclass
class ErrorPlan:
    step: int  # index of the corrupted step
    original: str  # correct equation
    corrupted: str  # same left-hand side, wrong result
    calculations: list[str]  # all steps after the error is propagated
    wrong_answer: str
    offset: str  # wrong result - correct result


def _substitute(lhs: str, changes: dict[Fraction, Fraction]) -> tuple[str, int]:
    count = 0

    def repl(m):
        nonlocal count
        v = Fraction(m.group(0))
        if v in changes:
            count += 1
            return fmt(changes[v])
        return m.group(0)

    return _NUM_TOKEN.sub(repl, lhs), count


def propagate(steps: list[Equation], i: int, new_value: Fraction, question: str) -> tuple[list[str], dict] | None:
    """Corrupt step i and recompute later steps. None if a later step is ambiguous."""
    old = steps[i].rhs_value
    given = {Fraction(n) for n in _NUM_TOKEN.findall(normalize_math(question))}
    unchanged = {s.rhs_value for s in steps[:i]}  # earlier results that keep their value
    changes = {old: new_value}
    calcs = [str(s) for s in steps[:i]] + [f"{steps[i].lhs} = {fmt(new_value)}"]
    for s in steps[i + 1:]:
        lhs, n = _substitute(s.lhs, changes)
        value = s.rhs_value
        if n:
            hits = [Fraction(t) for t in _NUM_TOKEN.findall(s.lhs) if Fraction(t) in changes]
            # If a changed value is also a number given in the problem, or the result of
            # an unchanged earlier step, we cannot tell which one this step uses.
            if any(h in given or h in unchanged for h in hits):
                return None
            try:
                value = safe_eval(lhs)
            except (ValueError, ZeroDivisionError):
                return None
            if s.rhs_value != value:
                changes[s.rhs_value] = value
        if value == s.rhs_value:
            unchanged.add(value)
        calcs.append(f"{lhs} = {fmt(value)}")
    return calcs, changes


def slip_offsets(r: Fraction, operands: set[Fraction]) -> set[Fraction]:
    """Candidate differences between the wrong and the correct result of a step.

    A slip keeps the look of the true result: 14400 can become 14300 but not 14401,
    which anyone would spot as wrong. Small or non-round results get small offsets.
    """
    base = set(range(1, 11)) | {12, 15, 20, 25, 50, 100}
    if r.denominator != 1:
        return {Fraction(k) for k in base} | {Fraction(1, 2)}
    digits = str(r.numerator)
    zeros = len(digits) - len(digits.rstrip("0"))
    scale = 10 ** min(zeros, max(len(digits) - 2, 0))
    offsets = {Fraction(k * scale) for k in base}
    offsets |= {o for o in operands if o.denominator == 1 and 0 < o <= r and o % scale == 0}
    return offsets


def _plausible(wrong: Fraction, gold: Fraction) -> bool:
    return (
        wrong.denominator == 1 and wrong > 0 and wrong != gold
        and (gold / 3 <= wrong <= gold * 3 or abs(wrong - gold) <= 10)
    )


def plan_errors(question: str, rationale: str, gold: str, seed: int, max_plans: int = 3,
                early_only: bool = True) -> list[ErrorPlan]:
    """Candidate error plans, earliest step first. Empty if none is possible.

    With early_only, the corrupted step must be in the first half of the solution.
    """
    steps = rationale_steps(rationale)
    gold_v = Fraction(gold)
    finals = [k for k, s in enumerate(steps) if s.rhs_value == gold_v]
    if not finals:
        return []
    last = finals[-1]
    rng = random.Random(seed)
    plans = []
    for i in range(last // 2 + 1 if early_only else last + 1):
        r = steps[i].rhs_value
        if r <= 0:
            continue
        operands = {Fraction(t) for t in _NUM_TOKEN.findall(steps[i].lhs)}
        valid = []
        offsets = slip_offsets(r, operands)
        for k in sorted(offsets):
            for new in (r + k, r - k):
                if new <= 0:
                    continue
                res = propagate(steps, i, new, question)
                if res is None:
                    continue
                calcs, changes = res
                wrong = changes.get(gold_v)
                if wrong is None or not _plausible(wrong, gold_v):
                    continue
                valid.append((abs(k), new, calcs, wrong))
        if not valid:
            continue
        # Prefer small slips, with some variety among the smallest ones.
        valid.sort(key=lambda v: v[0])
        pool = valid[:4]
        rng.shuffle(pool)
        for _, new, calcs, wrong in pool:
            plans.append(ErrorPlan(
                step=i, original=str(steps[i]), corrupted=f"{steps[i].lhs} = {fmt(new)}",
                calculations=calcs[: last + 1], wrong_answer=fmt(wrong), offset=fmt(new - r),
            ))
        if len(plans) >= max_plans:
            break
    return plans[:max_plans]


# ---------------------------------------------------------------- rewriting and checks

REWRITE_SYSTEM = (
    "You write worked solutions to math word problems. You are given a problem and the "
    "exact list of calculations to use. Write a clear step-by-step solution in your usual "
    "style that uses exactly these calculations, with exactly these numbers and results, in "
    "this order. Write each calculation explicitly as an equation (for example 12 * 5 = 60). "
    "Copy every calculation exactly as given, even if one looks wrong. Do not add other "
    "calculations and do not check or comment on them.\n"
    "End your response with a last line of the form:\nFinal answer: <number>"
)


def rewrite_messages(question: str, plan: ErrorPlan) -> list[dict]:
    calcs = "\n".join(f"{k}. {c}" for k, c in enumerate(plan.calculations, 1))
    user = (f"Problem: {question}\n\nCalculations to use:\n{calcs}\n\n"
            f"Write the solution. The final answer is {plan.wrong_answer}.")
    return [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}]


_FORBIDDEN = re.compile(
    r"\b(mistakes?|errors?|incorrect(ly)?|wrong|typo|actually|oops|correction|should be|"
    r"(given|provided) calculations?|as instructed|calculations to use)\b|<<|####|\?\?",
    re.IGNORECASE,
)
_MARKER = re.compile(r"final\s+answer", re.IGNORECASE)


def finalize(text: str, wrong: str) -> str:
    """Cut anything after the last 'Final answer' line and write that line canonically."""
    lines = text.strip().splitlines()
    idx = max(k for k, ln in enumerate(lines) if _MARKER.search(ln))
    body = "\n".join(lines[:idx]).rstrip()
    return f"{body}\n\nFinal answer: {wrong}"


def check_fault_text(text: str, plan: ErrorPlan, gold: str) -> list[str]:
    """Problems with a generated faulty solution (empty list = accepted)."""
    problems = []
    if parse_final_answer(text) != plan.wrong_answer:
        problems.append(f"final answer {parse_final_answer(text)} != {plan.wrong_answer}")
    if len(_MARKER.findall(text)) != 1:
        problems.append("'Final answer' must appear exactly once")
    if m := _FORBIDDEN.search(text):
        problems.append(f"forbidden wording {m.group(0)!r}")
    eqs = extract_equations(text)
    wrong_eqs = [e for e in eqs if not e.correct]
    injected = plan.corrupted.split(" = ")[1]
    if len(wrong_eqs) != 1:
        problems.append(f"{len(wrong_eqs)} wrong equations (need exactly 1): {[str(e) for e in wrong_eqs]}")
    elif Fraction(wrong_eqs[0].rhs) != Fraction(injected):
        problems.append(f"the wrong equation {wrong_eqs[0]} is not the injected one ({plan.corrupted})")
    if any(e.correct and e.rhs_value == Fraction(gold) and Fraction(gold) not in
           {Fraction(c.split(" = ")[1]) for c in plan.calculations} for e in eqs):
        problems.append("the correct answer is computed somewhere")
    # Every planned calculation must be written out (in any notation), so no step is skipped.
    written = {(e.lhs_value, e.rhs_value) for e in eqs}
    for c in plan.calculations:
        lhs, rhs = c.split(" = ")
        if (safe_eval(lhs), Fraction(rhs)) not in written:
            problems.append(f"missing calculation {c}")
            break
    return problems


def make_persuasive(neutral: str) -> str:
    """Neutral text with the confidence sentence inserted just before the final line."""
    body, sep, last = neutral.rpartition("\n")
    return f"{body}\n{PERSUASIVE_SENTENCE}{sep}{last}"


# ---------------------------------------------------------------- faults.json

def messages_sha(doc: dict) -> str:
    """Hash of what the faulty agent says (texts and injected answers) and nothing else, so
    that adding metadata such as covariates to faults.json does not change it. The
    persuasive texts are derived from the neutral ones, so they are covered too."""
    payload = {qid: [f["text"], f["wrong_answer"]] for qid, f in sorted(doc["faults"].items())}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[:12]


# The file is frozen once generated, so it is read and hashed once per process.
# Functions take the path at call time so tests can point FAULTS_PATH elsewhere.
_cache: dict[Path, tuple[dict, str]] = {}


def _load(path: Path | None) -> tuple[dict, str]:
    path = path or FAULTS_PATH
    if path not in _cache:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found, run `python -m scripts.make_faults` first")
        doc = json.loads(path.read_bytes())
        _cache[path] = (doc, messages_sha(doc))
    return _cache[path]


def load_faults(path: Path | None = None) -> dict:
    return _load(path)[0]


def faults_sha(path: Path | None = None) -> str | None:
    """Hash of the fault messages, stored in the logs; None if faults.json does not exist yet."""
    try:
        return _load(path)[1]
    except FileNotFoundError:
        return None


def covariates(plan: dict, gold: str) -> dict:
    """Where the slip is and how large it is, for secondary analyses.

    position: step of the slip / number of steps (1/4 = first of four steps; a one-step
    problem gives 1.0); early if <= 0.5. magnitude: relative error of the corrupted step
    and of the final answer.
    """
    relative = (plan["step"] + 1) / len(plan["calculations"])
    true_v = Fraction(plan["original"].split(" = ")[1])
    wrong_v = Fraction(plan["corrupted"].split(" = ")[1])
    gold_v = Fraction(gold)
    return {
        "slip_position": {"relative": round(relative, 4), "category": "early" if relative <= 0.5 else "late"},
        "slip_magnitude": {
            "step": round(float(abs(wrong_v - true_v) / true_v), 4),
            "answer": round(float(abs(Fraction(plan["wrong_answer"]) - gold_v) / gold_v), 4),
        },
    }


def fault_text(question_id: str, style: str, path: Path | None = None) -> tuple[str, str]:
    """(message text, injected wrong answer) for a question and a style."""
    f = load_faults(path)["faults"][question_id]
    if style == "neutral":
        return f["text"], f["wrong_answer"]
    if style == "persuasive":
        return make_persuasive(f["text"]), f["wrong_answer"]
    raise ValueError(f"unknown fault style {style!r}")


def plan_to_dict(plan: ErrorPlan) -> dict:
    return asdict(plan)


def faulty_message(question_id: str, style: str, agent_id: str) -> Message:
    """The scripted faulty solver: no LLM call, it returns the stored wrong solution."""
    text, _ = fault_text(question_id, style)
    return Message(agent_id=agent_id, role="solver", content=text,
                   parsed_answer=parse_final_answer(text), is_faulty=True)
