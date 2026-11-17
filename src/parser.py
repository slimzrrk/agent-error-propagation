"""Extract the final numeric answer from an agent message.

Agents are asked to end with "Final answer: <number>". We look at the text after the
LAST such marker and return a canonical number string, or None if there is no usable
number (counted as an invalid output).

The marker match is tolerant to case and markdown ("**Final Answer:** 42") because we
want to measure reasoning, not formatting. We do NOT fall back to "last number in the
text": a missing marker is an invalid output.
"""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

_MARKER = re.compile(r"final\s+answer\s*(?:\*\*|__)?\s*(?::|\bis\b)", re.IGNORECASE)

# Optional minus and dollar sign, a number with proper thousands separators or plain
# digits, optional decimals, optional "/ denominator". Must not be glued to a word.
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>-)?\$?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*(?P<den>\d+))?"
)


def _canonical(m: re.Match) -> str | None:
    value = Decimal(m.group("num").replace(",", ""))
    if m.group("den"):
        den = int(m.group("den"))
        if den == 0:
            return None
        frac = Fraction(value) / den
        value = Decimal(frac.numerator) / Decimal(frac.denominator)
    if m.group("sign"):
        value = -value
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def normalize_number(text: str) -> str | None:
    """Canonical form of a bare numeric string: '1,234.50' -> '1234.5', '$18.00' -> '18'."""
    text = text.strip().rstrip("%.").strip()
    m = _NUMBER.fullmatch(text)
    return _canonical(m) if m else None


def parse_final_answer(text: str | None) -> str | None:
    """Number after the last 'Final answer:' marker, or None."""
    if not text:
        return None
    markers = list(_MARKER.finditer(text))
    if not markers:
        return None
    rest = text[markers[-1].end():]

    # The answer is on the same line, or on the next non-empty one.
    line = next((ln for ln in rest.splitlines() if ln.strip()), "")
    # "Final answer: 10 - 2 = 8" -> keep what comes after the last "=".
    if "=" in line:
        line = line.rsplit("=", 1)[1]
    m = _NUMBER.search(line)
    return _canonical(m) if m else None


def last_number(text: str | None) -> str | None:
    """Last number in the text. Only used for the lenient sensitivity analysis, never by
    the protocol itself."""
    if not text:
        return None
    matches = list(_NUMBER.finditer(text))
    return _canonical(matches[-1]) if matches else None


def answers_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    return Decimal(pred) == Decimal(gold)
