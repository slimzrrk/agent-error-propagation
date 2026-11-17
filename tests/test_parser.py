import pytest

from src.parser import answers_match, normalize_number, parse_final_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Final answer: 42", "42"),
        ("Step 1...\nStep 2...\nFinal answer: 18", "18"),
        ("final answer: 7", "7"),
        ("FINAL ANSWER: 7", "7"),
        ("Final Answer: 7", "7"),
        ("**Final answer:** 72", "72"),
        ("**Final Answer**: 72", "72"),
        ("Final answer: **72**", "72"),
        ("The final answer is 15.", "15"),
        ("Final answer: $1,234", "1234"),
        ("Final answer: $18.00", "18"),
        ("Final answer: 1,000,000", "1000000"),
        ("Final answer: 50%", "50"),
        ("Final answer: 18.0", "18"),
        ("Final answer: 18.", "18"),
        ("Final answer: 12.50 dollars", "12.5"),
        ("Final answer: 0.75", "0.75"),
        ("Final answer: .5", "0.5"),
        ("Final answer: -15", "-15"),
        ("Final answer: -$5", "-5"),
        ("Final answer: 3/4", "0.75"),
        ("Final answer: 10/2", "5"),
        ("Final answer: 7 apples.", "7"),
        ("Final answer: The total cost is $1,250.", "1250"),
        ("Final answer: \\boxed{42}", "42"),
        ("Final answer:\n\n  42", "42"),
        ("Final answer: 10 - 2 = 8", "8"),
        ("Final answer: 20 / 4 = 5", "5"),
        ("Final answer: x = 5", "5"),
    ],
)
def test_parse_valid(text, expected):
    assert parse_final_answer(text) == expected


def test_uses_last_marker():
    text = "Final answer: 10\nWait, I made a mistake.\nFinal answer: 12"
    assert parse_final_answer(text) == "12"


def test_earlier_mention_does_not_hide_real_answer():
    text = "Before giving the final answer, check 5 + 3.\n5 + 3 = 8\nFinal answer: 8"
    assert parse_final_answer(text) == "8"


def test_numbers_in_reasoning_are_ignored():
    text = "She has 3 bags of 12 apples, so 36.\nFinal answer: 36"
    assert parse_final_answer(text) == "36"


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "The answer is 42",  # no marker: invalid, no fallback to last number
        "So she pays 18 dollars.",
        "Final answer:",
        "Final answer: none",
        "Final answer: I cannot determine this.",
        "Final answer: x2",  # digit glued to a word
        "Final answer: 5/0",
    ],
)
def test_parse_invalid(text):
    assert parse_final_answer(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("72", "72"),
        (" 1,000 ", "1000"),
        ("$18.00", "18"),
        ("25%", "25"),
        ("-3", "-3"),
        ("2.50", "2.5"),
        ("007", "7"),
        ("abc", None),
        ("", None),
        ("12 apples", None),
    ],
)
def test_normalize_number(text, expected):
    assert normalize_number(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Therefore, each book has on average 80 pages.", "80"),
        ("\\[ \\boxed{12} \\]", "12"),
        ("The total is $1,250.", "1250"),
        ("No numbers here", None),
        (None, None),
    ],
)
def test_last_number(text, expected):
    from src.parser import last_number

    assert last_number(text) == expected


def test_answers_match():
    assert answers_match("18", "18")
    assert answers_match("18.0", "18")
    assert not answers_match("17", "18")
    assert not answers_match(None, "18")
