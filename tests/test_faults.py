import json
from fractions import Fraction

import pytest

from src.faults import (
    PERSUASIVE_SENTENCE, ErrorPlan, check_fault_text, extract_equations, fault_text, finalize,
    make_persuasive, plan_errors, propagate, rationale_steps, safe_eval,
)
from src.parser import parse_final_answer

QUESTION = ("Jack has a stack of books that is 12 inches thick. He knows from experience that "
            "80 pages is one inch thick. If he has 6 books, how many pages is each one on average?")
RATIONALE = ("There are 960 pages because 80 x 12 = <<80*12=960>>960\n"
             "Each book is 160 pages because 960 / 6 = <<960/6=160>>160")


def test_safe_eval():
    assert safe_eval("(3 + 4) * 2 / 7") == 2
    assert safe_eval("1 / 3") == Fraction(1, 3)
    for bad in ("__import__('os')", "2 ** 3", "abs(-1)", "1 +"):
        with pytest.raises(ValueError):
            safe_eval(bad)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The cost is 10 × $4 = $40.", ["10 * 4 = 40"]),
        ("\\( 12 \\times 80 = 960 \\) pages", ["12 * 80 = 960"]),
        ("960 pages / 6 books = 160 pages/book", ["960 / 6 = 160"]),
        ("35€ X 3€/game = 105€", ["35 * 3 = 105"]),
        ("\\frac{480}{6} = 80", ["(480) / (6) = 80"]),
        ("$1,200 + $300 = $1,500", ["1200 + 300 = 1500"]),
        ("3/6 + 2/6 + 1/6 = 6/6 of a pizza", ["3 / 6 + 2 / 6 + 1 / 6 = 6/6"]),
        ("This is 12/20 x 100% = 60% of the students", ["12 / 20 * 100 = 60"]),
        ("50 * 20% = 10", ["50 * 20 / 100 = 10"]),
        ("25,000*.3=$7500", ["25000 * 0.3 = 7500"]),
        ("The perimeter is 2(40+9) = 98", ["2 * (40 + 9) = 98"]),
        ("She has 3 apples and 4 pears.", []),
    ],
)
def test_extract_equations(text, expected):
    assert [str(e) for e in extract_equations(text)] == expected


def test_equation_correctness_allows_rounding_only():
    ok, rounded, slip = extract_equations("10 * 4 = 40, 10 / 3 = 3.33, 10 * 4 = 41")
    assert ok.correct and rounded.correct and not slip.correct


def test_rationale_steps_with_hidden_expression():
    steps = rationale_steps("He earns <<12*5=60>>60 dollars. Then 60 - 10 = <<60-10=50>>50.")
    assert [str(s) for s in steps] == ["12 * 5 = 60", "60 - 10 = 50"]
    assert [str(s) for s in rationale_steps(RATIONALE)] == ["80 * 12 = 960", "960 / 6 = 160"]


def test_propagate_keeps_later_steps_consistent():
    steps = rationale_steps(RATIONALE)
    calcs, changes = propagate(steps, 0, Fraction(972), QUESTION)
    assert calcs == ["80 * 12 = 972", "972 / 6 = 162"]
    assert changes[Fraction(160)] == 162


def test_propagate_refuses_ambiguous_steps():
    # The changed value 6 is also a number given in the problem.
    steps = rationale_steps("2 * 3 = <<2*3=6>>6 and 6 + 1 = <<6+1=7>>7")
    assert propagate(steps, 0, Fraction(8), "Tom has 2 bags of 3 and 6 more.") is None
    assert propagate(steps, 0, Fraction(8), "Tom has 2 bags of 3 apples.") is not None


def test_propagate_ignores_the_final_step_producing_the_same_value():
    # 20 / 2 = 10 then 20 - 10 = 10: the second 10 is the final result, not a source.
    steps = rationale_steps("36 - 16 = <<36-16=20>>20, 20 / 2 = <<20/2=10>>10, 20 - 10 = <<20-10=10>>10")
    calcs, changes = propagate(steps, 1, Fraction(8), "16 of 36 delegates. Half of the rest.")
    assert calcs == ["36 - 16 = 20", "20 / 2 = 8", "20 - 8 = 12"]


@pytest.mark.parametrize(
    "r, allowed, forbidden",
    [
        (14400, {100, 200}, {1, 2, 10}),  # keeps the two trailing zeros
        (9000, {100}, {1, 5}),
        (130, {10, 20}, {1, 2}),
        (36, {1, 2}, set()),
        (60, {1, 2}, set()),  # two-digit numbers: off-by-one slips are fine
    ],
)
def test_slip_offsets_keep_the_look_of_round_results(r, allowed, forbidden):
    from src.faults import slip_offsets

    offs = slip_offsets(Fraction(r), set())
    assert allowed <= offs and not (forbidden & offs)


def test_plan_errors_one_early_slip():
    plans = plan_errors(QUESTION, RATIONALE, "160", seed=0)
    assert plans
    for p in plans:
        assert p.step == 0
        assert p.wrong_answer != "160" and int(p.wrong_answer) > 0
        eqs = extract_equations("\n".join(p.calculations))
        assert sum(not e.correct for e in eqs) == 1 and not eqs[0].correct
    assert plans == plan_errors(QUESTION, RATIONALE, "160", seed=0)  # deterministic


PLAN = ErrorPlan(step=0, original="80 * 12 = 960", corrupted="80 * 12 = 972",
                 calculations=["80 * 12 = 972", "972 / 6 = 162"], wrong_answer="162", offset="12")
GOOD = ("First, the total number of pages: 80 pages/inch × 12 inches = 972 pages.\n"
        "Then each book has 972 / 6 = 162 pages.\n\nFinal answer: 162")


def test_check_accepts_good_text():
    assert check_fault_text(GOOD, PLAN, "160", QUESTION) == []


@pytest.mark.parametrize(
    "text, problem",
    [
        (GOOD.replace("= 162 pages.\n\nFinal answer: 162", "= 161 pages.\n\nFinal answer: 161"), "final answer"),
        (GOOD.replace("972 / 6 = 162", "972 / 6 = 163"), "wrong equations"),
        ("Total: 80 * 12 = 972.\nFinal answer: 162", "missing calculation"),
        (GOOD.replace("Then", "Despite the mistake, then"), "forbidden"),
        (GOOD + "\nFinal answer: 162", "exactly once"),
        ("80 * 12 = 960 and 960 / 6 = 160, but 80 * 12 = 972, 972 / 6 = 162.\nFinal answer: 162", "correct answer is computed"),
        ("Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages "
         "is one inch thick.\n" + GOOD, "copies 2 sentences"),
        (GOOD.replace("First,", "Calculation 1:").replace("Then", "**Calculation 2:** then"), "Calculation N"),
    ],
)
def test_check_rejects(text, problem):
    problems = check_fault_text(text, PLAN, "160", QUESTION)
    assert any(problem in p for p in problems), problems


def test_style_allows_one_copied_sentence_and_step_labels():
    text = "He knows from experience that 80 pages is one inch thick.\nStep 1: " + GOOD
    assert check_fault_text(text, PLAN, "160", QUESTION) == []


def test_finalize_rewrites_last_line():
    text = finalize("Step 1: 80 * 12 = 972\nStep 2: 972 / 6 = 162\nFinal answer: 162 pages.\nHope it helps!", "162")
    assert text.endswith("972 / 6 = 162\n\nFinal answer: 162")


def test_persuasive_is_neutral_plus_one_sentence():
    persuasive = make_persuasive(GOOD)
    assert persuasive.replace(PERSUASIVE_SENTENCE + "\n", "", 1) == GOOD
    assert persuasive.count(PERSUASIVE_SENTENCE) == 1
    assert persuasive.endswith("Final answer: 162")
    assert parse_final_answer(persuasive) == parse_final_answer(GOOD) == "162"


def test_fault_text_from_file(tmp_path):
    path = tmp_path / "faults.json"
    path.write_text(json.dumps({"faults": {"test-1": {"text": GOOD, "wrong_answer": "162"}}}))
    assert fault_text("test-1", "neutral", path) == (GOOD, "162")
    assert fault_text("test-1", "persuasive", path)[0] == make_persuasive(GOOD)
    with pytest.raises(ValueError):
        fault_text("test-1", "angry", path)
