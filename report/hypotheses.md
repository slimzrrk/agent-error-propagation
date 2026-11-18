# Pre-registered confirmatory hypotheses

This file and `scripts/confirmatory.py` are committed **before any run on the test split**
(the git history gives the date). The test split is run once, with the protocol of the
commit that contains this file. Dev results (50 questions) were used to design the study
and to formulate these hypotheses; they are not used as evidence.

## Data and design

- Test split: 200 GSM8K test questions (`data/splits.json`, seed 42).
- Faults: `data/faults.json`, fault messages hash `6a2fc4273a88` (texts and injected answers).
  *Superseded by amendment 1: `3655c7b6f668`.*
- Model: `qwen2.5:1.5b-instruct-fp16` (Ollama), temperature 0.7, seed per (question, agent).
- Configurations (all without defense): `chain-{neutral,persuasive}-p{1,2,3}-none`,
  `chain_rotated-{neutral,persuasive}-p{1,2,3}-none`, `star-{neutral,persuasive}-p1-none`.

## Outcome, unit and test

- **Outcome: team adoption**, i.e. the team's final answer (aggregator) is the injected
  wrong answer, with the strict parser.
- **Unit of analysis: the question.** For each hypothesis we compute one difference per
  question, averaged over the two fault styles unless stated otherwise.
- **Test:** one-sided (H2 and H4: two-sided since amendment 1) sign-flip permutation test on the 200 per-question differences
  (H0: mean difference = 0; 100,000 permutations, seed 0). Effect size: mean difference
  with a 95% bootstrap confidence interval over questions.
- **Multiplicity:** Holm correction over H1–H4, family-wise alpha = 0.05.
  *Amendment 2: over H1–H7. Amendment 3: over H1–H8.*
- Code: `python -m scripts.confirmatory results/test/<log>.jsonl`.

## Hypotheses

**H1 — Position in the chain.** In the chronological chain, team adoption is higher when
the faulty solver is at position 1 than at positions 2 and 3.
Per question: mean over styles of adoption(p1) minus mean over styles and p2, p3 of
adoption(p). Alternative: difference > 0.

**H2 — Persuasion.** A persuasive faulty solver is adopted more than a neutral one.
Per question: mean over the four cells (chain p1, p2, p3 and star p1) of
adoption(persuasive) minus adoption(neutral). Alternative: difference > 0.
*Amendment 1: two-sided, the alternative is difference ≠ 0.*

**H3 — Primacy of the aggregator.** Team adoption is higher when the faulty solution is
shown first to the aggregator than when it is shown second or third. Tested on
`chain_rotated`, where for each question the faulty solver at p1, p2, p3 is shown at
three different ranks (Latin square), so the chain position is balanced across ranks.
Per question and style: adoption at rank 1 minus mean adoption at ranks 2 and 3; then
mean over styles. Alternative: difference > 0.

**H4 — Topology.** The chain adopts the error more than the star.
Per question: mean adoption over the six chronological chain fault configs minus mean
adoption over the two star fault configs. Alternative: difference > 0.
*Amendment 1: two-sided, the alternative is difference ≠ 0.*

## Exploratory analyses (no correction, reported as exploratory)

Recovery (final answer still correct); adoption by each downstream agent and the
coincidence baseline of independent agents; persuasion within each cell; star results
by presentation slot; position effect within `chain_rotated`; early vs late slips and
slip magnitude; results without the faults flagged `obvious`; lenient parser; LLM
aggregator vs counterfactual majority vote; natural propagation without faults;
accuracy and cost against the single agent.

## Defenses

Hypotheses about the defenses (research question 3) will be added to this file as a dated
amendment, after the defense design is fixed on dev and before any run on the test split.
*Done in amendment 2 (H5–H7).*

## Amendment 1 (2026-11-18, before any run on the test split)

1. **H2 and H4 become two-sided.** On dev, the observed effects go the other way from the
   original one-sided alternatives (persuasive minus neutral: −0.02; chain minus star:
   −0.04). A one-sided test could then only ever fail, so the honest question is whether
   there is a difference at all. H2 and H4 are now tested two-sided (alternative:
   difference ≠ 0), with the same statistics, the same per-question differences and the
   same Holm family. H1 and H3 stay one-sided.
2. **Fault style checks (faults revision 3).** Some faulty solutions copied several
   sentences of the problem word for word or labelled their steps "Calculation N:", which
   honest solvers almost never do and which could give the faulty agent away. Two checks
   were added to the generator (at most one copied sentence of four words or more, no
   "Calculation N:" labels) and the 51 failing faults were rewritten, keeping their error
   plan when possible (43 same plan, 7 new plan); every other text is unchanged. One test
   fault (`test-198`), whose steps follow the problem sentence by sentence, could not be
   rewritten and is kept, flagged `style_check_failed`. The fault messages hash becomes
   **`3655c7b6f668`**; the test run must use it.

## Amendment 2 (2026-11-18, after the defenses were fixed on dev, before any run on the test split)

Three hypotheses are added; the confirmatory family becomes **H1–H7**, with the Holm
correction over all seven (family-wise alpha = 0.05). Outcome, unit of analysis, test
(sign-flip permutation, 100,000 permutations, seed 0) and effect size are unchanged.

- Defense prompts: `src/prompts.py` at the commit of this amendment (prompts hash
  `7cad7f5851d8`), tuned on dev as documented in `report/defense_tuning.md`. Protocol
  hash of the test run (prompts, parser, agents, topologies, faults, defenses, splits):
  `58917055adcf`.
- The **eight fault cells** of the main design: `chain-{neutral,persuasive}-p{1,2,3}-<d>`
  and `star-{neutral,persuasive}-p1-<d>`, for each defense `<d>` in none, skeptical,
  verifier.

**H5 — Verifier vs no defense.** The verifier lowers team adoption. Per question: mean
over the eight cells of adoption(none) minus adoption(verifier). One-sided, alternative:
difference > 0.

**H6 — Verifier vs skeptical instruction.** Their team adoption differs. Per question:
mean over the eight cells of adoption(skeptical) minus adoption(verifier). Two-sided,
alternative: difference ≠ 0.

**H7 — Majority vote vs LLM aggregator.** Without defense, a majority vote over the three
solver answers adopts the error less often than the LLM aggregator of the same runs. The
majority answer is the answer given by at least two of the three solvers (the faulty one
included); with three different answers there is no majority, which counts as not
adopted. Per question: mean over the eight no-defense cells of adoption(aggregator) minus
adoption(majority vote). One-sided, alternative: difference > 0. The `chain_rotated`
configs are not included (they are used by H3).

Everything else stays exploratory, in particular: the skeptical instruction against no
defense; accuracy with and without fault for each defense; cost (LLM calls, tokens); the
verifier's recall, precision and false positives (correct messages dropped), overall and
without fault; the defenses against the majority vote; and the correction of the error by
downstream solvers (in `chain_rotated`, at equal rank, adoption with the faulty solver at
p3 compared with p1).

## Amendment 3 (2026-11-18, before any run on the test split)

**H8 — Hybrid vote vs LLM aggregator.** The hybrid vote takes the strict majority of the
three solver answers when there is one (at least two equal valid answers, the faulty
solver included), and otherwise the answer of the LLM aggregator of the same run; it needs
no extra LLM call. Without defense and with a fault, the hybrid vote is more accurate than
the LLM aggregator. Outcome: **accuracy** (the final answer is the gold answer, strict
parser), not adoption. Per question: mean over the eight no-defense fault cells of
correct(hybrid vote) minus correct(aggregator). One-sided, alternative: difference > 0.
`chain_rotated` is not included. The confirmatory family becomes **H1–H8**, with the Holm
correction over all eight (family-wise alpha = 0.05).

The hybrid vote is reported in every table next to the majority vote. New exploratory
analyses: anchoring of the verifier (difflib similarity ratio between the verifier's text
and the checked message, near copies = ratio ≥ 0.9, for honest and faulty targets), and a
pooled vote (strict majority over the answers of the three solvers and the three
verifiers, i.e. at least 4 of 6, in the verifier configs).

Handling of verifier outputs without a valid final answer, fixed since the defenses were
implemented and unchanged here: there is no verdict and the checked message is **kept**.
A checked message without a final answer of its own is rejected.

**The test run.** The protocol is pinned in `report/preregistration.json` (protocol hash
`58917055adcf`, prompts `7cad7f5851d8`, faults `3655c7b6f668`). `scripts/run_experiment`
refuses to run the test split with any other protocol hash, with `--limit` or `--reuse`,
or with a config other than the full grid (`configs/grid.yaml`). The commit of this
amendment is tagged `pre-test-run`.

## Deviations

Any deviation from this plan (failed runs, reruns, changed analysis) will be reported in
the final report with its reason.
