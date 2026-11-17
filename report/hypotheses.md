# Pre-registered confirmatory hypotheses

This file and `scripts/confirmatory.py` are committed **before any run on the test split**
(the git history gives the date). The test split is run once, with the protocol of the
commit that contains this file. Dev results (50 questions) were used to design the study
and to formulate these hypotheses; they are not used as evidence.

## Data and design

- Test split: 200 GSM8K test questions (`data/splits.json`, seed 42).
- Faults: `data/faults.json`, fault messages hash `6a2fc4273a88` (texts and injected answers).
- Model: `qwen2.5:1.5b-instruct-fp16` (Ollama), temperature 0.7, seed per (question, agent).
- Configurations (all without defense): `chain-{neutral,persuasive}-p{1,2,3}-none`,
  `chain_rotated-{neutral,persuasive}-p{1,2,3}-none`, `star-{neutral,persuasive}-p1-none`.

## Outcome, unit and test

- **Outcome: team adoption**, i.e. the team's final answer (aggregator) is the injected
  wrong answer, with the strict parser.
- **Unit of analysis: the question.** For each hypothesis we compute one difference per
  question, averaged over the two fault styles unless stated otherwise.
- **Test:** one-sided sign-flip permutation test on the 200 per-question differences
  (H0: mean difference = 0; 100,000 permutations, seed 0). Effect size: mean difference
  with a 95% bootstrap confidence interval over questions.
- **Multiplicity:** Holm correction over H1–H4, family-wise alpha = 0.05.
- Code: `python -m scripts.confirmatory results/test/<log>.jsonl`.

## Hypotheses

**H1 — Position in the chain.** In the chronological chain, team adoption is higher when
the faulty solver is at position 1 than at positions 2 and 3.
Per question: mean over styles of adoption(p1) minus mean over styles and p2, p3 of
adoption(p). Alternative: difference > 0.

**H2 — Persuasion.** A persuasive faulty solver is adopted more than a neutral one.
Per question: mean over the four cells (chain p1, p2, p3 and star p1) of
adoption(persuasive) minus adoption(neutral). Alternative: difference > 0.

**H3 — Primacy of the aggregator.** Team adoption is higher when the faulty solution is
shown first to the aggregator than when it is shown second or third. Tested on
`chain_rotated`, where for each question the faulty solver at p1, p2, p3 is shown at
three different ranks (Latin square), so the chain position is balanced across ranks.
Per question and style: adoption at rank 1 minus mean adoption at ranks 2 and 3; then
mean over styles. Alternative: difference > 0.

**H4 — Topology.** The chain adopts the error more than the star.
Per question: mean adoption over the six chronological chain fault configs minus mean
adoption over the two star fault configs. Alternative: difference > 0.

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

## Deviations

Any deviation from this plan (failed runs, reruns, changed analysis) will be reported in
the final report with its reason.
