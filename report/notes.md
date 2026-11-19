# Notes for the final report

## Test run

- **Deviations: none.** The test split was run once, on 2026-11-19 (00:54–05:57 UTC,
  5.0 h in a single session; the estimate was 4.2 h), from commit `3d2d846` (tag
  `pre-test-run`) with the pre-registered protocol `58917055adcf` (prompts
  `7cad7f5851d8`, faults `3655c7b6f668`): 37 configs x 200 questions = 7,400 runs, no
  failed run, no interruption, no resume. The confirmatory analysis is the one fixed by
  amendment 3; since the tag, `scripts/confirmatory.py` and `scripts/summarize.py` only
  changed their output path so that they can read the compressed log, which gives the
  saved tables byte for byte (`results/test/grid.jsonl.gz`).
- **Confirmatory results:** H1, H3, H5, H6, H7 and H8 are supported after Holm; H2 and
  H4 are not (`results/test/grid_confirmatory.csv`). Every supported effect kept the sign
  and roughly the size of the dev dry run (H1 +22.8 points on test vs +25 on dev, H3
  +22.9 vs +18, H5 +15.8 vs +13.5, H6 +13.6 vs +11, H7 +17.6 vs +16.3, H8 +10.5 vs +13.5).

## Dev against test

- **The single agent is much weaker on test: 65% [58%, 72%] against 84% [74%, 94%] on
  dev**, with the same 6% of invalid outputs. The two sets look alike (3.2 calculation
  steps per gold rationale on both, 44 and 46 words per problem). Likely reasons: dev was
  drawn from the GSM8K *train* split, which is widely used for instruction tuning and may
  have been seen by the model (contamination); every prompt and threshold was tuned on
  dev; and 50 questions give a wide interval. Dev numbers are therefore a guide, not an
  estimate. One consequence: the teams make far more natural errors on test (59 natural
  wrong answers of solver 1 in the chain without fault, against 4-5 on dev), so the
  natural-propagation analysis only becomes informative on test.

## Exploratory analyses on test

- **Correction by downstream solvers: suggested by dev, not reproduced.** In
  chain_rotated, dev suggested that a fault at p1 is adopted less than the same fault at
  p3 *at equal presentation rank*, as if solvers 2 and 3, who read and recompute it,
  helped the aggregator reject it: p3 − p1 = +6.0 points averaged over the three ranks,
  mostly at rank 1 (38% vs 21%), with an interval that already included zero (cluster
  bootstrap [−4.6, +16.1], from the raw dev log). On test: +1.6 points [−3.9, +7.2]
  (rank 1: 36% vs 32%, rank 2: 11% vs 7%, rank 3: 13% vs 17%). Position in the chain
  acts through the presentation rank: H1 (+22.8) and H3 (+22.9) measure the same effect,
  since in the chronological chain p1 is also read first.
- **Natural against injected propagation.** In the chain without fault, when solver 1 is
  wrong on its own (59 runs), 53% [42%, 64%] of the downstream agents end with its wrong
  answer (solver 2 58%, solver 3 41%, aggregator 61%). The coincidence baseline (star
  without fault: an independent solver gives the same wrong answer as solver 1 without
  having read it) is 11% [4%, 18%], so most of the 53% is influence, not a shared
  difficulty. An injected slip at p1 is adopted by solvers 2 and 3 in only 4% and 2% of
  the runs (both styles), and by the aggregator in 34%. Solvers recompute an arithmetic
  slip but take over the model's own errors, which are mostly misreadings of the problem
  (first limitation below). Injected slips thus measure the weakness of the aggregation
  step; they understate how far a natural error travels along a chain.

## Limitations

- **Type of injected error.** The injected faults are calculation slips: one wrong
  arithmetic result inside otherwise correct reasoning. The natural errors that
  downstream solvers copied on dev (phase 2, e.g. `train-204`) were comprehension errors
  (a misreading of the problem). On test, downstream solvers adopt the injected slips in
  2-4% of the runs (they recompute the arithmetic), against 41-58% for the natural errors
  of solver 1 (see above). Our propagation rates therefore describe calculation errors
  and underestimate the spread of conceptual errors.
- **Plausibility of the faults.** Faults follow a rule fixed in advance (true result
  >= 10, slip of 5-30%, trailing zeros kept). 25 of 250 tiny problems cannot satisfy it
  (7 in dev, 18 in test); their faults are kept and flagged `obvious`, and results are
  also reported without them. The rule has a cost: since early steps often have results
  below 10, the slip is late in the reasoning for 58 of 250 faults (one-step problems,
  or early steps that are too small or do not affect the answer); `slip_position`
  allows a breakdown.
- **Style of the faults.** A faulty text must not copy more than one sentence of the
  problem word for word nor label its steps "Calculation N:" (faults revision 3: 51 texts
  rewritten, 6 in dev). The check is slightly stricter than the honest style: 19 of 551
  distinct honest solver messages on dev (3%) copy two sentences or more, mostly short
  clauses restated as reasoning steps, and 2 use such labels. One test fault (`test-198`)
  could not pass the checks and is flagged `style_check_failed`.
- **Presentation order.** The LLM aggregator has a strong primacy bias. In the
  chronological chain, the position of the faulty solver and its rank in the aggregator
  prompt are confounded; the `chain_rotated` variant is there to separate them.
- **Reproducibility of sampling.** Seeds make most outputs identical across configs, but
  batched inference is not bit-for-bit deterministic: with the same prompt and seed, 2-10%
  of the texts drift after a while (1-2% of the answers change). Paired tests stay valid
  (they pair by question); the pairing is just a little looser than exact common random
  numbers. The dev rerun after faults revision 3 copied the 766 runs with unchanged
  inputs and redid the 84 others; a sample of 24 copied runs rerun as a check gave 19
  identical texts and 23 identical answers. Between two sessions (the 550 no-defense
  runs of phases 3 and 4 on dev), 75% of the runs are identical and 94% end with the same
  team answer: differences of a few points between configs on dev are within this noise.
- **Defenses.** The verifier is strong against the injected slips (on test it rejects
  83% of them, and 2% of correct messages) because it solves the problem again: its
  errors are correlated with the solvers' own errors (recall 30% on their natural wrong
  answers without fault, 48% on dev), and its verdict is an answer comparison made in code, not a judgement
  of the model (the model could not compare answers reliably, see
  `report/defense_tuning.md`). It doubles the number of calls. The skeptical instruction
  was tuned on aggregator reruns only, and the majority vote, which costs nothing, is
  computed on the same solver answers: both are counterfactual or proxy comparisons, not
  separate runs.
- **Scale.** One small model (Qwen2.5-1.5B-Instruct), one task (GSM8K), three agents.
  The dev split (50 questions) is only used for development; with 200 test questions,
  differences of a few points remain within the confidence intervals.
- **Counterfactual votes.** The majority and hybrid votes are computed on the solver
  answers of the no-defense runs, not run as separate systems. Their answers are exact
  (they only depend on the solvers' final answers); the hybrid cost assumes that the
  aggregator is called only when there is no majority.

## For the report

- **Describe the verifier** as a verifier that solves the problem again, with a verdict
  made by programmatic comparison of its final answer with the checked message's.
- **Present tuning iterations 1-2 as a result**, not only as a design step: asked to judge
  a solution (iteration 1) or to compare its own answer with the solution's (iteration 2),
  the 1.5B model rejects 41-46% of the correct solutions; in iteration 2 it had found the
  same answer as the solution in 51 of its 56 false positives. The model can recompute
  but cannot compare. Details in `report/defense_tuning.md`.
- **Unparsed verifier outputs** give no verdict and the message is kept: 4.3% of the
  verdicts on test (256 of 6,000; 2.7% on dev). 161 of these 256 kept messages were
  wrong (81 injected faults, 80 honest wrong answers): a fail-closed rule would have
  stopped them, at the cost of 95 correct messages. Exploratory only; the rule was fixed
  with the defenses.
- **Verifier anchoring** (test): 28% of the verifier texts are near copies (difflib
  ratio >= 0.9) of the honest solution they check, 7% for the injected faults (dev: 29%
  and 11%). On honest wrong answers, recall is 11% for near copies against 38% otherwise
  (dev: 34% and 52%); on the faults, 64% against 85%.
- **Hybrid vote** (H8): on test, without defense and with a fault, accuracy 71% vs 60%
  (chain) and 68% vs 58% (star) for the LLM aggregator, adoption 9% vs 19% and 13% vs
  21%, and no loss without fault (72% vs 71%, 77% vs 72%), at no extra call. It is as
  accurate as the verifier in the star (68% vs 69%) and more in the chain (71% vs 67%),
  with about a third of the verifier's tokens; the verifier keeps the lowest adoption
  (4% and 5%). Dev: 86% vs 72% (chain), 80% vs 66% (star).
- **Pooled vote** (test, verifier configs with a fault): 62% accuracy, between the team
  (67%) and the solvers' majority (60%); dev: 80%, 84%, 74%.
