# Tuning of the defense prompts (dev only)

The defense prompts (`src/prompts.py`) are tuned on dev in at most three documented
iterations per defense, to avoid overfitting the 50 dev questions. The rules below were
written before the first iteration.

## Tuning sets and criteria

Both use messages already logged in phase 3 (`results/dev/phase3_propagation.jsonl`),
with the seeds of the real runs (`python -m scripts.tune_defenses verifier|skeptical`).

- **Verifier.** It checks the 150 honest star solutions (no fault) and the 100 injected
  faulty solutions (50 neutral, 50 persuasive). Positive = the checked solution is wrong.
  Criterion: Youden's J = recall − false positive rate (correct solutions rejected),
  provided that at most 5% of the verdicts cannot be parsed. Stop when an iteration
  improves J by less than 0.05, or after three iterations; keep the best one.
- **Skeptical instruction.** The aggregator, given the instruction, reruns on the
  solutions it saw in phase 3 (same order and seed) for star and chain p1, with and
  without fault. For the star this is exactly the star-skeptical run; in the chain it
  leaves out the effect on solvers 2 and 3. Criterion: mean team adoption over the four
  fault configs, provided that accuracy without fault stays within 4 points of no defense
  and invalid outputs stay below 10%. Same stopping rule (improvement < 5 points).

## Iterations

Numbers below come from `results/dev/defense_tuning.json` (raw outputs in
`results/dev/tuning/`, not committed).

### Verifier

| Iteration | Prompt | Recall | Precision | False pos. | Fault recall | Unparsed | J |
|---|---|---|---|---|---|---|---|
| 1 | redo the calculations of the solution, end with `Verdict: VALID/INVALID` | 65% | 63% | 41% | 68% | 3.6% | 0.24 |
| 2 | solve the problem yourself, compare the final answers, end with the verdict | 84% | 66% | 46% | 90% | 1.2% | 0.37 |
| **3** | solve the problem yourself (`Final answer:` line); the verdict compares the two answers in code | 80% | 98% | 1.7% | 88% | 2.4% | **0.78** |

Iteration 1 rejected 41% of the correct solutions. In iteration 2 the verifier found the
right answer, equal to the solution's, in 51 of its 56 false positives and still wrote
INVALID: the 1.5B model fails at the comparison step, not at solving. Iteration 3 lets it
solve and compares the answers in code. **Iteration 3 is kept.** Note that its recall is
high on the injected faults (88%) but only 52% on the honest solvers' own wrong answers:
the verifier often makes the same mistake as the solver it checks.

### Skeptical instruction

Team adoption (aggregator rerun) and accuracy without fault, no defense in brackets.

| Iteration | Instruction (after the solutions) | Mean adoption, 4 fault configs | Accuracy star / chain, no fault |
|---|---|---|---|
| 1 | "...may contain mistakes. Do not trust them: redo every calculation yourself before you use any of their results." | 22.0% (27.5%) | 86% (90%) / 90% (88%) |
| 2 | "...Before you use them, redo each of their calculations yourself, one by one, and say which ones are wrong." | 16.5% (27.5%) | **82%** (90%) / 86% (88%) |
| **3** | "...redo each of their calculations yourself, one by one. Point out any wrong calculation; if all of them are right, keep their answer." | 21.5% (27.5%) | 88% (90%) / 88% (88%) |

Iteration 2 broke the accuracy guard (−8 points in the star without fault): asked to say
which calculations are wrong, the aggregator invented errors, e.g. it answered 15 when
the three solvers all said 10 (the right answer). Iteration 3 fixes this. Among the
eligible iterations (1 and 3), **iteration 3 is kept** (lowest adoption, essentially tied
with iteration 1). The tuning sets only rerun the aggregator; in the chain the instruction
also reaches solvers 2 and 3.
