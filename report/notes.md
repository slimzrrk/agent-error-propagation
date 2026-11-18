# Notes for the final report

## Limitations

- **Type of injected error.** The injected faults are calculation slips: one wrong
  arithmetic result inside otherwise correct reasoning. The natural errors that
  downstream solvers copied on dev (phase 2, e.g. `train-204`) were comprehension errors
  (a misreading of the problem). On dev, downstream solvers almost never adopted the
  injected slips (they recompute the arithmetic), whereas they copied half of the natural
  errors of solver 1 (3 of 6 cases). Our propagation rates therefore describe calculation
  errors and may underestimate the spread of conceptual errors.
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
  identical texts and 23 identical answers.
- **Scale.** One small model (Qwen2.5-1.5B-Instruct), one task (GSM8K), three agents.
  The dev split (50 questions) is only used for development; with 200 test questions,
  differences of a few points remain within the confidence intervals.
