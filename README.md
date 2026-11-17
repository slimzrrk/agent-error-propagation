# When One Agent Goes Wrong

**Error Propagation and Containment in Small Multi-Agent LLM Systems**

Course project — *Machine Learning Techniques and Applications*, Université Lyon 1.

In a multi-agent LLM system, a single wrong agent can lead the whole team to adopt its
wrong answer. We inject a scripted faulty agent into small teams of LLM solvers working on
GSM8K math problems, measure how far the error spreads, and test whether simple defenses
contain it.

## Research questions

1. **Propagation.** When one agent gives a wrong solution, how often do the other agents
   adopt it in a *chain* team vs a *star* (supervisor) team? Does the position of the
   faulty agent matter?
2. **Persuasion.** Does a confident, *persuasive* faulty agent spread errors more than a
   *neutral* one making the same mistake?
3. **Containment.** Can simple defenses stop the spread? Is a dedicated *verifier* agent
   more effective than a *skeptical* instruction given to all agents, and at what cost?

## Setup

Everything runs locally (developed on an Apple M4 Pro, 24 GB).

```bash
python3.13 -m venv .venv            # Python 3.11+
source .venv/bin/activate
pip install -r requirements.txt
```

The model is `Qwen2.5-1.5B-Instruct` (fp16) served by [Ollama](https://ollama.com) and
accessed only through its OpenAI-compatible API:

```bash
ollama pull qwen2.5:1.5b-instruct-fp16
# allow 8 requests to be decoded in parallel (macOS app: set it, then restart Ollama)
launchctl setenv OLLAMA_NUM_PARALLEL 8
# or, without the app:
OLLAMA_NUM_PARALLEL=8 ollama serve
```

Any OpenAI-compatible server works (e.g. `mlx_lm.server`, vLLM): change `llm.base_url`
and `llm.model` in the config file, nothing else.

## Usage

```bash
pytest                                                    # unit tests (no server needed)
python -m src.data                                        # rebuild data/splits.json
python -m scripts.run_experiment --config configs/phase2_nofault.yaml --split dev
python -m scripts.summarize results/dev/phase2_nofault.jsonl --save   # tables + csv/json
python -m scripts.show_run results/dev/phase2_nofault.jsonl --config chain-none-none --question train-204
python -m scripts.make_grid --estimate-from results/dev/phase2_nofault.jsonl  # configs/grid.yaml + time
python -m scripts.benchmark_speed --levels 1 4 8          # raw throughput
```

Runs are resumable: rerunning a command skips the (config, question) pairs already in the
output file (`results/<split>/<config name>.jsonl`).

## Data

- GSM8K (`openai/gsm8k`, config `main`).
- **test**: 200 questions sampled from the test split with seed 42.
- **dev**: 50 questions sampled from the train split with seed 42.
- The selected ids are in `data/splits.json`. Gold answer = number after `####`.
- All prompt and threshold tuning is done on dev; the test set is run once, at the end.

## Method

*To be completed as the project progresses.*

- **Agents** — Solver, Aggregator, Verifier: same model, different system prompts.
  Every agent ends with `Final answer: <number>`, parsed by `src/parser.py`. Without any
  defense, prompts are neutral: they neither encourage nor discourage trusting others.
- **Topologies** — *chain*: Solver1 → Solver2 → Solver3, each sees the problem and the
  previous solution and may keep or revise it; the aggregator then sees the three
  solutions in chain order. *Star*: three independent solvers; the aggregator sees them in
  a balanced rotating order: the six possible orders are assigned in turn along a seeded
  ranking of the questions, so each solver is shown at each rank a third of the time.
  *Chain_rotated* (faults only, no defense): exactly the same chain of solvers, but the
  aggregator sees the solutions in the star's rotating order. For a given question the
  faulty solver at p1, p2, p3 is then shown at three different ranks (a Latin square),
  which separates the effect of the position in the chain from the effect of the
  presentation rank. The aggregator prompt is identical in all topologies and solutions
  are labelled 1, 2, 3 only. A *single* solver (same seed as Solver1) is the reference.
- **Natural propagation** — even without injected faults, we measure how often agents
  downstream of a solver that is wrong on its own end with the same wrong answer.
  Independent agents (e.g. the other star solvers) give the coincidence baseline.
- **Fault injection** — the faulty agent is not an LLM: a scripted solver replaces the
  solver at position *p* and returns a pre-computed wrong solution from `data/faults.json`
  (`python -m scripts.make_faults`). Each wrong solution contains exactly one calculation
  slip, as early as possible: the calculation steps of the gold rationale are extracted,
  one result is changed, and every later step is recomputed with it. **Plausibility
  rule** (fixed before looking at any result): the corrupted step must have a true result
  ≥ 10, the slip must change it by 5% to 30%, and it keeps its trailing zeros (14400 can
  become 14500, never 14401); the earliest step that allows such a slip is used. When no
  step allows it (tiny problems), the fault is kept but flagged `obvious: true`. The
  final wrong answer is a positive integer different from the gold answer.
  The model then writes the solution in its usual style from these calculations, and a
  text is kept only if automatic checks pass (final answer = target, exactly one wrong
  equation which is the injected one, every planned calculation present, no hint of a
  mistake). The *persuasive* style is the same text plus one sentence ("I am completely
  sure, I double-checked every step."). In the star, the faulty solution is shown to the
  aggregator in each slot exactly a third of the time (rotation over the six orders).
  The hash of the fault messages (texts and injected answers) is stored in every log.
  Each fault also records two covariates for secondary analyses: `slip_position` (step
  of the slip / number of steps, *early* if ≤ 0.5) and `slip_magnitude` (relative error
  of the corrupted step and of the final answer). When the calculations of a rationale
  cannot be extracted, the steps come from a correct solution of the model, or, for one
  question (`data/fault_steps_manual.json`), are written by hand from the rationale.
  History: the first version of `faults.json` used small offsets without the rule above;
  the faults breaking it were then regenerated with `python -m scripts.make_faults
  --fix-implausible`, every other text unchanged (listed under `revisions` in the file).
- **Defenses** — none, skeptical instruction, verifier agent.
- **Metrics** — team accuracy, invalid-output rate, error adoption rate (downstream agents
  giving the injected answer, against independent agents as a coincidence baseline),
  recovery rate (final answer still correct), verifier precision/recall, cost (calls and
  tokens), with bootstrap 95% CIs (resampling questions) and exact McNemar tests for
  paired comparisons. Main results use the strict parser; a *lenient* variant (an output
  without a valid "Final answer" line falls back to its last number) is reported as a
  robustness check. For every team config we also compute the counterfactual answer of a
  *majority vote* over the same solver answers (no extra LLM call) and compare it with
  the LLM aggregator.
- **Pre-registration** — the confirmatory hypotheses, their tests and the Holm
  correction are fixed in `report/hypotheses.md` and `scripts/confirmatory.py`, committed
  before any run on the test split. Everything else is reported as exploratory.

## Reproducibility

- Solvers sample at temperature 0.7 with a deterministic seed per (question, agent),
  independent of the experiment config, so the same agent draws the same sample across
  configs (paired comparisons).
- Every results file records the model tag and digest, temperature, seeds, server
  version, package versions, git commit, and a protocol hash (prompts, parser, agents,
  topologies, faults and splits): a log can only be resumed with the same protocol.
- Raw JSONL logs are not committed (size); aggregated summaries are.

## Repository layout

```
configs/     experiment configs (YAML)
data/        splits.json, faults.json
src/         llm, data, parser, prompts, agents, topologies, faults, defenses, metrics, logging
scripts/     run_experiment, make_grid, benchmark_speed
notebooks/   analysis and demo notebooks
results/     run logs and summaries
tests/       unit tests
report/      final report
```

## Results

*Coming in later phases.*
