import pytest

from src.config import ExperimentConfig, RunConfig, full_grid


def test_experiment_ids():
    assert ExperimentConfig("single").id == "single"
    assert ExperimentConfig("single", temperature=0.0).id == "single-t0"
    assert ExperimentConfig("chain", "neutral", 2, "verifier").id == "chain-neutral-p2-verifier"
    assert ExperimentConfig("star", "none", None, "skeptical").id == "star-none-skeptical"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(topology="ring"),
        dict(topology="chain", fault="neutral"),  # fault without position
        dict(topology="chain", position=2),  # position without fault
        dict(topology="chain", fault="neutral", position=4),
        dict(topology="single", defense="verifier"),
    ],
)
def test_invalid_experiments(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_full_grid_size():
    grid = full_grid(include_single=False)
    # chain: 3 defenses x (1 no-fault + 2 styles x 3 positions) = 21; star: 3 x (1 + 2) = 9
    assert len(grid) == 30
    assert len({e.id for e in grid}) == 30
    assert all(e.position in (None, 1) for e in grid if e.topology == "star")


def test_expected_llm_calls():
    from src.topologies import expected_llm_calls

    assert expected_llm_calls(ExperimentConfig("single")) == 1
    assert expected_llm_calls(ExperimentConfig("star")) == 4
    assert expected_llm_calls(ExperimentConfig("chain", "neutral", 2)) == 3  # faulty solver is scripted
    assert expected_llm_calls(ExperimentConfig("chain", "none", None, "verifier")) == 7
    assert expected_llm_calls(ExperimentConfig("star", "persuasive", 1, "verifier")) == 6


def test_run_config_from_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: demo\nworkers: 4\nllm:\n  temperature: 0.5\n"
        "experiments:\n  - topology: single\n  - {topology: chain, fault: persuasive, position: 3}\n"
    )
    cfg = RunConfig.from_yaml(path)
    assert cfg.name == "demo" and cfg.workers == 4 and cfg.seed == 42
    assert cfg.llm.temperature == 0.5
    assert [e.id for e in cfg.experiments] == ["single", "chain-persuasive-p3-none"]
