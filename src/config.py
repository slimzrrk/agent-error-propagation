"""Experiment configuration: YAML files -> dataclasses, and the full experiment grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from src.llm import LLMConfig

TOPOLOGIES = ("single", "chain", "star")
FAULTS = ("none", "neutral", "persuasive")
DEFENSES = ("none", "skeptical", "verifier")


@dataclass(frozen=True)
class ExperimentConfig:
    topology: str
    fault: str = "none"
    position: int | None = None  # index of the faulty solver (1-based), None if no fault
    defense: str = "none"
    temperature: float | None = None  # overrides llm.temperature for this experiment
    name: str | None = None  # optional explicit id

    def __post_init__(self):
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology {self.topology!r}")
        if self.fault not in FAULTS:
            raise ValueError(f"unknown fault {self.fault!r}")
        if self.defense not in DEFENSES:
            raise ValueError(f"unknown defense {self.defense!r}")
        if (self.fault == "none") != (self.position is None):
            raise ValueError("position must be set iff a fault is injected")
        if self.position is not None and self.position not in (1, 2, 3):
            raise ValueError("position must be 1, 2 or 3")
        if self.topology == "single" and (self.fault != "none" or self.defense != "none"):
            raise ValueError("the single-agent baseline takes no fault and no defense")

    @property
    def id(self) -> str:
        if self.name:
            return self.name
        if self.topology == "single":
            parts = ["single"]
        else:
            parts = [self.topology, self.fault] + ([f"p{self.position}"] if self.position else []) + [self.defense]
        if self.temperature is not None:
            parts.append(f"t{self.temperature:g}")
        return "-".join(parts)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RunConfig:
    name: str
    llm: LLMConfig = field(default_factory=LLMConfig)
    experiments: list[ExperimentConfig] = field(default_factory=list)
    workers: int = 8
    seed: int = 42  # base for the per-(question, agent) sampling seeds

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        exps = [ExperimentConfig(**e) for e in raw.get("experiments", [])]
        ids = [e.id for e in exps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate experiment ids in {path}")
        return cls(
            name=raw.get("name", path.stem),
            llm=LLMConfig.from_dict(raw.get("llm")),
            experiments=exps,
            workers=raw.get("workers", 8),
            seed=raw.get("seed", 42),
        )


def full_grid(include_single: bool = True) -> list[ExperimentConfig]:
    """topology x fault x position x defense. Star uses position 1 only (symmetry)."""
    grid = [ExperimentConfig("single")] if include_single else []
    for topology in ("chain", "star"):
        positions = (1, 2, 3) if topology == "chain" else (1,)
        for defense in DEFENSES:
            grid.append(ExperimentConfig(topology, "none", None, defense))
            for fault in ("neutral", "persuasive"):
                for p in positions:
                    grid.append(ExperimentConfig(topology, fault, p, defense))
    return grid
