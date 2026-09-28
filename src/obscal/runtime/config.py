"""Typed binding of the YAML configuration tree, with per-run inheritance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import yaml

RUN_DIR = ("configs", "run")

RELEASE_SLUG = "obscal"
"""The release's own name, used wherever an artefact records which tree it describes.

Deriving it from the checkout directory would make the shipped bytes depend on
what the directory happens to be called, so a clone under a different name could
not reproduce them. The slug is a constant of the release instead.
"""


def repo_root(anchor: Path | None = None) -> Path:
    """Repository root, resolved from a config file inside ``configs/run/``."""
    if anchor is None:
        return Path(__file__).resolve().parents[3]
    return Path(anchor).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must hold a mapping at the top level")
    return cast("dict[str, Any]", payload)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge(
                cast("Mapping[str, Any]", current), cast("Mapping[str, Any]", value)
            )
        else:
            merged[key] = value
    return merged


def load_run_payload(path: Path) -> dict[str, Any]:
    """Resolve an ``inherit:`` chain into one flat payload."""
    payload = _load_yaml(path)
    parent = payload.pop("inherit", None)
    if parent is None:
        return payload
    parent_path = path.parent / str(parent)
    return _merge(load_run_payload(parent_path), payload)


@dataclass(frozen=True)
class FeatureToggles:
    """Which components of the delivered platform a run keeps in place."""

    multi_frequency_front_end: bool = True
    substrate_integrated_electrodes: bool = True
    coverage_covariate: bool = True
    split_chip_replication: bool = True
    decoupling_ladders: bool = True
    artefact_confound_channel: bool = True
    treatment_stratified_calibration: bool = True
    errors_in_variables_correction: bool = True
    d_optimal_allocation: bool = True
    two_axis_state: bool = True
    damping: bool = True
    commanded_flow_step_excitation: bool = True
    perfusion: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureToggles:
        known = {
            name: bool(value) for name, value in payload.items() if name in cls.__annotations__
        }
        return cls(**known)

    def removed(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self.__annotations__ if not getattr(self, name)))


@dataclass(frozen=True)
class AnalysisSpec:
    """One named analysis a run performs."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChipConfig:
    """Device, culture and batching constants (Sec. Device, cultures and injury model)."""

    tissue_channel: Mapping[str, Any]
    perfusion_channel: Mapping[str, Any]
    electrodes: Mapping[str, Any]
    gates: Mapping[str, Any]
    held_out_geometry: Mapping[str, Any]
    culture: Mapping[str, Any]
    batching: Mapping[str, Any]
    perfusion_contrast: Mapping[str, Any]

    @classmethod
    def load(cls, root: Path) -> ChipConfig:
        device = _load_yaml(root / "configs" / "chip" / "device.yaml")
        culture = _load_yaml(root / "configs" / "chip" / "culture.yaml")
        return cls(
            tissue_channel=cast("Mapping[str, Any]", device["device"]["tissue_channel"]),
            perfusion_channel=cast("Mapping[str, Any]", device["device"]["perfusion_channel"]),
            electrodes=cast("Mapping[str, Any]", device["device"]["electrodes"]),
            gates=cast("Mapping[str, Any]", device["gates"]),
            held_out_geometry=cast("Mapping[str, Any]", device["device"]["held_out_geometry"]),
            culture=cast("Mapping[str, Any]", culture["culture"]),
            batching=cast("Mapping[str, Any]", culture["batching"]),
            perfusion_contrast=cast("Mapping[str, Any]", device["perfusion_contrast"]),
        )


@dataclass(frozen=True)
class FrontendConfig:
    """Sweep grid, equivalent circuit and covariate set (Sec. Continuous sensing)."""

    sweep: Mapping[str, Any]
    circuit: Mapping[str, Any]
    comparator: Mapping[str, Any]
    resolvable: Mapping[str, Any]
    optical: Mapping[str, Any]
    effluent: Mapping[str, Any]
    ladders: Mapping[str, Any]

    @classmethod
    def load(cls, root: Path) -> FrontendConfig:
        grid = _load_yaml(root / "configs" / "frontend" / "impedance_grid.yaml")
        ladders = _load_yaml(root / "configs" / "frontend" / "ladders.yaml")
        return cls(
            sweep=cast("Mapping[str, Any]", grid["sweep"]),
            circuit=cast("Mapping[str, Any]", grid["equivalent_circuit"]),
            comparator=cast("Mapping[str, Any]", grid["comparator"]),
            resolvable=cast("Mapping[str, Any]", grid["resolvable_change"]),
            optical=cast("Mapping[str, Any]", grid["optical"]),
            effluent=cast("Mapping[str, Any]", grid["effluent"]),
            ladders=cast("Mapping[str, Any]", ladders),
        )

    @property
    def frequencies_hz(self) -> tuple[float, ...]:
        values = cast("Sequence[float]", self.sweep["frequencies_hz"])
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class CalibrationConfig:
    """Pooled fit, attenuation and identifiability settings (Eq. (3)-(5))."""

    pooled: Mapping[str, Any]
    attenuation: Mapping[str, Any]
    audit: Mapping[str, Any]
    bootstrap: Mapping[str, Any]

    @classmethod
    def load(cls, root: Path) -> CalibrationConfig:
        payload = _load_yaml(root / "configs" / "calibration" / "pooled.yaml")
        return cls(
            pooled=cast("Mapping[str, Any]", payload["pooled_fit"]),
            attenuation=cast("Mapping[str, Any]", payload["attenuation"]),
            audit=cast("Mapping[str, Any]", payload["identifiability_audit"]),
            bootstrap=cast("Mapping[str, Any]", payload["bootstrap"]),
        )

    @property
    def feature_dimension(self) -> int:
        return int(self.pooled["feature_dimension_p"])

    @property
    def biological_devices(self) -> int:
        return int(self.pooled["biological_calibration_devices"])

    @property
    def covariance_inflation(self) -> float:
        n_bio = self.biological_devices
        return n_bio / (n_bio - self.feature_dimension - 2)


@dataclass(frozen=True)
class AllocationConfig:
    """D-optimal objective, budget grid and comparator policies (Eq. (6))."""

    criterion: Mapping[str, Any]
    core: Mapping[str, Any]
    grid: Mapping[str, Any]
    policies: Mapping[str, Any]
    checks: Mapping[str, Any]
    operating_point: Mapping[str, Any]

    @classmethod
    def load(cls, root: Path) -> AllocationConfig:
        payload = _load_yaml(root / "configs" / "allocation" / "budget_grid.yaml")
        policies = {
            "names": payload["policies"],
            "random_matched_rate": payload["random_matched_rate"],
            "fixed_schedule": payload["fixed_schedule"],
            "prior_art_allocator": payload["prior_art_allocator"],
            "clairvoyant_oracle": payload["clairvoyant_oracle"],
        }
        return cls(
            criterion=cast("Mapping[str, Any]", payload["criterion"]),
            core=cast("Mapping[str, Any]", payload["core"]),
            grid=cast("Mapping[str, Any]", payload["budget_grid"]),
            policies=cast("Mapping[str, Any]", policies),
            checks=cast("Mapping[str, Any]", payload["structural_checks"]),
            operating_point=cast("Mapping[str, Any]", payload["operating_point"]),
        )

    @property
    def budget_fractions(self) -> tuple[float, ...]:
        values = cast("Sequence[float]", self.grid["budget_fractions"])
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class TwinConfig:
    """Backbone, training and forecast settings (Eq. (7))."""

    backbone: Mapping[str, Any]
    training: Mapping[str, Any]
    forecast: Mapping[str, Any]
    ladder: Mapping[str, Any]

    @classmethod
    def load(cls, root: Path) -> TwinConfig:
        backbone = _load_yaml(root / "configs" / "twin" / "backbone.yaml")
        ladder = _load_yaml(root / "configs" / "twin" / "ladder.yaml")
        return cls(
            backbone=cast("Mapping[str, Any]", backbone["backbone"]),
            training=cast("Mapping[str, Any]", backbone["training"]),
            forecast=cast("Mapping[str, Any]", backbone["forecast"]),
            ladder=cast("Mapping[str, Any]", ladder),
        )

    @property
    def latent_dimension(self) -> int:
        return int(self.backbone["latent_state_dimension_deployed"])

    @property
    def horizons_days(self) -> tuple[int, ...]:
        values = cast("Sequence[int]", self.forecast["horizons_days"])
        return tuple(int(value) for value in values)


@dataclass(frozen=True)
class RunSpec:
    """One entry of ``configs/run``: which components are in and what is asked."""

    name: str
    kind: str
    title: str
    seed: int
    toggles: FeatureToggles
    analyses: tuple[AnalysisSpec, ...]
    payload: Mapping[str, Any]

    @property
    def is_harness(self) -> bool:
        return self.kind == "harness"

    def analysis(self, name: str) -> AnalysisSpec | None:
        for spec in self.analyses:
            if spec.name == name:
                return spec
        return None

    def option(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def with_toggles(self, **changes: bool) -> RunSpec:
        return replace(self, toggles=replace(self.toggles, **changes))


def load_run_spec(path: str | Path) -> RunSpec:
    run_path = Path(path)
    if not run_path.is_absolute():
        candidates = [Path.cwd() / run_path, repo_root() / run_path]
        run_path = next((item for item in candidates if item.is_file()), candidates[-1])
    payload = load_run_payload(run_path)
    run_block = cast("Mapping[str, Any]", payload.get("run", {}))
    analyses_raw = payload.get("analyses", [])
    analyses = tuple(
        AnalysisSpec(
            name=str(cast("Mapping[str, Any]", entry)["name"]),
            options={
                key: value
                for key, value in cast("Mapping[str, Any]", entry).items()
                if key != "name"
            },
        )
        for entry in cast("Sequence[Any]", analyses_raw)
    )
    return RunSpec(
        name=str(run_block.get("name", run_path.stem)),
        kind=str(run_block.get("kind", "primary")),
        title=str(run_block.get("title", run_path.stem)),
        seed=int(payload.get("seed", 0)),
        toggles=FeatureToggles.from_mapping(cast("Mapping[str, Any]", payload.get("toggles", {}))),
        analyses=analyses,
        payload=payload,
    )
