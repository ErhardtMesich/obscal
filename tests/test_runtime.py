"""Seeding, atomic writes and configuration binding."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from obscal.runtime.atomic import (
    SCHEMA_VERSION,
    file_digest,
    payload_digest,
    read_json,
    write_json,
    write_text,
)
from obscal.runtime.config import (
    AllocationConfig,
    CalibrationConfig,
    ChipConfig,
    FeatureToggles,
    FrontendConfig,
    TwinConfig,
    load_run_payload,
    load_run_spec,
    repo_root,
)
from obscal.runtime.seed import active_seed, numpy_generator, set_seed


def test_set_seed_is_reproducible() -> None:
    assert set_seed(7) == 7
    first = np.random.random(4)
    set_seed(7)
    assert np.array_equal(first, np.random.random(4))
    assert active_seed() == 7
    assert not np.array_equal(numpy_generator(1).random(3), numpy_generator(2).random(3))


def test_atomic_writes_are_readable_and_leave_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"
    write_json(target, {"value": 1})
    assert read_json(target) == {"value": 1}
    assert target.stat().st_mode & 0o777 == 0o644
    assert [path.name for path in tmp_path.iterdir()] == ["payload.json"]
    text = tmp_path / "notes.txt"
    write_text(text, "hello\n")
    assert text.read_text() == "hello\n"


def test_digests_track_contents(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_json(tmp_path / "unserialisable.json", {"bad": object()})
    assert payload_digest([1.0, 2.0]) == payload_digest([1.0, 2.0])
    assert payload_digest([1.0, 2.0]) != payload_digest([1.0, 2.5])
    assert SCHEMA_VERSION.startswith("obscal/")


def test_toggles_report_what_a_run_removes() -> None:
    toggles = FeatureToggles.from_mapping({"damping": False, "coverage_covariate": False})
    assert not toggles.damping
    assert toggles.two_axis_state
    assert toggles.removed() == ("coverage_covariate", "damping")


def test_run_spec_inheritance_and_overrides(project_root: Path) -> None:
    primary = load_run_spec(project_root / "configs/run/primary.yaml")
    ablation = load_run_spec(project_root / "configs/run/ablation_single_frequency.yaml")
    assert primary.toggles.multi_frequency_front_end
    assert not ablation.toggles.multi_frequency_front_end
    assert ablation.toggles.split_chip_replication
    assert ablation.analysis("calibration") is not None
    assert primary.analysis("missing") is None
    assert primary.option("seed", 0) == primary.seed


def test_module_configs_bind_to_typed_containers(project_root: Path) -> None:
    chip = ChipConfig.load(project_root)
    frontend = FrontendConfig.load(project_root)
    calibration = CalibrationConfig.load(project_root)
    allocation = AllocationConfig.load(project_root)
    twin = TwinConfig.load(project_root)
    assert chip.batching["batches"] == 3
    assert frontend.frequencies_hz[0] == 100.0
    assert frontend.frequencies_hz[-1] == 100000.0
    assert calibration.feature_dimension == 5
    assert calibration.biological_devices == 116
    assert calibration.covariance_inflation == pytest.approx(1.0642, abs=5e-5)
    assert allocation.budget_fractions[-1] == 1.0
    assert twin.latent_dimension == 32
    assert twin.horizons_days == (1, 3, 7, 14)


def test_run_payload_merges_an_inheritance_chain(project_root: Path) -> None:
    payload = load_run_payload(project_root / "configs/run/ablation_undamped_backbone.yaml")
    assert payload["run"]["component_removed"] == "energy_dissipation_term"
    assert payload["toggles"]["damping"] is False
    assert payload["toggles"]["split_chip_replication"] is True


def test_repo_root_resolves_from_a_run_config(project_root: Path) -> None:
    assert repo_root(project_root / "configs/run/primary.yaml") == project_root
    assert (repo_root() / "pyproject.toml").is_file()


def test_shipped_artefacts_are_valid_json(project_root: Path) -> None:
    for name in ("claim_to_code.json", "verification_report.json"):
        payload = json.loads((project_root / name).read_text(encoding="utf-8"))
        assert payload
    assert os.access(project_root / "verification_summary.txt", os.R_OK)
    assert file_digest(project_root / "LICENSE")
