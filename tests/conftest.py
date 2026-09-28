"""Shared fixtures for the release's test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from obscal.cohort import Cohort, FixtureSpec, build_cohort_fixture
from obscal.runtime.config import RunSpec, load_run_spec, repo_root
from obscal.runtime.seed import set_seed
from obscal.study import StudyInputs


@pytest.fixture(scope="session")
def project_root() -> Path:
    return repo_root(Path("configs/run/primary.yaml"))


@pytest.fixture(scope="session")
def run_spec(project_root: Path) -> RunSpec:
    return load_run_spec(project_root / "configs/run/primary.yaml")


@pytest.fixture(scope="session")
def harness_cohort(project_root: Path) -> Cohort:
    return build_cohort_fixture(FixtureSpec.load(project_root))


@pytest.fixture(scope="session")
def study_inputs(project_root: Path, harness_cohort: Cohort) -> StudyInputs:
    return StudyInputs.load(project_root, harness_cohort)


@pytest.fixture(autouse=True)
def _deterministic_seed() -> Iterator[None]:
    set_seed(20260928)
    yield
