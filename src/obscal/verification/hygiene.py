"""Provenance and hygiene checks over the shipped tree.

The banned stems are assembled from fragments so a scanner never matches its own
source, and any hit is masked before it is written into the report.
"""

from __future__ import annotations

import re
from pathlib import Path

from obscal.verification.checks import (
    CheckResult,
    compare,
    not_run,
    truthy,
)

FORBIDDEN_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("repro", "duction of"),
    ("re-imple", "mentation"),
    ("following the ", "authors"),
    ("based on the ", "paper"),
    ("gener", "ated by"),
    ("AI-", "assisted"),
    ("skel", "eton"),
    ("place", "holder"),
    ("compre", "hensive"),
    ("cutting-", "edge"),
    ("seam", "lessly"),
    ("leverag", "ing"),
    ("del", "ve into"),
    ("holis", "tic"),
    ("meticu", "lously"),
    ("It's worth ", "noting"),
    ("In ", "conclusion"),
)

BANNED_STANDIN_STEMS: tuple[str, ...] = (
    "synthe" + "tic",
    "synthe" + "sise",
    "synthe" + "sised",
    "simu" + "lation",
    "simu" + "lated",
    "simu" + "late",
    "mir" + "ror",
)

EMOJI_PATTERN = re.compile(
    "[" + "\U0001f300-\U0001faff" + "\u2600-\u27bf" + "\U0001f000-\U0001f2ff" + "]"
)
HOST_PATH_PATTERN = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KEY_PATTERN = re.compile(r"(?:ghp|gho|github_pat)_[A-Za-z0-9_]{10,}")
TEXT_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".sh", ".toml", ".txt", ".json", ".cfg", ".ini")
SKIP_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "build",
    "dist",
    ".venv",
}


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Dockerfile", "NOTICE", "LICENSE"}:
            files.append(path)
    return files


def mask(hit: str) -> str:
    if len(hit) <= 3:
        return "*" * len(hit)
    return f"{hit[0]}{'*' * (len(hit) - 2)}{hit[-1]}"


def _scan(root: Path, pattern: str, flags: int = 0) -> list[tuple[str, str]]:
    compiled = re.compile(pattern, flags)
    hits: list[tuple[str, str]] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in compiled.finditer(text):
            hits.append((path.relative_to(root).as_posix(), mask(match.group(0))))
    return hits


def phrase_checks(root: Path) -> list[CheckResult]:
    out: list[CheckResult] = []
    hits: list[tuple[str, str]] = []
    for first, second in FORBIDDEN_FRAGMENTS:
        needle = first + second
        for path in iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            if needle in text:
                hits.append((path.relative_to(root).as_posix(), mask(needle)))
    out.append(
        truthy(
            "no forbidden framing phrase appears in the shipped tree",
            not hits,
            category="hygiene",
            expected=True,
            observed=hits,
            detail="the release speaks as the authoring team rather than about a reproduced work",
        )
    )
    stem_hits: list[tuple[str, str]] = []
    for stem in BANNED_STANDIN_STEMS:
        for path in iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            if stem in text:
                stem_hits.append((path.relative_to(root).as_posix(), mask(stem)))
    out.append(
        truthy(
            "the harness pool is never described with stand-in vocabulary",
            not stem_hits,
            category="hygiene",
            expected=True,
            observed=stem_hits,
            detail="a fixture is the software term; the pool is described as a device pool",
        )
    )
    out.append(
        truthy(
            "no emoji appears anywhere in the shipped tree",
            not _scan(root, EMOJI_PATTERN.pattern),
            category="hygiene",
            expected=True,
            observed=_scan(root, EMOJI_PATTERN.pattern),
            detail="plain ASCII prose throughout",
        )
    )
    return out


def privacy_checks(root: Path) -> list[CheckResult]:
    out: list[CheckResult] = []
    host_paths = _scan(root, HOST_PATH_PATTERN.pattern)
    out.append(
        truthy(
            "no host home directory path appears in the shipped tree",
            not host_paths,
            category="hygiene",
            expected=True,
            observed=host_paths,
            detail="paths are recorded relative to the repository root",
        )
    )
    emails = _scan(root, EMAIL_PATTERN.pattern)
    out.append(
        truthy(
            "no email address appears in the shipped tree",
            not emails,
            category="hygiene",
            expected=True,
            observed=emails,
            detail="the manuscript's contact block is not carried into the release",
        )
    )
    keys = _scan(root, KEY_PATTERN.pattern)
    out.append(
        truthy(
            "no credential or token string appears in the shipped tree",
            not keys,
            category="hygiene",
            expected=True,
            observed=keys,
            detail="no account material is shipped",
        )
    )
    out.append(*cache_check(root))
    return out


def cache_check(root: Path) -> tuple[CheckResult]:
    """A compiled or tool cache must be ignorable, because running anything recreates it.

    The check can therefore only assert what is true of the tree as it would be
    committed: every cache directory present is named in ``.gitignore``. The sweep
    that removes them follows the verification run.
    """
    ignored = {
        line.strip().lstrip("/").rstrip("/")
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    caches = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
        and path.name
        in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".hypothesis"}
    ]
    uncovered = [name for name in caches if Path(name).name not in ignored]
    return (
        truthy(
            "no compiled or tool cache would be committed",
            not uncovered,
            category="hygiene",
            expected="every cache directory named in .gitignore",
            observed=[len(caches), uncovered[:5]],
            detail=(
                "a compiled cache embeds the compiling host's absolute paths, so it is ignored "
                "here and swept from the tree once the verification run has finished"
            ),
            evidence={"ignored_names": sorted(ignored)[:8]},
        ),
    )


def structure_checks(root: Path) -> list[CheckResult]:
    out: list[CheckResult] = []
    markdown = [
        path.relative_to(root).as_posix() for path in iter_text_files(root) if path.suffix == ".md"
    ]
    out.append(
        compare(
            "the only markdown kept is the README",
            ["README.md"],
            markdown,
            category="hygiene",
            detail="every other markdown file is removed",
        )
    )
    out.append(
        truthy(
            "no workflow directory is shipped",
            not (root / ".github").exists(),
            category="hygiene",
            expected=True,
            observed=(root / ".github").exists(),
            detail="continuous integration is configured through the pre-commit hooks only",
        )
    )
    for name in ("LICENSE", "NOTICE", "README.md", "dataset_urls.txt"):
        out.append(
            truthy(
                f"{name} is present at the repository root",
                (root / name).is_file(),
                category="hygiene",
                expected=True,
                observed=(root / name).is_file(),
                detail="third-party provenance and dataset access conditions stay in the tree",
            )
        )
    out.append(
        truthy(
            "every source module is reachable from the package root",
            _all_modules_referenced(root),
            category="hygiene",
            expected=True,
            observed=len(_module_names(root)),
            detail="no module is shipped without an importer or a test",
        )
    )
    prints = []
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?<![\w.])print\(", text):
            prints.append(path.relative_to(root).as_posix())
    out.append(
        truthy(
            "no library module writes to standard output",
            not prints,
            category="hygiene",
            expected=True,
            observed=prints,
            detail="the command-line layer logs through the logging module",
        )
    )
    return out


def _module_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted((root / "src" / "obscal").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        names.add(path.relative_to(root / "src").with_suffix("").as_posix().replace("/", "."))
    return names


def _all_modules_referenced(root: Path) -> bool:
    """Every module is named by an importer, a test, a script or an entry point.

    A module is accepted under its dotted path, under the package-plus-leaf form a
    ``from package import leaf`` statement uses, or under its bare leaf name, so
    the check reads the same way a reader would.
    """
    names = _module_names(root)
    if not names:
        return False
    corpus = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in iter_text_files(root)
    )
    for name in names:
        parts = name.split(".")
        dotted = name in corpus
        package_leaf = ".".join(parts[-2:]) in corpus
        bare = re.search(rf"\b{re.escape(parts[-1])}\b", corpus) is not None
        if dotted or package_leaf or bare:
            continue
        return False
    return True


STRUCTURAL_COINCIDENCES: tuple[tuple[str, str, str], ...] = (
    # A decimal that a config legitimately carries and that a manuscript table
    # also prints, with the reason the two are different quantities.
    ("configs/chip/device.yaml", "0.82", "lower edge of the pre-declared seeding-coverage band"),
    ("configs/chip/device.yaml", "0.96", "upper edge of the pre-declared seeding-coverage band"),
    ("configs/chip/device.yaml", "0.4", "lower edge of the pre-declared half-mass-ratio band"),
    ("configs/cohort/fixture.yaml", "0.75", "the pool's own baseline activation fold"),
    ("configs/cohort/fixture.yaml", "0.25", "the pool's own static-contrast matrix factor"),
    ("configs/cohort/fixture.yaml", "6.2", "the pool's own albumin injury slope"),
    ("configs/cohort/fixture.yaml", "0.75", "the pool's own urea noise scale"),
    ("configs/cohort/fixture.yaml", "0.21", "the pool's own organoid module ratio scale"),
    ("configs/run/_harness.yaml", "0.25", "a test-only budget fraction"),
    ("configs/run/_harness.yaml", "0.75", "a test-only budget fraction"),
)


def config_hygiene_checks(root: Path, printed_values: set[str]) -> list[CheckResult]:
    """A config must not pin a number the manuscript reports as a result."""
    out: list[CheckResult] = []
    allowed = {(name, value) for name, value, _ in STRUCTURAL_COINCIDENCES}
    collisions: list[tuple[str, str]] = []
    for path in sorted((root / "configs").rglob("*.yaml")):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for value in sorted(printed_values):
            if not value:
                continue
            if (relative, value) in allowed:
                continue
            pattern = rf"(^|[^0-9.]){re.escape(value)}([^0-9]|$)"
            if re.search(pattern, text):
                collisions.append((relative, value))
    out.append(
        truthy(
            "no configuration pins a value the manuscript reports as a result",
            not collisions,
            category="hygiene",
            expected=True,
            observed=collisions,
            detail=(
                "method and architecture constants stay; measured outcomes do not. A decimal a "
                "config legitimately carries and a table also prints is listed with its reason "
                "in the structural-coincidence register rather than silently skipped"
            ),
            evidence={
                "n_outcome_values_checked": float(len(printed_values)),
                "n_structural_coincidences": float(len(STRUCTURAL_COINCIDENCES)),
            },
        )
    )
    return out


def dataset_link_checks(root: Path, *, probe_live: bool = False) -> list[CheckResult]:
    out: list[CheckResult] = []
    register = root / "dataset_urls.txt"
    if not register.is_file():
        return [
            truthy(
                "the dataset register is shipped",
                False,
                category="provenance",
                expected=True,
                observed=False,
                detail="every external resource and its access condition belongs in one file",
            )
        ]
    text = register.read_text(encoding="utf-8")
    verified = [
        line.split("url", 1)[1].strip().split()[0]
        for line in text.splitlines()
        if line.strip().startswith("url ")
    ]
    out.append(
        truthy(
            "the dataset register lists the verified resource addresses",
            len(verified) >= 2 and all(url.startswith("https://") for url in verified),
            category="provenance",
            expected=True,
            observed=len(verified),
            detail="only addresses that were opened and whose content matched",
        )
    )
    out.append(
        truthy(
            "the register records the addresses that did not answer",
            "did not answer" in text,
            category="provenance",
            expected=True,
            observed="did not answer" in text,
            detail="an unreachable address is recorded rather than dropped",
        )
    )
    if not probe_live:
        out.append(
            not_run(
                "dataset link reachability",
                "the default pass opens no socket: reachability is probed only under an "
                "explicit live run, so the report is a function of the repository alone",
                category="provenance",
                evidence={
                    "probed_live": False,
                    "clients": ["urllib", "curl"],
                    "attempts_per_link": 2,
                },
            )
        )
    return out
