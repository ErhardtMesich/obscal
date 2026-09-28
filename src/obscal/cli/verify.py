"""Run the release's verification driver.

The driver writes the claim map, the verification report and the plain-text
summary, then the integrity manifest, in that order.
"""

from __future__ import annotations

from obscal.verification.driver import main as _driver_main


def main(argv: list[str] | None = None) -> int:
    return _driver_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
