"""Plain-text rendering of the release's own tables.

Every width is taken from the data, so a long value is never silently truncated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from obscal.runtime.atomic import write_text

DEFAULT_GAP = "  "


def format_value(value: object) -> str:
    """Stable, width-friendly rendering of one cell."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        if value == 0.0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1e5 or magnitude < 1e-4:
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if value is None:
        return "-"
    return str(value)


def column_widths(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> list[int]:
    widths = [len(column) for column in columns]
    for row in rows:
        for index, column in enumerate(columns):
            widths[index] = max(widths[index], len(format_value(row.get(column))))
    return widths


def render_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str] | None = None,
    *,
    title: str | None = None,
    gap: str = DEFAULT_GAP,
) -> str:
    if not rows:
        return f"{title}\n(no rows)\n" if title else "(no rows)\n"
    names = (
        list(columns)
        if columns is not None
        else list(dict.fromkeys(key for row in rows for key in row))
    )
    widths = column_widths(rows, names)
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(gap.join(name.ljust(width) for name, width in zip(names, widths, strict=True)))
    lines.append(gap.join("-" * width for width in widths))
    for row in rows:
        lines.append(
            gap.join(
                format_value(row.get(name)).ljust(width)
                for name, width in zip(names, widths, strict=True)
            )
        )
    return "\n".join(lines) + "\n"


def render_key_values(values: Mapping[str, object], *, width: int = 42) -> str:
    if not values:
        return "(no values)\n"
    label_width = max(width, max(len(key) for key in values))
    return (
        "\n".join(
            f"{key.ljust(label_width)} : {format_value(value)}" for key, value in values.items()
        )
        + "\n"
    )


def render_scalar_table(name: str, values: Mapping[str, float], *, title: str | None = None) -> str:
    rows = [{"key": key, "value": value} for key, value in values.items()]
    return render_table(rows, ["key", "value"], title=title or name)


def render_matrix(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    *,
    title: str | None = None,
) -> str:
    if len(matrix) != len(row_labels):
        raise ValueError("every matrix row needs a label")
    for row in matrix:
        if len(row) != len(column_labels):
            raise ValueError("every matrix column needs a label")
    rows = [
        {"": label, **{column: row[index] for index, column in enumerate(column_labels)}}
        for label, row in zip(row_labels, matrix, strict=True)
    ]
    columns = ["", *column_labels]
    return render_table(rows, columns, title=title)


def render_sections(sections: Sequence[tuple[str, Sequence[Mapping[str, object]]]]) -> str:
    blocks: list[str] = []
    for title, rows in sections:
        blocks.append(render_table(rows, title=title))
    return "\n".join(blocks)


def write_report(path: Path, text: str) -> None:
    write_text(Path(path), text)


def join_sections(*parts: str) -> str:
    return "\n".join(part.rstrip("\n") for part in parts if part) + "\n"
