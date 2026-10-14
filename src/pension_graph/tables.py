from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any

from pension_graph.models import ChunkRecord

MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
LAYOUT_CELL_GAP_RE = re.compile(r"\s{3,}")


@dataclass(frozen=True)
class TableRecord:
    table_id: str
    chunk_id: str
    source_id: str
    table_index: int
    format: str
    heading_path: tuple[str, ...]
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        value["headers"] = list(self.headers)
        value["rows"] = [list(row) for row in self.rows]
        return value


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    stripped = stripped.removeprefix("|")
    stripped = stripped.removesuffix("|")
    values = re.split(r"(?<!\\)\|", stripped)
    return [
        html.unescape(value.replace("\\|", "|").strip()).replace("<br>", "\n") for value in values
    ]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(MARKDOWN_SEPARATOR_RE.match(cell.replace(" ", "")) for cell in cells)


def _split_layout_row(line: str) -> list[str]:
    if len(line) - len(line.lstrip()) < 2 or line.lstrip().startswith(("|", "#")):
        return []
    return [html.unescape(value.strip()) for value in LAYOUT_CELL_GAP_RE.split(line.strip())]


def _layout_groups(lines: list[str]) -> list[list[tuple[int, list[str]]]]:
    groups: list[list[tuple[int, list[str]]]] = []
    current: list[tuple[int, list[str]]] = []
    for line_index, line in enumerate(lines):
        cells = _split_layout_row(line)
        if len(cells) < 2:
            continue
        if current:
            previous_index = current[-1][0]
            between = lines[previous_index + 1 : line_index]
            joinable = line_index - previous_index <= 3 and all(
                not value.strip() or len(value) - len(value.lstrip()) >= 2
                for value in between
            )
            if not joinable:
                if len(current) >= 3:
                    groups.append(current)
                current = []
        current.append((line_index, cells))
    if len(current) >= 3:
        groups.append(current)
    return groups


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.header_flags: list[list[bool]] = []
        self._row: list[str] | None = None
        self._flags: list[bool] | None = None
        self._cell: list[str] | None = None
        self._is_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row, self._flags = [], []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []
            self._is_header = tag == "th"
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            value = re.sub(r"[ \t]+", " ", "".join(self._cell)).strip()
            self._row.append(value)
            assert self._flags is not None
            self._flags.append(self._is_header)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
                self.header_flags.append(self._flags or [False] * len(self._row))
            self._row, self._flags = None, None


def _make_table(
    chunk: ChunkRecord,
    table_index: int,
    format_name: str,
    headers: list[str],
    rows: list[list[str]],
    raw: str,
) -> TableRecord | None:
    if not headers or not rows:
        return None
    width = max(len(headers), *(len(row) for row in rows))
    normalized_headers = tuple(
        (headers[index] if index < len(headers) and headers[index] else f"column_{index + 1}")
        for index in range(width)
    )
    normalized_rows = tuple(
        tuple(row[index] if index < len(row) else "" for index in range(width)) for row in rows
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return TableRecord(
        table_id=f"{chunk.chunk_id}:table:{table_index:03d}:{digest}",
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        table_index=table_index,
        format=format_name,
        heading_path=chunk.heading_path,
        headers=normalized_headers,
        rows=normalized_rows,
        raw_text=raw,
    )


def extract_tables(chunk: ChunkRecord) -> list[TableRecord]:
    results: list[TableRecord] = []
    index = 0

    # Inline HTML tables are common inside Markdown cells; parse them first.
    for match in HTML_TABLE_RE.finditer(chunk.content):
        parser = _HTMLTableParser()
        parser.feed(match.group(0))
        if not parser.rows:
            continue
        header_row = 0
        for row_index, flags in enumerate(parser.header_flags):
            if any(flags):
                header_row = row_index
                break
        table = _make_table(
            chunk,
            index,
            "html",
            parser.rows[header_row],
            parser.rows[header_row + 1 :],
            match.group(0),
        )
        if table:
            results.append(table)
            index += 1

    # Parse Markdown rows from the original chunk. Inline HTML can be a cell value;
    # removing it would create a synthetic raw_text that no longer exists in source.
    markdown_only = chunk.content
    lines = markdown_only.splitlines()
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].lstrip().startswith("|"):
            cursor += 1
            continue
        group: list[str] = []
        while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
            group.append(lines[cursor])
            cursor += 1
        parsed = [_split_markdown_row(line) for line in group]
        if len(parsed) < 2 or not _is_separator(parsed[1]):
            continue
        table = _make_table(chunk, index, "markdown", parsed[0], parsed[2:], "\n".join(group))
        if table:
            results.append(table)
            index += 1

    # PDF-to-Markdown output can preserve a table only as aligned text. Treat
    # repeated indented rows separated by 3+ spaces as a derived layout table;
    # raw_text remains an exact substring of the original chunk for provenance.
    for group in _layout_groups(lines):
        start = group[0][0]
        end = group[-1][0]
        raw = "\n".join(lines[start : end + 1])
        if raw not in chunk.content:
            continue
        table = _make_table(
            chunk,
            index,
            "layout",
            group[0][1],
            [cells for _, cells in group[1:]],
            raw,
        )
        if table:
            results.append(table)
            index += 1
    return results


def extract_all_tables(chunks: list[ChunkRecord]) -> list[TableRecord]:
    return [table for chunk in chunks for table in extract_tables(chunk)]
