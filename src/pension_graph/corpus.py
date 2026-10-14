from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pension_graph.models import ChunkRecord, DocumentRecord

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DOC_NUMBER_RE = re.compile(r"^doc(\d+)\.md$")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _natural_doc_key(path: Path) -> tuple[int, str]:
    match = DOC_NUMBER_RE.match(path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def discover_documents(corpus_dir: Path) -> list[Path]:
    files = sorted(corpus_dir.glob("doc*.md"), key=_natural_doc_key)
    invalid = [path for path in files if not DOC_NUMBER_RE.match(path.name)]
    if invalid:
        raise ValueError(f"허용되지 않은 파일명: {invalid}")
    return files


def _first_title(lines: list[str], fallback: str) -> str:
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    for line in lines:
        if line.strip():
            return line.strip()[:160]
    return fallback


def build_document(path: Path) -> tuple[DocumentRecord, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    source_id = path.stem
    return (
        DocumentRecord(
            document_id=f"document:{source_id}",
            source_id=source_id,
            filename=path.name,
            title=_first_title(lines, source_id),
            sha256=_sha256(text),
            byte_size=len(text.encode("utf-8")),
            line_count=len(lines),
        ),
        text,
    )


@dataclass
class _Block:
    headings: tuple[str, ...]
    start_line: int
    end_line: int
    content: str


def _section_blocks(text: str) -> list[_Block]:
    """Split on Markdown headings while preserving tables and exact source text."""
    lines = text.splitlines()
    headings: list[str] = []
    blocks: list[_Block] = []
    current: list[str] = []
    start = 1
    current_headings: tuple[str, ...] = ()

    def flush(end_line: int) -> None:
        nonlocal current, start
        content = "\n".join(current).strip()
        if content:
            blocks.append(_Block(current_headings, start, end_line, content))
        current = []

    for line_no, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            flush(line_no - 1)
            level = len(match.group(1))
            headings[:] = headings[: level - 1]
            headings.append(match.group(2).strip())
            current_headings = tuple(headings)
            start = line_no
        current.append(line)
    flush(len(lines))
    return blocks


def _split_oversized(block: _Block, max_chars: int) -> Iterable[_Block]:
    if len(block.content) <= max_chars:
        yield block
        return

    lines = block.content.splitlines()
    buffer: list[str] = []
    start = block.start_line
    current_len = 0
    for offset, line in enumerate(lines):
        addition = len(line) + (1 if buffer else 0)
        # Never split a Markdown table row from adjacent table rows unless unavoidable.
        is_table_line = line.lstrip().startswith("|")
        buffer_is_table = bool(buffer and buffer[-1].lstrip().startswith("|"))
        if (
            buffer
            and current_len + addition > max_chars
            and not (is_table_line and buffer_is_table)
        ):
            end = block.start_line + offset - 1
            yield _Block(block.headings, start, end, "\n".join(buffer).strip())
            buffer = []
            start = block.start_line + offset
            current_len = 0
        buffer.append(line)
        current_len += addition
    if buffer:
        yield _Block(block.headings, start, block.end_line, "\n".join(buffer).strip())


def chunk_document(
    document: DocumentRecord,
    text: str,
    *,
    max_chars: int = 4000,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    expanded = (
        split for block in _section_blocks(text) for split in _split_oversized(block, max_chars)
    )
    for index, block in enumerate(expanded):
        digest = _sha256(block.content)
        chunk_id = f"{document.source_id}:chunk:{index:04d}:{digest[:12]}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document.document_id,
                source_id=document.source_id,
                index=index,
                heading_path=block.headings,
                content=block.content,
                start_line=block.start_line,
                end_line=block.end_line,
                sha256=digest,
            )
        )
    return chunks
