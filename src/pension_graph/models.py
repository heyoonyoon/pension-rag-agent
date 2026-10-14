from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source_id: str
    filename: str
    title: str
    sha256: str
    byte_size: int
    line_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    source_id: str
    index: int
    heading_path: tuple[str, ...]
    content: str
    start_line: int
    end_line: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        return value


@dataclass(frozen=True)
class MentionRecord:
    mention_id: str
    chunk_id: str
    concept_id: str
    matched_text: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
