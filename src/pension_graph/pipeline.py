from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

from pension_graph.concepts import Concept, find_mentions, load_concepts
from pension_graph.config import Paths
from pension_graph.corpus import build_document, chunk_document, discover_documents
from pension_graph.models import ChunkRecord, DocumentRecord, MentionRecord
from pension_graph.rdf_store import build_corpus_graph, validate_graph
from pension_graph.tables import TableRecord, extract_all_tables


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build(paths: Paths, *, max_chars: int = 4000) -> dict[str, int]:
    corpus_dir = paths.assert_corpus_path(paths.raw_markdown)
    concept_config = paths.project_root / "config" / "concepts.yaml"
    concepts = load_concepts(concept_config)
    documents: list[DocumentRecord] = []
    chunks: list[ChunkRecord] = []
    for file_path in discover_documents(corpus_dir):
        document, text = build_document(file_path)
        documents.append(document)
        chunks.extend(chunk_document(document, text, max_chars=max_chars))
    mentions = find_mentions(chunks, concepts)
    tables = extract_all_tables(chunks)

    output = paths.processed
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "documents.jsonl", (item.to_dict() for item in documents))
    _write_jsonl(output / "chunks.jsonl", (item.to_dict() for item in chunks))
    _write_jsonl(output / "mentions.jsonl", (item.to_dict() for item in mentions))
    _write_jsonl(output / "tables.jsonl", (item.to_dict() for item in tables))

    graph = build_corpus_graph(documents, chunks, concepts, mentions, tables)
    graph.serialize(output / "knowledge-base.ttl", format="turtle")
    conforms, report = validate_graph(graph, paths.ontology)
    (output / "shacl-report.txt").write_text(report, encoding="utf-8")
    if not conforms:
        raise ValueError(f"SHACL 검증 실패. 보고서: {output / 'shacl-report.txt'}")

    _write_neo4j_csv(output / "neo4j", documents, chunks, concepts, mentions, tables)
    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "concepts": len(concepts),
        "mentions": len(mentions),
        "tables": len(tables),
        "table_rows": sum(len(table.rows) for table in tables),
        "rdf_triples": len(graph),
    }
    _write_jsonl(output / "build-stats.jsonl", [stats])
    return stats


def _write_neo4j_csv(
    output: Path,
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    concepts: list[Concept],
    mentions: list[MentionRecord],
    tables: list[TableRecord],
) -> None:
    _write_csv(
        output / "documents.csv",
        ["source_id", "filename", "title", "sha256", "byte_size", "line_count"],
        (
            {
                "source_id": d.source_id,
                "filename": d.filename,
                "title": d.title,
                "sha256": d.sha256,
                "byte_size": d.byte_size,
                "line_count": d.line_count,
            }
            for d in documents
        ),
    )
    _write_csv(
        output / "chunks.csv",
        [
            "chunk_id",
            "source_id",
            "index",
            "heading_path",
            "content",
            "start_line",
            "end_line",
            "sha256",
        ],
        (
            {
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "index": c.index,
                "heading_path": " > ".join(c.heading_path),
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "sha256": c.sha256,
            }
            for c in chunks
        ),
    )
    _write_csv(
        output / "concepts.csv",
        ["concept_id", "label", "ontology_class", "aliases"],
        (
            {
                "concept_id": c.concept_id,
                "label": c.label,
                "ontology_class": c.ontology_class,
                "aliases": json.dumps(c.aliases, ensure_ascii=False),
            }
            for c in concepts
        ),
    )
    _write_csv(
        output / "mentions.csv",
        ["mention_id", "chunk_id", "concept_id", "matched_text", "count"],
        (m.to_dict() for m in mentions),
    )
    next_rows = []
    for source_id in {chunk.source_id for chunk in chunks}:
        ordered = sorted((c for c in chunks if c.source_id == source_id), key=lambda c: c.index)
        next_rows.extend(
            {"from_chunk_id": left.chunk_id, "to_chunk_id": right.chunk_id}
            for left, right in pairwise(ordered)
        )
    _write_csv(output / "next_chunks.csv", ["from_chunk_id", "to_chunk_id"], next_rows)
    _write_csv(
        output / "tables.csv",
        ["table_id", "chunk_id", "source_id", "table_index", "format", "heading_path", "raw_text"],
        (
            {
                "table_id": table.table_id,
                "chunk_id": table.chunk_id,
                "source_id": table.source_id,
                "table_index": table.table_index,
                "format": table.format,
                "heading_path": " > ".join(table.heading_path),
                "raw_text": table.raw_text,
            }
            for table in tables
        ),
    )
    table_cells = []
    for table in tables:
        for row_index, row in enumerate(table.rows):
            for column_index, value in enumerate(row):
                table_cells.append(
                    {
                        "table_id": table.table_id,
                        "row_index": row_index,
                        "column_index": column_index,
                        "column_header": table.headers[column_index],
                        "cell_value": value,
                    }
                )
    _write_csv(
        output / "table_cells.csv",
        ["table_id", "row_index", "column_index", "column_header", "cell_value"],
        table_cells,
    )


def audit(paths: Paths) -> dict[str, object]:
    corpus_dir = paths.assert_corpus_path(paths.raw_markdown)
    files = discover_documents(corpus_dir)
    titles_missing = []
    sha_counts: Counter[str] = Counter()
    total_bytes = 0
    for path in files:
        document, _ = build_document(path)
        total_bytes += document.byte_size
        sha_counts[document.sha256] += 1
        if document.title == document.source_id:
            titles_missing.append(path.name)
    return {
        "documents": len(files),
        "total_bytes": total_bytes,
        "duplicate_content_groups": sum(1 for count in sha_counts.values() if count > 1),
        "missing_titles": titles_missing,
        "input_root": str(corpus_dir),
    }
