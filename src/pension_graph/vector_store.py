from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from pension_graph.config import Paths
from pension_graph.corpus import discover_documents
from pension_graph.deadline import RequestDeadlineExceeded, check_deadline, remaining_timeout

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
POINT_NAMESPACE = uuid.UUID("82f6d28f-7918-44da-a974-3df214b66d0e")


@dataclass(frozen=True)
class VectorSettings:
    url: str = "http://localhost:6335"
    api_key: str | None = None
    collection: str = "pension_chunks_qwen3_v1"
    dense_url: str = "http://localhost:8081"
    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    dense_revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    dense_dimension: int = 1024
    sparse_model: str = "Qdrant/bm25"
    cache_dir: str = ".cache/fastembed"
    batch_size: int = 4
    segment_chars: int = 1400
    segment_overlap: int = 180
    query_instruction: str = (
        "Given a Korean pension question, retrieve source passages containing the facts, "
        "conditions, exceptions, amounts, taxes, eligibility, or procedures needed to answer it."
    )

    @classmethod
    def from_env(cls) -> VectorSettings:
        return cls(
            url=os.getenv("QDRANT_URL", cls.url),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection=os.getenv("QDRANT_COLLECTION", cls.collection),
            dense_url=os.getenv("EMBEDDING_DENSE_URL", cls.dense_url),
            dense_model=os.getenv("EMBEDDING_DENSE_MODEL", cls.dense_model),
            dense_revision=os.getenv("EMBEDDING_DENSE_REVISION", cls.dense_revision),
            dense_dimension=int(
                os.getenv("EMBEDDING_DENSE_DIMENSION", str(cls.dense_dimension))
            ),
            sparse_model=os.getenv("EMBEDDING_SPARSE_MODEL", cls.sparse_model),
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR", cls.cache_dir),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", str(cls.batch_size))),
            segment_chars=int(os.getenv("EMBEDDING_SEGMENT_CHARS", str(cls.segment_chars))),
            segment_overlap=int(
                os.getenv("EMBEDDING_SEGMENT_OVERLAP", str(cls.segment_overlap))
            ),
            query_instruction=os.getenv(
                "EMBEDDING_QUERY_INSTRUCTION", cls.query_instruction
            ),
        )


@dataclass(frozen=True)
class VectorSegment:
    segment_id: str
    chunk_id: str
    document_id: str
    source_id: str
    title: str
    chunk_index: int
    segment_index: int
    heading_path: list[str]
    content: str
    segment_content: str
    chunk_sha256: str
    segment_sha256: str
    segment_kind: str = "chunk"
    table_id: str | None = None
    table_row_index: int | None = None

    def embedding_text(self) -> str:
        heading = " > ".join(self.heading_path)
        prefix = f"문서 제목: {self.title}\n제목 경로: {heading}\n" if heading else ""
        return prefix + self.segment_content

    def payload(
        self, *, dense_model: str, dense_revision: str, sparse_model: str
    ) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "dense_model": dense_model,
                "dense_revision": dense_revision,
                "sparse_model": sparse_model,
                "input_scope": "raw_markdown/doc*.md",
            }
        )
        return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_scoped_chunks(paths: Paths) -> list[dict[str, Any]]:
    """Load generated chunks only after proving every chunk exists in raw_markdown/doc*.md."""
    corpus_dir = paths.assert_corpus_path(paths.raw_markdown)
    allowed = {path.stem: path.read_text(encoding="utf-8") for path in discover_documents(corpus_dir)}
    documents = {
        str(row["source_id"]): str(row["title"])
        for row in _read_jsonl(paths.processed / "documents.jsonl")
    }
    chunks = _read_jsonl(paths.processed / "chunks.jsonl")
    for chunk in chunks:
        source_id = str(chunk["source_id"])
        content = str(chunk["content"])
        if source_id not in allowed:
            raise ValueError(f"raw_markdown 범위 밖 source_id: {source_id}")
        if content not in allowed[source_id]:
            raise ValueError(f"원문에서 찾을 수 없는 청크: {chunk['chunk_id']}")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != chunk["sha256"]:
            raise ValueError(f"청크 해시 불일치: {chunk['chunk_id']}")
        if source_id not in documents:
            raise ValueError(f"문서 메타데이터가 없는 청크: {chunk['chunk_id']}")
        chunk["_document_title"] = documents[source_id]
    return chunks


def load_scoped_tables(
    paths: Paths, chunks: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Load only table rows whose raw table is an exact substring of a validated chunk."""
    table_path = paths.processed / "tables.jsonl"
    if not table_path.exists():
        return []
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    tables = _read_jsonl(table_path)
    for table in tables:
        chunk_id = str(table["chunk_id"])
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"원문 청크가 없는 표: {table['table_id']}")
        if str(table["source_id"]) != str(chunk["source_id"]):
            raise ValueError(f"표와 청크의 source_id 불일치: {table['table_id']}")
        if str(table["raw_text"]) not in str(chunk["content"]):
            raise ValueError(f"원문에서 찾을 수 없는 표: {table['table_id']}")
    return tables


def split_for_embedding(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Create overlapping, lossless windows while preferring newline/sentence boundaries."""
    if max_chars < 200:
        raise ValueError("max_chars는 200 이상이어야 합니다.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap은 0 이상 max_chars 미만이어야 합니다.")
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            search_from = max(start + max_chars // 2, hard_end - 280)
            candidates = [
                text.rfind("\n", search_from, hard_end),
                text.rfind(". ", search_from, hard_end),
                text.rfind("다. ", search_from, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (1 if text[boundary] == "\n" else 2)
        window = text[start:end]
        if not window:
            raise RuntimeError("임베딩 세그먼트 분할이 진행되지 않았습니다.")
        windows.append(window)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return windows


def build_segments(
    chunks: Sequence[dict[str, Any]], *, max_chars: int, overlap: int
) -> list[VectorSegment]:
    segments: list[VectorSegment] = []
    for chunk in chunks:
        parts = split_for_embedding(str(chunk["content"]), max_chars=max_chars, overlap=overlap)
        for index, part in enumerate(parts):
            digest = hashlib.sha256(part.encode("utf-8")).hexdigest()
            segment_id = f"{chunk['chunk_id']}:segment:{index:03d}:{digest[:12]}"
            segments.append(
                VectorSegment(
                    segment_id=segment_id,
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(chunk["document_id"]),
                    source_id=str(chunk["source_id"]),
                    title=str(chunk.get("_document_title", "")),
                    chunk_index=int(chunk["index"]),
                    segment_index=index,
                    heading_path=list(chunk["heading_path"]),
                    content=str(chunk["content"]),
                    segment_content=part,
                    chunk_sha256=str(chunk["sha256"]),
                    segment_sha256=digest,
                )
            )
    return segments


def build_table_segments(
    chunks: Sequence[dict[str, Any]], tables: Sequence[dict[str, Any]]
) -> list[VectorSegment]:
    """Index each table row for sparse retrieval while retaining its parent citation chunk."""
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    segments: list[VectorSegment] = []
    for table in tables:
        chunk = chunks_by_id[str(table["chunk_id"])]
        headers = [str(value) for value in table.get("headers", [])]
        for row_index, raw_row in enumerate(table.get("rows", [])):
            row = [str(value) for value in raw_row]
            cells = [
                f"{headers[index] if index < len(headers) else f'column_{index + 1}'}: {value}"
                for index, value in enumerate(row)
                if value.strip()
            ]
            if not cells:
                continue
            structured = "표의 한 행\n" + "\n".join(cells)
            digest = hashlib.sha256(structured.encode("utf-8")).hexdigest()
            table_id = str(table["table_id"])
            segment_id = f"{table_id}:row:{row_index:03d}:{digest[:12]}"
            segments.append(
                VectorSegment(
                    segment_id=segment_id,
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(chunk["document_id"]),
                    source_id=str(chunk["source_id"]),
                    title=str(chunk.get("_document_title", "")),
                    chunk_index=int(chunk["index"]),
                    segment_index=row_index,
                    heading_path=list(chunk["heading_path"]),
                    content=str(chunk["content"]),
                    segment_content=structured,
                    chunk_sha256=str(chunk["sha256"]),
                    segment_sha256=digest,
                    segment_kind="table_row",
                    table_id=table_id,
                    table_row_index=row_index,
                )
            )
    return segments


def _batches(items: Sequence[VectorSegment], size: int) -> Iterator[Sequence[VectorSegment]]:
    if size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("임베딩 모델이 영벡터를 반환했습니다.")
    return [value / norm for value in vector]


def format_qwen_query(query: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: {query}"


class EmbeddingModels:
    def __init__(self, settings: VectorSettings) -> None:
        check_deadline()
        from fastembed import SparseTextEmbedding

        cache_dir = str(Path(settings.cache_dir).expanduser())
        self.settings = settings
        self.http = httpx.Client(base_url=settings.dense_url, timeout=300)
        self.sparse = SparseTextEmbedding(model_name=settings.sparse_model, cache_dir=cache_dir)
        check_deadline()

    def dimension(self) -> int:
        return self.settings.dense_dimension

    def _dense(self, texts: list[str]) -> list[list[float]]:
        check_deadline()
        response = self.http.post(
            "/embed", json={"inputs": texts, "truncate": True},
            timeout=remaining_timeout(300),
        )
        check_deadline()
        response.raise_for_status()
        vectors = response.json()
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("TEI 임베딩 응답 개수가 입력과 다릅니다.")
        normalized = [_normalized([float(value) for value in vector]) for vector in vectors]
        if any(len(vector) != self.settings.dense_dimension for vector in normalized):
            raise ValueError("TEI 임베딩 차원이 설정과 다릅니다.")
        check_deadline()
        return normalized

    def passages(self, texts: Iterable[str]) -> list[list[float]]:
        # Qwen3 documents are intentionally embedded without a query instruction.
        return self._dense(list(texts))

    def query(self, text: str) -> list[float]:
        instructed = format_qwen_query(text, self.settings.query_instruction)
        return self._dense([instructed])[0]

    def sparse_documents(self, texts: Iterable[str]) -> list[models.SparseVector]:
        check_deadline()
        result = []
        for vector in self.sparse.embed(list(texts)):
            check_deadline()
            result.append(models.SparseVector(
                indices=vector.indices.tolist(), values=vector.values.tolist(),
            ))
        check_deadline()
        return result

    def sparse_query(self, text: str) -> models.SparseVector:
        check_deadline()
        vector = next(iter(self.sparse.query_embed(text)))
        check_deadline()
        return models.SparseVector(
            indices=vector.indices.tolist(), values=vector.values.tolist()
        )


def make_client(settings: VectorSettings) -> QdrantClient:
    check_deadline()
    # Keep the reusable client's baseline stable; query_points receives the current
    # request's timeout below, rather than inheriting a previous request's budget.
    # Recent clients otherwise spawn a daemon thread for a version probe, outside
    # the request context. Validate version compatibility during deployment instead.
    return QdrantClient(
        url=settings.url, api_key=settings.api_key, timeout=60, check_compatibility=False,
    )


def qdrant_query_timeout() -> int:
    """Qdrant search timeout is whole seconds >= 1; rounding up would exceed budget."""
    timeout = math.floor(remaining_timeout(60))
    if timeout < 1:
        raise RequestDeadlineExceeded("Not enough request budget for a Qdrant query")
    return timeout


def _ensure_collection(
    client: QdrantClient,
    settings: VectorSettings,
    *,
    dimension: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(settings.collection)
    if exists and recreate:
        client.delete_collection(settings.collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=settings.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        client.create_payload_index(
            collection_name=settings.collection,
            field_name="chunk_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        client.create_payload_index(
            collection_name=settings.collection,
            field_name="source_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        return

    info = client.get_collection(settings.collection)
    vector_config = info.config.params.vectors
    configured = vector_config.get(DENSE_VECTOR) if isinstance(vector_config, dict) else None
    sparse_config = info.config.params.sparse_vectors or {}
    if configured is None or configured.size != dimension or SPARSE_VECTOR not in sparse_config:
        raise ValueError(
            "기존 Qdrant 컬렉션 스키마가 현재 모델과 다릅니다. "
            "명시적으로 --recreate를 사용하거나 새 컬렉션 이름을 지정하세요."
        )


def _point_id(segment_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, segment_id))


def _payload_matches_segment(
    payload: dict[str, Any], segment: VectorSegment, settings: VectorSettings
) -> bool:
    expected = {
        "segment_id": segment.segment_id,
        "chunk_id": segment.chunk_id,
        "source_id": segment.source_id,
        "chunk_sha256": segment.chunk_sha256,
        "segment_sha256": segment.segment_sha256,
        "segment_kind": segment.segment_kind,
        "table_id": segment.table_id,
        "table_row_index": segment.table_row_index,
        "dense_model": settings.dense_model,
        "dense_revision": settings.dense_revision,
        "sparse_model": settings.sparse_model,
        "input_scope": "raw_markdown/doc*.md",
    }
    return all(payload.get(key) == value for key, value in expected.items())


def _valid_existing_points(
    client: QdrantClient,
    settings: VectorSettings,
    segments: Sequence[VectorSegment],
) -> set[str]:
    expected = {_point_id(segment.segment_id): segment for segment in segments}
    valid: set[str] = set()
    expected_ids = list(expected)
    for offset in range(0, len(expected_ids), 100):
        records = client.retrieve(
            collection_name=settings.collection,
            ids=expected_ids[offset : offset + 100],
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            point_id = str(record.id)
            segment = expected.get(point_id)
            if segment is not None and _payload_matches_segment(
                record.payload or {}, segment, settings
            ):
                valid.add(point_id)
    return valid


def _delete_stale_points(
    client: QdrantClient,
    settings: VectorSettings,
    segments: Sequence[VectorSegment],
) -> int:
    """Remove points no longer derived from the current validated corpus."""
    expected_ids = {_point_id(segment.segment_id) for segment in segments}
    stale_ids: list[str | int] = []
    offset: str | int | None = None
    while True:
        records, offset = client.scroll(
            collection_name=settings.collection,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        stale_ids.extend(record.id for record in records if str(record.id) not in expected_ids)
        if offset is None:
            break
    for start in range(0, len(stale_ids), 256):
        client.delete(
            collection_name=settings.collection,
            points_selector=models.PointIdsList(points=stale_ids[start : start + 256]),
            wait=True,
        )
    return len(stale_ids)


def index_chunks(
    *,
    paths: Paths,
    settings: VectorSettings,
    recreate: bool = False,
    client: QdrantClient | None = None,
    embedder: EmbeddingModels | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    chunks = load_scoped_chunks(paths)
    tables = load_scoped_tables(paths, chunks)
    chunk_segments = build_segments(
        chunks,
        max_chars=settings.segment_chars,
        overlap=settings.segment_overlap,
    )
    table_segments = build_table_segments(chunks, tables)
    segments = [*chunk_segments, *table_segments]
    vectorizer = embedder or EmbeddingModels(settings)
    qdrant = client or make_client(settings)
    dimension = vectorizer.dimension()
    _ensure_collection(qdrant, settings, dimension=dimension, recreate=recreate)

    stale_deleted = 0 if recreate else _delete_stale_points(qdrant, settings, segments)
    reusable = set() if recreate else _valid_existing_points(qdrant, settings, segments)
    indexed = 0
    written = 0
    for batch in _batches(segments, settings.batch_size):
        pending = [
            segment
            for segment in batch
            if _point_id(segment.segment_id) not in reusable
        ]
        if not pending:
            indexed += len(batch)
            if progress is not None:
                progress(indexed, len(segments))
            continue
        texts = [segment.embedding_text() for segment in pending]
        sparse_vectors = vectorizer.sparse_documents(texts)
        dense_pending = [
            (index, segment)
            for index, segment in enumerate(pending)
            if segment.segment_kind == "chunk"
        ]
        dense_vectors = (
            vectorizer.passages([texts[index] for index, _ in dense_pending])
            if dense_pending
            else []
        )
        dense_by_index = {
            index: vector
            for (index, _), vector in zip(dense_pending, dense_vectors, strict=True)
        }
        points = [
            models.PointStruct(
                id=_point_id(segment.segment_id),
                vector={
                    **(
                        {DENSE_VECTOR: dense_by_index[index]}
                        if index in dense_by_index
                        else {}
                    ),
                    SPARSE_VECTOR: sparse,
                },
                payload=segment.payload(
                    dense_model=settings.dense_model,
                    dense_revision=settings.dense_revision,
                    sparse_model=settings.sparse_model,
                ),
            )
            for index, (segment, sparse) in enumerate(
                zip(pending, sparse_vectors, strict=True)
            )
        ]
        qdrant.upsert(collection_name=settings.collection, points=points, wait=True)
        written += len(points)
        indexed += len(batch)
        if progress is not None:
            progress(indexed, len(segments))

    info = qdrant.get_collection(settings.collection)
    manifest = {
        "input_scope": "raw_markdown/doc*.md via validated data/processed/chunks.jsonl",
        "documents": len({str(chunk["source_id"]) for chunk in chunks}),
        "chunks": len(chunks),
        "tables": len(tables),
        "segments": len(segments),
        "chunk_segments": len(chunk_segments),
        "table_row_segments": len(table_segments),
        "table_row_dense_vectors": 0,
        "indexed_segments": indexed,
        "reused_segments": len(reusable),
        "stale_segments_deleted": stale_deleted,
        "written_segments": written,
        "qdrant_points": info.points_count,
        "collection": settings.collection,
        "dense_provider": "Hugging Face Text Embeddings Inference",
        "dense_model": settings.dense_model,
        "dense_revision": settings.dense_revision,
        "dense_dimension": dimension,
        "sparse_model": settings.sparse_model,
        "segment_chars": settings.segment_chars,
        "segment_overlap": settings.segment_overlap,
        "query_instruction": settings.query_instruction,
    }
    manifest_path = paths.processed / "vector-index-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def query_hybrid_vectors(
    query: str,
    *,
    settings: VectorSettings,
    limit: int = 10,
    candidate_multiplier: int = 5,
    client: QdrantClient | None = None,
    embedder: EmbeddingModels | None = None,
) -> list[dict[str, Any]]:
    check_deadline()
    vectorizer = embedder or EmbeddingModels(settings)
    check_deadline()
    qdrant = client or make_client(settings)
    candidate_limit = max(limit * candidate_multiplier, 20)
    dense_query = vectorizer.query(query)
    check_deadline()
    sparse_query = vectorizer.sparse_query(query)
    check_deadline()
    result = qdrant.query_points(
        collection_name=settings.collection,
        prefetch=[
            models.Prefetch(
                query=dense_query, using=DENSE_VECTOR, limit=candidate_limit
            ),
            models.Prefetch(
                query=sparse_query,
                using=SPARSE_VECTOR,
                limit=candidate_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidate_limit,
        with_payload=True,
        with_vectors=False,
        timeout=qdrant_query_timeout(),
    )
    check_deadline()

    # Multiple vector windows can point to the same source chunk. Keep the best window.
    by_chunk: dict[str, dict[str, Any]] = {}
    for point in result.points:
        payload = dict(point.payload or {})
        chunk_id = str(payload["chunk_id"])
        candidate = {
            "chunk_id": chunk_id,
            "source_id": payload["source_id"],
            "heading_path": payload["heading_path"],
            "content": payload["content"],
            "matched_segment": payload["segment_content"],
            "segment_id": payload["segment_id"],
            "segment_kind": payload.get("segment_kind", "chunk"),
            "table_id": payload.get("table_id"),
            "table_row_index": payload.get("table_row_index"),
            "vector_score": float(point.score),
            "retrieval_channels": (
                ["sparse"]
                if payload.get("segment_kind") == "table_row"
                else ["dense", "sparse"]
            ),
        }
        current = by_chunk.get(chunk_id)
        if current is None or candidate["vector_score"] > current["vector_score"]:
            by_chunk[chunk_id] = candidate
    check_deadline()
    return sorted(by_chunk.values(), key=lambda item: item["vector_score"], reverse=True)[:limit]


def verify_vector_index(
    *,
    paths: Paths,
    settings: VectorSettings,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Verify every expected segment and provenance payload without recomputing vectors."""
    chunks = load_scoped_chunks(paths)
    tables = load_scoped_tables(paths, chunks)
    chunk_segments = build_segments(
        chunks, max_chars=settings.segment_chars, overlap=settings.segment_overlap
    )
    table_segments = build_table_segments(chunks, tables)
    segments = [*chunk_segments, *table_segments]
    expected = {_point_id(segment.segment_id): segment for segment in segments}
    qdrant = client or make_client(settings)
    info = qdrant.get_collection(settings.collection)
    found: set[str] = set()
    mismatches: list[str] = []
    expected_ids = list(expected)
    for offset in range(0, len(expected_ids), 100):
        records = qdrant.retrieve(
            collection_name=settings.collection,
            ids=expected_ids[offset : offset + 100],
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            point_id = str(record.id)
            found.add(point_id)
            segment = expected[point_id]
            payload = record.payload or {}
            checks = {
                "segment_id": segment.segment_id,
                "chunk_id": segment.chunk_id,
                "source_id": segment.source_id,
                "chunk_sha256": segment.chunk_sha256,
                "segment_sha256": segment.segment_sha256,
                "segment_kind": segment.segment_kind,
                "table_id": segment.table_id,
                "table_row_index": segment.table_row_index,
                "dense_model": settings.dense_model,
                "dense_revision": settings.dense_revision,
                "sparse_model": settings.sparse_model,
                "input_scope": "raw_markdown/doc*.md",
            }
            for key, value in checks.items():
                if payload.get(key) != value:
                    mismatches.append(f"{segment.segment_id}:{key}")
    missing = sorted(set(expected) - found)
    exact_count = info.points_count == len(expected)
    conforms = not missing and not mismatches and exact_count
    return {
        "conforms": conforms,
        "collection": settings.collection,
        "expected_segments": len(expected),
        "chunk_segments": len(chunk_segments),
        "table_row_segments": len(table_segments),
        "qdrant_points": info.points_count,
        "verified_segments": len(found),
        "missing_segments": len(missing),
        "payload_mismatches": len(mismatches),
        "sample_errors": (missing + mismatches)[:20],
        "input_scope": "raw_markdown/doc*.md",
    }


def create_vector_snapshot(
    *, settings: VectorSettings, client: QdrantClient | None = None
) -> dict[str, Any]:
    qdrant = client or make_client(settings)
    snapshot = qdrant.create_snapshot(collection_name=settings.collection, wait=True)
    return {
        "collection": settings.collection,
        "snapshot_name": snapshot.name,
        "created_at": str(snapshot.creation_time),
        "size_bytes": snapshot.size,
        "checksum": snapshot.checksum,
    }
