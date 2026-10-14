from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from pension_graph.config import Paths
from pension_graph.direct_kg import run_direct_extraction
from pension_graph.hybrid_retrieval import retrieve_hybrid_context
from pension_graph.kg_builder import run_extract_candidates
from pension_graph.neo4j_store import load_lexical_graph, load_semantic_graph
from pension_graph.pipeline import audit as audit_pipeline
from pension_graph.pipeline import build as build_pipeline
from pension_graph.retrieval import retrieve_graph_context
from pension_graph.semantic_kg import run_staged_extraction, validate_curated_extraction
from pension_graph.vector_store import (
    VectorSettings,
    create_vector_snapshot,
    index_chunks,
    verify_vector_index,
)

app = typer.Typer(no_args_is_help=True, help="연금 온톨로지·GraphRAG 구축 도구")


@app.command()
def audit() -> None:
    """raw_markdown 입력의 범위와 기본 품질을 점검합니다."""
    typer.echo(json.dumps(audit_pipeline(Paths()), ensure_ascii=False, indent=2))


@app.command()
def build(max_chars: int = typer.Option(4000, min=500, max=12000)) -> None:
    """raw_markdown만 읽어 청크, RDF 지식베이스, Neo4j 적재 파일을 생성합니다."""
    stats = build_pipeline(Paths(), max_chars=max_chars)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("load-neo4j")
def load_neo4j(
    uri: str = typer.Option("bolt://localhost:7688", envvar="NEO4J_URI"),
    user: str = typer.Option("neo4j", envvar="NEO4J_USER"),
    password: str = typer.Option(..., envvar="NEO4J_PASSWORD", prompt=True, hide_input=True),
    database: str = typer.Option("neo4j", envvar="NEO4J_DATABASE"),
) -> None:
    """검증된 lexical graph를 Neo4j에 멱등 적재합니다."""
    stats = load_lexical_graph(
        Paths().processed,
        uri=uri,
        user=user,
        password=password,
        database=database,
    )
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("extract-kg")
def extract_kg(
    provider: str = typer.Option("auto", envvar="KG_LLM_PROVIDER"),
    model: str | None = typer.Option(None, envvar="KG_LLM_MODEL"),
    limit: int = typer.Option(0, min=0, help="0은 전체 청크, 양수는 앞에서부터 시험 실행"),
    max_concurrency: int = typer.Option(3, min=1, max=20),
) -> None:
    """공식 Neo4j KG Builder로 스키마 제한 관계 후보를 추출하고 근거를 검증합니다."""
    try:
        stats = run_extract_candidates(
            project_root=Paths().project_root,
            provider=provider,
            model=model,
            limit=limit,
            max_concurrency=max_concurrency,
        )
    except Exception as exc:
        typer.echo(f"실행 불가: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("extract-semantic-kg")
def extract_semantic_kg(
    model: str = typer.Option("qwen3:14b", envvar="KG_LLM_MODEL"),
    limit: int = typer.Option(0, min=0),
    source_id: str | None = typer.Option(None, help="예: doc20"),
    chunk_id: str | None = typer.Option(None, help="정확한 청크 ID 한 개"),
    max_concurrency: int = typer.Option(1, min=1, max=4),
    retries: int = typer.Option(2, min=0, max=5),
    resume: bool = typer.Option(True, help="검증 완료 체크포인트 재사용"),
) -> None:
    """Ollama JSON Schema로 청크별 의미 관계를 추출·검증·체크포인트합니다."""
    try:
        stats = run_staged_extraction(
            project_root=Paths().project_root,
            model=model,
            limit=limit,
            source_id=source_id,
            chunk_id=chunk_id,
            max_concurrency=max_concurrency,
            retries=retries,
            resume=resume,
        )
    except Exception as exc:
        typer.echo(f"실행 불가: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("extract-direct-kg")
def extract_direct_kg() -> None:
    """외부 LLM 없이 전체 청크를 고정 규칙과 수동 교정본으로 전수 처리합니다."""
    stats = run_direct_extraction(Paths().project_root)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("load-semantic-neo4j")
def load_semantic_neo4j(
    uri: str = typer.Option("bolt://localhost:7688", envvar="NEO4J_URI"),
    user: str = typer.Option("neo4j", envvar="NEO4J_USER"),
    password: str = typer.Option(..., envvar="NEO4J_PASSWORD", prompt=True, hide_input=True),
    database: str = typer.Option("neo4j", envvar="NEO4J_DATABASE"),
) -> None:
    """직접 전수 처리 후 검증 통과한 direct/promoted.jsonl만 승격합니다."""
    stats = load_semantic_graph(
        Paths().processed,
        uri=uri,
        user=user,
        password=password,
        database=database,
    )
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("search-graph")
def search_graph(
    query: Annotated[str, typer.Argument(help="검색할 자연어 또는 핵심어")],
    uri: str = typer.Option("bolt://localhost:7688", envvar="NEO4J_URI"),
    user: str = typer.Option("neo4j", envvar="NEO4J_USER"),
    password: str = typer.Option(..., envvar="NEO4J_PASSWORD", prompt=True, hide_input=True),
    database: str = typer.Option("neo4j", envvar="NEO4J_DATABASE"),
    limit: int = typer.Option(10, min=1, max=50),
) -> None:
    """전문검색과 활성 의미 관계를 융합해 근거 청크를 검색합니다."""
    rows = retrieve_graph_context(
        query=query,
        uri=uri,
        user=user,
        password=password,
        database=database,
        limit=limit,
    )
    typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))


@app.command("load-qdrant")
def load_qdrant(
    recreate: bool = typer.Option(
        False, help="기존 컬렉션을 삭제하고 다시 생성합니다. 모델 차원 변경 때만 사용"
    ),
) -> None:
    """검증된 raw_markdown 청크 전부를 Dense+Sparse 벡터로 Qdrant에 적재합니다."""
    last_bucket = -1

    def report(done: int, total: int) -> None:
        nonlocal last_bucket
        bucket = int(done / total * 10)
        if bucket != last_bucket or done == total:
            last_bucket = bucket
            typer.echo(f"임베딩·적재 진행: {done}/{total} ({done / total:.0%})")

    stats = index_chunks(
        paths=Paths(),
        settings=VectorSettings.from_env(),
        recreate=recreate,
        progress=report,
    )
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("search-hybrid")
def search_hybrid(
    query: Annotated[str, typer.Argument(help="검색할 자연어 질문")],
    neo4j_uri: str = typer.Option("bolt://localhost:7688", envvar="NEO4J_URI"),
    neo4j_user: str = typer.Option("neo4j", envvar="NEO4J_USER"),
    neo4j_password: str = typer.Option(
        ..., envvar="NEO4J_PASSWORD", prompt=True, hide_input=True
    ),
    neo4j_database: str = typer.Option("neo4j", envvar="NEO4J_DATABASE"),
    limit: int = typer.Option(10, min=1, max=30),
) -> None:
    """Qdrant 일반 RAG와 Neo4j GraphRAG를 RRF로 융합 검색합니다."""
    result = retrieve_hybrid_context(
        query=query,
        qdrant_settings=VectorSettings.from_env(),
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        limit=limit,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("verify-qdrant")
def verify_qdrant() -> None:
    """Qdrant의 모든 세그먼트와 raw_markdown 출처·해시·모델 정보를 검증합니다."""
    result = verify_vector_index(paths=Paths(), settings=VectorSettings.from_env())
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["conforms"]:
        raise typer.Exit(code=2)


@app.command("snapshot-qdrant")
def snapshot_qdrant() -> None:
    """클라우드 이전·백업에 쓸 Qdrant 컬렉션 스냅샷을 생성합니다."""
    result = create_vector_snapshot(settings=VectorSettings.from_env())
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("validate-curated-kg")
def validate_curated_kg(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """사람 또는 Codex가 직접 작성한 관계도 동일한 검증 관문으로 검사합니다."""
    stats = validate_curated_extraction(Paths().project_root, path)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
