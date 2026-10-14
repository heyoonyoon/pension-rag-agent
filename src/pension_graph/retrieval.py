from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from neo4j import GraphDatabase, Query

from pension_graph.deadline import check_deadline, get_deadline, remaining_timeout
from pension_graph.structured_rules import (
    MEASURE_VALUE,
    row_matches_query_range,
)

LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
QUERY_TOKEN = re.compile(r"[0-9A-Za-z가-힣.%]+")
QUERY_STOPWORDS = {
    "무엇",
    "뭐",
    "뭐가",
    "어떤",
    "어떻게",
    "알려줘",
    "알려주세요",
    "궁금해요",
    "뭔지",
    "뜻",
    "정의",
    "개념",
    "설명",
    "설명해줘",
    "설명해주세요",
    "말해줘",
    "말해주세요",
}
KOREAN_QUERY_SUFFIXES = (
    "인가요",
    "하나요",
    "되나요",
    "하면",
    "해요",
    "나요",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에서",
    "으로",
    "에",
    "로",
    "의",
    "과",
    "와",
)


@dataclass(frozen=True)
class GraphEvidence:
    chunk_id: str
    source_id: str
    title: str
    heading_path: list[str] | str
    content: str
    graph_context: list[str]
    score: float


def _driver_deadline_config() -> dict[str, float]:
    check_deadline()
    if get_deadline() is None:
        return {}
    return {
        "connection_timeout": remaining_timeout(30),
        "connection_acquisition_timeout": remaining_timeout(60),
        # A managed retry could reuse an old transaction timeout after the request
        # has expired. The outer Agent owns bounded retries for API requests.
        "max_transaction_retry_time": 0.0,
    }


def _deadline_query(cypher: str) -> str | Query:
    check_deadline()
    if get_deadline() is None:
        return cypher
    return Query(cypher, timeout=remaining_timeout(270))


def escape_fulltext_query(query: str) -> str:
    cleaned = LUCENE_SPECIAL.sub(r"\\\1", query).strip()
    return cleaned or '""'


def graph_query_terms(query: str) -> str:
    """Keep content-bearing terms so Korean question endings do not dominate BM25."""
    terms: list[str] = []
    for raw in QUERY_TOKEN.findall(query):
        if raw in QUERY_STOPWORDS:
            continue
        term = raw
        for suffix in KOREAN_QUERY_SUFFIXES:
            if term.endswith(suffix) and len(term) > len(suffix):
                term = term[: -len(suffix)]
                break
        if len(term) < 2 or term in QUERY_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return " ".join(terms) or query


def table_row_matches_query_range(query: str, row_text: str) -> bool:
    """Compatibility wrapper for the shared typed interval engine."""
    return row_matches_query_range(query, row_text)


def table_row_semantic_score(query: str, row_text: str) -> int:
    """Prefer range-matching rows whose labels also overlap the question semantics."""
    terms = [
        term.casefold()
        for term in graph_query_terms(query).split()
        if not MEASURE_VALUE.fullmatch(term) and not term.isdigit()
    ]
    normalized_row = row_text.casefold()
    headers = " ".join(
        cell.split(":", 1)[0].casefold()
        for cell in row_text.split(" | ")
        if ":" in cell
    )
    return sum(
        (100 + len(term)) if term in headers else len(term)
        for term in terms
        if term in normalized_row
    )


def retrieve_graph_context(
    *,
    query: str,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    limit: int = 10,
) -> list[dict[str, object]]:
    """Fuse lexical chunks with active semantic entities, then expand to evidence."""
    cypher = """
    CALL () {
      CALL db.index.fulltext.queryNodes(
        'chunk_fulltext', $query, {limit: $candidate_limit}
      ) YIELD node, score
      RETURN node, score, 'lexical' AS channel
      UNION ALL
      CALL db.index.fulltext.queryNodes(
        'semantic_entity_fulltext', $query, {limit: $entity_limit}
      ) YIELD node AS entity, score
      MATCH (claim:SemanticClaim {active: true})-[:SUBJECT|OBJECT]->(entity)
      MATCH (claim)-[:SUPPORTED_BY]->(node:Chunk)
      RETURN node, score * 0.85 AS score, 'semantic' AS channel
    }
    WITH node, max(score) AS score, collect(DISTINCT channel) AS channels
    MATCH (node)-[:PART_OF_DOCUMENT]->(doc:Document)
    OPTIONAL MATCH (node)-[:MENTIONS]->(concept:Concept)
    OPTIONAL MATCH (node)<-[:IN_SOURCE_CHUNK]-(table:DataTable)
    OPTIONAL MATCH (claim:SemanticClaim {active: true})-[:SUPPORTED_BY]->(node)
    OPTIONAL MATCH (claim)-[:SUBJECT]->(subject:SemanticEntity)
    OPTIONAL MATCH (claim)-[:OBJECT]->(object:SemanticEntity)
    WITH node, doc, score, channels,
         collect(DISTINCT concept.label)[0..20] AS concepts,
         collect(DISTINCT table.table_id)[0..10] AS table_ids,
         collect(DISTINCT CASE WHEN claim IS NULL THEN NULL ELSE
           coalesce(subject.name, '?') + ' -[' + claim.predicate + ']-> ' +
           coalesce(object.name, '?') + ' | evidence: ' + claim.evidence_quote
         END)[0..30] AS semantic_claims
    OPTIONAL MATCH (previous:Chunk)-[:NEXT_CHUNK]->(node)
    OPTIONAL MATCH (node)-[:NEXT_CHUNK]->(following:Chunk)
    RETURN node.chunk_id AS chunk_id,
           node.source_id AS source_id,
           doc.title AS title,
           node.heading_path AS heading_path,
           node.content AS content,
           ['channels:' + reduce(value = '', item IN channels |
             value + CASE WHEN value = '' THEN '' ELSE ',' END + item)] +
           concepts + table_ids + semantic_claims +
             CASE WHEN previous IS NULL THEN [] ELSE ['previous:' + previous.chunk_id] END +
             CASE WHEN following IS NULL THEN [] ELSE ['next:' + following.chunk_id] END
             AS graph_context,
           score
    ORDER BY score DESC
    LIMIT $limit
    """
    check_deadline()
    with GraphDatabase.driver(uri, auth=(user, password), **_driver_deadline_config()) as driver:
        driver.verify_connectivity()
        check_deadline()
        records, _, _ = driver.execute_query(
            _deadline_query(cypher),
            query=escape_fulltext_query(graph_query_terms(query)),
            candidate_limit=max(limit * 3, 30),
            entity_limit=max(limit * 2, 20),
            limit=limit,
            database_=database,
        )
        check_deadline()
        table_records = []
        if MEASURE_VALUE.search(query):
            table_records, _, _ = driver.execute_query(
                _deadline_query("""
                MATCH (table:DataTable)-[:HAS_ROW]->(row:TableRow)-[:HAS_CELL]->(cell:TableCell)
                MATCH (table)-[:IN_SOURCE_CHUNK]->(node:Chunk)
                      -[:PART_OF_DOCUMENT]->(doc:Document)
                WITH table, row, node, doc, cell ORDER BY cell.column_index
                RETURN table.table_id AS table_id,
                       row.row_index AS row_index,
                       node.chunk_id AS chunk_id,
                       node.source_id AS source_id,
                       doc.title AS title,
                       node.heading_path AS heading_path,
                       node.content AS content,
                       collect(cell.column_header + ': ' + cell.value) AS cells
                """),
                database_=database,
            )
            check_deadline()
    rows = [asdict(GraphEvidence(**dict(record))) for record in records]
    by_chunk = {str(row["chunk_id"]): row for row in rows}
    for record in table_records:
        check_deadline()
        record_data = dict(record)
        row_text = " | ".join(str(value) for value in record_data["cells"])
        if not table_row_matches_query_range(query, row_text):
            continue
        chunk_id = str(record_data["chunk_id"])
        table_hint = (
            f"table_range_match:{record_data['table_id']}:row:{record_data['row_index']} | "
            f"{row_text}"
        )
        existing = by_chunk.get(chunk_id)
        deterministic_score = 1_000_000.0 + 1_000.0 * table_row_semantic_score(
            query, row_text
        )
        if existing is not None:
            existing["score"] = max(float(existing["score"]), deterministic_score)
            existing["graph_context"] = list(
                dict.fromkeys([table_hint, *existing["graph_context"]])
            )
            continue
        by_chunk[chunk_id] = asdict(
            GraphEvidence(
                chunk_id=chunk_id,
                source_id=str(record_data["source_id"]),
                title=str(record_data["title"]),
                heading_path=record_data["heading_path"],
                content=str(record_data["content"]),
                graph_context=[table_hint],
                score=deterministic_score,
            )
        )
    check_deadline()
    return sorted(
        by_chunk.values(), key=lambda row: (-float(row["score"]), str(row["chunk_id"]))
    )[:limit]


def expand_chunk_context(
    chunk_ids: list[str],
    *,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
) -> list[dict[str, object]]:
    """Attach provenance-preserving graph facts to already selected vector chunks."""
    check_deadline()
    if not chunk_ids:
        return []
    cypher = """
    UNWIND range(0, size($chunk_ids) - 1) AS requested_rank
    WITH requested_rank, $chunk_ids[requested_rank] AS requested_id
    MATCH (node:Chunk {chunk_id: requested_id})-[:PART_OF_DOCUMENT]->(doc:Document)
    OPTIONAL MATCH (node)-[:MENTIONS]->(concept:Concept)
    OPTIONAL MATCH (node)<-[:IN_SOURCE_CHUNK]-(table:DataTable)
    OPTIONAL MATCH (claim:SemanticClaim {active: true})-[:SUPPORTED_BY]->(node)
    OPTIONAL MATCH (claim)-[:SUBJECT]->(subject:SemanticEntity)
    OPTIONAL MATCH (claim)-[:OBJECT]->(object:SemanticEntity)
    WITH requested_rank, requested_id, node, doc,
         collect(DISTINCT concept.label)[0..20] AS concepts,
         collect(DISTINCT table.table_id)[0..10] AS table_ids,
         collect(DISTINCT CASE WHEN claim IS NULL THEN NULL ELSE
           coalesce(subject.name, '?') + ' -[' + claim.predicate + ']-> ' +
           coalesce(object.name, '?') + ' | evidence: ' + claim.evidence_quote
         END)[0..30] AS semantic_claims
    RETURN node.chunk_id AS chunk_id,
           node.source_id AS source_id,
           doc.title AS title,
           node.heading_path AS heading_path,
           node.content AS content,
           concepts + table_ids + semantic_claims AS graph_context
    ORDER BY requested_rank
    """
    with GraphDatabase.driver(uri, auth=(user, password), **_driver_deadline_config()) as driver:
        driver.verify_connectivity()
        check_deadline()
        records, _, _ = driver.execute_query(
            _deadline_query(cypher), chunk_ids=chunk_ids, database_=database
        )
        check_deadline()
    check_deadline()
    return [dict(record) for record in records]
