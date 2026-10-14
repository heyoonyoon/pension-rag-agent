from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from neo4j import GraphDatabase

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT document_source_id IF NOT EXISTS FOR (n:Document) REQUIRE n.source_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT concept_concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.concept_id IS UNIQUE",
    "CREATE CONSTRAINT table_table_id IF NOT EXISTS FOR (n:DataTable) REQUIRE n.table_id IS UNIQUE",
    "CREATE CONSTRAINT table_row_id IF NOT EXISTS FOR (n:TableRow) REQUIRE n.row_id IS UNIQUE",
    "CREATE CONSTRAINT table_cell_id IF NOT EXISTS FOR (n:TableCell) REQUIRE n.cell_id IS UNIQUE",
    "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.content, n.heading_path]",
    "CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS FOR (n:Concept) ON EACH [n.label, n.aliases]",
]

SEMANTIC_SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT semantic_entity_id IF NOT EXISTS FOR (n:SemanticEntity) REQUIRE n.entity_id IS UNIQUE",
    "CREATE CONSTRAINT semantic_claim_id IF NOT EXISTS FOR (n:SemanticClaim) REQUIRE n.claim_id IS UNIQUE",
    "CREATE FULLTEXT INDEX semantic_entity_fulltext IF NOT EXISTS FOR (n:SemanticEntity) ON EACH [n.name]",
    "CREATE INDEX semantic_claim_chunk IF NOT EXISTS FOR (n:SemanticClaim) ON (n.chunk_id)",
]
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_lexical_graph(
    processed_dir: Path,
    *,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
) -> dict[str, Any]:
    documents = _read_jsonl(processed_dir / "documents.jsonl")
    chunks = _read_jsonl(processed_dir / "chunks.jsonl")
    mentions = _read_jsonl(processed_dir / "mentions.jsonl")
    tables = _read_jsonl(processed_dir / "tables.jsonl")
    concepts_payload = __import__("yaml").safe_load(
        (processed_dir.parents[1] / "config" / "concepts.yaml").read_text(encoding="utf-8")
    )["concepts"]
    concepts = [
        {
            "concept_id": item["id"],
            "label": item["label"],
            "ontology_class": item["ontology_class"],
            "aliases": item["aliases"],
        }
        for item in concepts_payload
    ]

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        # Neo4j requires schema operations and data writes in separate transactions.
        for statement in SCHEMA_STATEMENTS:
            driver.execute_query(statement, database_=database)
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (d:Document {source_id: row.source_id})
            SET d += row
            """,
            rows=documents,
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c += row
            WITH c, row
            MATCH (d:Document {source_id: row.source_id})
            MERGE (c)-[:PART_OF_DOCUMENT]->(d)
            """,
            rows=chunks,
            database_=database,
        )
        table_rows = []
        table_cells = []
        for table in tables:
            for row_index, row in enumerate(table["rows"]):
                row_id = f"{table['table_id']}:row:{row_index}"
                table_rows.append(
                    {"row_id": row_id, "table_id": table["table_id"], "row_index": row_index}
                )
                for column_index, value in enumerate(row):
                    table_cells.append(
                        {
                            "cell_id": f"{row_id}:cell:{column_index}",
                            "row_id": row_id,
                            "column_index": column_index,
                            "column_header": table["headers"][column_index],
                            "value": value,
                        }
                    )
        driver.execute_query(
            """
            MATCH (t:DataTable)
            WHERE NOT t.table_id IN $table_ids
            DETACH DELETE t
            """,
            table_ids=[table["table_id"] for table in tables],
            database_=database,
        )
        driver.execute_query(
            """
            MATCH (r:TableRow)
            WHERE NOT r.row_id IN $row_ids
            DETACH DELETE r
            """,
            row_ids=[row["row_id"] for row in table_rows],
            database_=database,
        )
        driver.execute_query(
            """
            MATCH (c:TableCell)
            WHERE NOT c.cell_id IN $cell_ids
            DETACH DELETE c
            """,
            cell_ids=[cell["cell_id"] for cell in table_cells],
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (t:DataTable {table_id: row.table_id})
            SET t.source_id = row.source_id, t.table_index = row.table_index,
                t.format = row.format, t.heading_path = row.heading_path, t.raw_text = row.raw_text
            WITH t, row
            MATCH (c:Chunk {chunk_id: row.chunk_id})
            MERGE (t)-[:IN_SOURCE_CHUNK]->(c)
            """,
            rows=tables,
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (r:TableRow {row_id: row.row_id}) SET r.row_index = row.row_index
            WITH r, row MATCH (t:DataTable {table_id: row.table_id})
            MERGE (t)-[:HAS_ROW]->(r)
            """,
            rows=table_rows,
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (c:TableCell {cell_id: row.cell_id})
            SET c.column_index = row.column_index, c.column_header = row.column_header, c.value = row.value
            WITH c, row MATCH (r:TableRow {row_id: row.row_id})
            MERGE (r)-[:HAS_CELL]->(c)
            """,
            rows=table_cells,
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (c:Concept {concept_id: row.concept_id})
            SET c += row
            """,
            rows=concepts,
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (chunk:Chunk {chunk_id: row.chunk_id})
            MATCH (concept:Concept {concept_id: row.concept_id})
            MERGE (chunk)-[r:MENTIONS]->(concept)
            SET r.matched_text = row.matched_text, r.count = row.count, r.mention_id = row.mention_id
            """,
            rows=mentions,
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (current:Chunk {chunk_id: row.chunk_id})
            MATCH (following:Chunk {source_id: row.source_id, index: row.index + 1})
            MERGE (current)-[:NEXT_CHUNK]->(following)
            """,
            rows=chunks,
            database_=database,
        )
        records, _, _ = driver.execute_query(
            """
            MATCH (d:Document) WITH count(d) AS documents
            MATCH (c:Chunk) WITH documents, count(c) AS chunks
            MATCH (:Chunk)-[m:MENTIONS]->(:Concept)
            WITH documents, chunks, count(m) AS mentions
            MATCH (t:DataTable) WITH documents, chunks, mentions, count(t) AS tables
            MATCH (r:TableRow) WITH documents, chunks, mentions, tables, count(r) AS table_rows
            MATCH (c:TableCell)
            RETURN documents, chunks, mentions, tables, table_rows, count(c) AS table_cells
            """,
            database_=database,
        )
        return dict(records[0])


def _safe_identifier(value: str, allowed: set[str]) -> str:
    if value not in allowed or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"스키마에 없는 Neo4j 식별자입니다: {value}")
    return value


def _neo4j_properties(value: dict[str, Any]) -> dict[str, Any]:
    """Neo4j properties can only contain scalar values or homogeneous scalar lists."""
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "label" or not IDENTIFIER.fullmatch(str(key)):
            continue
        if (
            item is None
            or isinstance(item, (str, int, float, bool))
            or isinstance(item, list)
            and all(isinstance(element, (str, int, float, bool)) for element in item)
        ):
            result[str(key)] = item
    return result


def load_semantic_graph(
    processed_dir: Path,
    *,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    run_name: str = "direct",
) -> dict[str, Any]:
    """Promote only validator-approved claims into provenance-linked semantic edges."""
    safe_run_name = _safe_identifier(run_name, {"direct"})
    promoted_path = processed_dir / "semantic-kg" / safe_run_name / "promoted.jsonl"
    if not promoted_path.exists():
        raise FileNotFoundError(f"검증된 의미 그래프 파일이 없습니다: {promoted_path}")
    rows = _read_jsonl(promoted_path)
    schema_payload = yaml.safe_load(
        (processed_dir.parents[1] / "config" / "kg_schema.yaml").read_text(encoding="utf-8")
    )
    allowed_labels = set(schema_payload["node_types"])
    allowed_relationships = set(schema_payload["relationship_types"])

    entities: dict[str, dict[str, Any]] = {}
    claims_by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for endpoint in ("source", "target"):
            entity = dict(row[endpoint])
            label = _safe_identifier(str(entity.pop("label")), allowed_labels)
            entity_id = str(entity["entity_id"])
            entities[entity_id] = {
                "entity_id": entity_id,
                "label": label,
                "properties": _neo4j_properties(entity),
            }
        relationship_type = _safe_identifier(
            str(row["relationship_type"]), allowed_relationships
        )
        claims_by_type.setdefault(relationship_type, []).append(
            {
                "claim_id": row["claim_id"],
                "edge_key": row["edge_key"],
                "source_entity_id": row["source"]["entity_id"],
                "target_entity_id": row["target"]["entity_id"],
                "source_id": row["source_id"],
                "chunk_id": row["chunk_id"],
                "evidence_quote": row["evidence_quote"],
                "confidence": float(row["confidence"]),
                "status": row["status"],
                "predicate": relationship_type,
                "extractor_prompt_version": row["extractor_prompt_version"],
            }
        )
    active_versions = sorted({str(row["extractor_prompt_version"]) for row in rows})

    entities_by_label: dict[str, list[dict[str, Any]]] = {}
    for entity in entities.values():
        entities_by_label.setdefault(entity["label"], []).append(entity)

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        for statement in SEMANTIC_SCHEMA_STATEMENTS:
            driver.execute_query(statement, database_=database)
        for label, label_entities in entities_by_label.items():
            safe_label = _safe_identifier(label, allowed_labels)
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                MERGE (entity:SemanticEntity:`{safe_label}` {{entity_id: row.entity_id}})
                SET entity += row.properties,
                    entity.ontology_type = row.label,
                    entity.updated_at = datetime()
                """,
                rows=label_entities,
                database_=database,
            )
        for relationship_type, claims in claims_by_type.items():
            safe_type = _safe_identifier(relationship_type, allowed_relationships)
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                MATCH (source:SemanticEntity {{entity_id: row.source_entity_id}})
                MATCH (target:SemanticEntity {{entity_id: row.target_entity_id}})
                MATCH (chunk:Chunk {{chunk_id: row.chunk_id}})
                MERGE (source)-[semantic:`{safe_type}` {{edge_key: row.edge_key}}]->(target)
                SET semantic.semantic = true, semantic.active = true,
                    semantic.semantic_run = $run_name, semantic.updated_at = datetime()
                MERGE (claim:SemanticClaim {{claim_id: row.claim_id}})
                SET claim += row, claim.active = true, claim.experimental = false,
                    claim.semantic_run = $run_name, claim.updated_at = datetime()
                MERGE (claim)-[:SUBJECT]->(source)
                MERGE (claim)-[:OBJECT]->(target)
                MERGE (claim)-[:SUPPORTED_BY]->(chunk)
                WITH semantic, row
                MATCH (evidence:SemanticClaim {{edge_key: row.edge_key, status: 'validated'}})
                WITH semantic, count(evidence) AS evidence_count,
                     max(evidence.confidence) AS max_confidence
                SET semantic.evidence_count = evidence_count,
                    semantic.max_confidence = max_confidence
                """,
                rows=claims,
                run_name=safe_run_name,
                database_=database,
            )
        driver.execute_query(
            """
            MATCH (claim:SemanticClaim)
            WHERE NOT claim.extractor_prompt_version IN $active_versions
            SET claim.active = false, claim.experimental = true
            """,
            active_versions=active_versions,
            database_=database,
        )
        driver.execute_query(
            """
            MATCH ()-[relation]->() WHERE relation.semantic = true
            OPTIONAL MATCH (claim:SemanticClaim {edge_key: relation.edge_key})
            WITH relation, collect(claim) AS claims
            SET relation.active = any(
                claim IN claims WHERE claim.extractor_prompt_version IN $active_versions
            ),
            relation.semantic_run = CASE
                WHEN any(
                    claim IN claims WHERE claim.extractor_prompt_version IN $active_versions
                )
                THEN $run_name ELSE 'experiment'
            END
            """,
            active_versions=active_versions,
            run_name=safe_run_name,
            database_=database,
        )
        records, _, _ = driver.execute_query(
            """
            MATCH (entity:SemanticEntity) WITH count(entity) AS entities
            MATCH (claim:SemanticClaim) WITH entities, count(claim) AS claims
            MATCH ()-[relation]->() WHERE relation.semantic = true
            RETURN entities, claims, count(relation) AS semantic_relationships
            """,
            database_=database,
        )
        result = dict(records[0]) if records else {
            "entities": 0,
            "claims": 0,
            "semantic_relationships": 0,
        }
        result["input_claims"] = len(rows)
        result["run_name"] = safe_run_name
        return result
