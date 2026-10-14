from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from urllib.parse import quote

from pyshacl import validate
from rdflib import DCTERMS, RDF, SKOS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import XSD

from pension_graph.concepts import Concept
from pension_graph.models import ChunkRecord, DocumentRecord, MentionRecord
from pension_graph.tables import TableRecord

PGO = Namespace("urn:kr:pension:ontology:")
PGR = Namespace("urn:kr:pension:resource:")


def resource_uri(kind: str, identifier: str) -> URIRef:
    return URIRef(f"{PGR}{kind}/{quote(identifier, safe=':-._~')}")


def load_ontology(ontology_dir: Path) -> Graph:
    graph = Graph()
    for path in sorted(ontology_dir.glob("*.ttl")):
        if path.name == "shapes.ttl" or path.name.endswith("-shapes.ttl"):
            continue
        graph.parse(path, format="turtle")
    return graph


def build_corpus_graph(
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    concepts: list[Concept],
    mentions: list[MentionRecord],
    tables: list[TableRecord],
) -> Graph:
    graph = Graph()
    graph.bind("pgo", PGO)
    graph.bind("pgr", PGR)
    graph.bind("dcterms", DCTERMS)
    graph.bind("skos", SKOS)

    for doc in documents:
        node = resource_uri("document", doc.source_id)
        graph.add((node, RDF.type, PGO.Document))
        graph.add((node, PGO.sourceId, Literal(doc.source_id)))
        graph.add((node, DCTERMS.title, Literal(doc.title, lang="ko")))
        graph.add((node, DCTERMS.identifier, Literal(doc.filename)))
        graph.add((node, PGO.checksum, Literal(doc.sha256)))

    for concept in concepts:
        node = resource_uri("concept", concept.concept_id)
        graph.add((node, RDF.type, PGO.DomainConcept))
        graph.add((node, SKOS.prefLabel, Literal(concept.label, lang="ko")))
        graph.add((node, PGO.denotesClass, PGO[concept.ontology_class]))
        for alias in concept.aliases:
            if alias != concept.label:
                graph.add((node, SKOS.altLabel, Literal(alias, lang="ko")))

    chunks_by_document: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
        node = resource_uri("chunk", chunk.chunk_id)
        graph.add((node, RDF.type, PGO.TextChunk))
        graph.add((node, PGO.chunkId, Literal(chunk.chunk_id)))
        graph.add((node, PGO.content, Literal(chunk.content)))
        graph.add((node, PGO.chunkIndex, Literal(chunk.index, datatype=XSD.nonNegativeInteger)))
        graph.add((node, PGO.startLine, Literal(chunk.start_line, datatype=XSD.positiveInteger)))
        graph.add((node, PGO.endLine, Literal(chunk.end_line, datatype=XSD.positiveInteger)))
        graph.add((node, PGO.checksum, Literal(chunk.sha256)))
        graph.add((node, PGO.partOfDocument, resource_uri("document", chunk.source_id)))
        if chunk.heading_path:
            graph.add((node, PGO.headingPath, Literal(" > ".join(chunk.heading_path))))

    for doc_chunks in chunks_by_document.values():
        ordered = sorted(doc_chunks, key=lambda item: item.index)
        for current, following in pairwise(ordered):
            graph.add(
                (
                    resource_uri("chunk", current.chunk_id),
                    PGO.nextChunk,
                    resource_uri("chunk", following.chunk_id),
                )
            )

    mention_map = {(item.chunk_id, item.concept_id): item for item in mentions}
    for (chunk_id, concept_id), mention in mention_map.items():
        chunk_node = resource_uri("chunk", chunk_id)
        concept_node = resource_uri("concept", concept_id)
        graph.add((chunk_node, PGO.mentions, concept_node))
        # RDF-star portability is uneven, so mention metadata is a first-class resource.
        mention_node = resource_uri("mention", mention.mention_id)
        graph.add((mention_node, RDF.type, PGO.EvidenceSpan))
        graph.add((mention_node, PGO.inChunk, chunk_node))
        graph.add((mention_node, PGO.quote, Literal(mention.matched_text)))
        graph.add((mention_node, PGO.matchedText, Literal(mention.matched_text)))
        graph.add(
            (mention_node, PGO.mentionCount, Literal(mention.count, datatype=XSD.positiveInteger))
        )
        graph.add((mention_node, PGO.denotesClass, concept_node))

    for table in tables:
        table_node = resource_uri("table", table.table_id)
        graph.add((table_node, RDF.type, PGO.DataTable))
        graph.add((table_node, PGO.tableId, Literal(table.table_id)))
        graph.add((table_node, PGO.rawTable, Literal(table.raw_text)))
        graph.add((table_node, PGO.inSourceChunk, resource_uri("chunk", table.chunk_id)))
        for row_index, row in enumerate(table.rows):
            row_node = resource_uri("table-row", f"{table.table_id}:row:{row_index}")
            graph.add((row_node, RDF.type, PGO.TableRow))
            graph.add((row_node, PGO.rowIndex, Literal(row_index, datatype=XSD.nonNegativeInteger)))
            graph.add((table_node, PGO.hasRow, row_node))
            for column_index, value in enumerate(row):
                cell_node = resource_uri(
                    "table-cell", f"{table.table_id}:row:{row_index}:cell:{column_index}"
                )
                graph.add((cell_node, RDF.type, PGO.TableCell))
                graph.add(
                    (
                        cell_node,
                        PGO.columnIndex,
                        Literal(column_index, datatype=XSD.nonNegativeInteger),
                    )
                )
                graph.add((cell_node, PGO.columnHeader, Literal(table.headers[column_index])))
                graph.add((cell_node, PGO.cellValue, Literal(value)))
                graph.add((row_node, PGO.hasCell, cell_node))
    return graph


def validate_graph(data_graph: Graph, ontology_dir: Path) -> tuple[bool, str]:
    shapes = Graph().parse(ontology_dir / "shapes.ttl", format="turtle")
    ontology = load_ontology(ontology_dir)
    conforms, _, report = validate(
        data_graph,
        shacl_graph=shapes,
        ont_graph=ontology,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    return bool(conforms), str(report)
