from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from rdflib import RDF, RDFS, SKOS, Graph, URIRef
from rdflib.namespace import OWL

from pension_graph.models import ChunkRecord, MentionRecord


@dataclass(frozen=True)
class Concept:
    concept_id: str
    label: str
    ontology_class: str
    aliases: tuple[str, ...]


@lru_cache(maxsize=8)
def load_concepts(path: Path) -> list[Concept]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    concepts: list[Concept] = []
    seen: set[str] = set()
    for item in payload.get("concepts", []):
        concept_id = str(item["id"])
        if concept_id in seen:
            raise ValueError(f"중복 concept id: {concept_id}")
        seen.add(concept_id)
        aliases = tuple(dict.fromkeys([str(item["label"]), *map(str, item["aliases"])]))
        concepts.append(
            Concept(
                concept_id=concept_id,
                label=str(item["label"]),
                ontology_class=str(item["ontology_class"]),
                aliases=aliases,
            )
        )
    return concepts


@lru_cache(maxsize=8)
def load_query_concepts(concept_path: Path, ontology_dir: Path) -> list[Concept]:
    """Merge curated aliases with every queryable term declared in the ontology."""
    controlled = load_concepts(concept_path)
    by_class = {concept.ontology_class: concept for concept in controlled}

    graph = Graph()
    for path in sorted(ontology_dir.glob("*.ttl")):
        graph.parse(path, format="turtle")

    queryable_types = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty)
    subjects = {
        subject
        for queryable_type in queryable_types
        for subject in graph.subjects(RDF.type, queryable_type)
        if isinstance(subject, URIRef)
    }
    for subject in sorted(subjects, key=str):
        ontology_class = _local_name(subject)
        labels = [str(value) for value in graph.objects(subject, RDFS.label)]
        alternate = [str(value) for value in graph.objects(subject, SKOS.altLabel)]
        aliases = tuple(dict.fromkeys([*labels, *alternate]))
        if not aliases:
            continue
        existing = by_class.get(ontology_class)
        if existing is not None:
            by_class[ontology_class] = Concept(
                concept_id=existing.concept_id,
                label=existing.label,
                ontology_class=existing.ontology_class,
                aliases=tuple(dict.fromkeys([*existing.aliases, *aliases])),
            )
            continue
        by_class[ontology_class] = Concept(
            concept_id=f"ontology:{ontology_class}",
            label=labels[0] if labels else aliases[0],
            ontology_class=ontology_class,
            aliases=aliases,
        )
    return list(by_class.values())


def _local_name(uri: URIRef) -> str:
    value = str(uri)
    return value.rsplit("#", 1)[-1].rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias)
    # ASCII acronyms need token boundaries; Korean terms are matched as substrings.
    if alias.isascii() and alias.replace("-", "").isalnum():
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def find_mentions(chunks: list[ChunkRecord], concepts: list[Concept]) -> list[MentionRecord]:
    results: list[MentionRecord] = []
    for chunk in chunks:
        for concept in concepts:
            matches: list[str] = []
            for alias in sorted(concept.aliases, key=len, reverse=True):
                matches.extend(
                    match.group(0) for match in _alias_pattern(alias).finditer(chunk.content)
                )
            if not matches:
                continue
            # Keep one relation per chunk/concept while preserving representative source spelling.
            canonical_match = max(matches, key=len)
            raw_id = f"{chunk.chunk_id}|{concept.concept_id}"
            mention_id = "mention:" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
            results.append(
                MentionRecord(
                    mention_id=mention_id,
                    chunk_id=chunk.chunk_id,
                    concept_id=concept.concept_id,
                    matched_text=canonical_match,
                    count=len(matches),
                )
            )
    return results
