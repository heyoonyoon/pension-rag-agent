from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any

from pension_graph.deadline import check_deadline
from pension_graph.retrieval import expand_chunk_context, retrieve_graph_context
from pension_graph.vector_store import VectorSettings, query_hybrid_vectors


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[dict[str, Any]]],
    *,
    key: str = "chunk_id",
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Fuse retrievers by rank, avoiding incomparable raw-score arithmetic."""
    scores: dict[str, float] = defaultdict(float)
    merged: dict[str, dict[str, Any]] = {}
    channels: dict[str, set[str]] = defaultdict(set)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for channel, rows in ranked_lists.items():
        seen: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            item_key = str(row[key])
            if item_key in seen:
                continue
            seen.add(item_key)
            scores[item_key] += 1.0 / (rank_constant + rank)
            channels[item_key].add(channel)
            ranks[item_key][channel] = rank
            merged.setdefault(item_key, {}).update(row)
    result = []
    for item_key, score in scores.items():
        row = dict(merged[item_key])
        row["fusion_score"] = score
        row["retrieval_channels"] = sorted(channels[item_key])
        row["channel_ranks"] = ranks[item_key]
        result.append(row)
    return sorted(result, key=lambda row: (-row["fusion_score"], row[key]))


def retrieve_hybrid_context(
    *,
    query: str,
    qdrant_settings: VectorSettings,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    limit: int = 10,
) -> dict[str, Any]:
    check_deadline()
    total_started = perf_counter()
    candidate_limit = max(limit * 3, 30)
    vector_started = perf_counter()
    vector_rows = query_hybrid_vectors(
        query,
        settings=qdrant_settings,
        limit=candidate_limit,
    )
    check_deadline()
    vector_ms = (perf_counter() - vector_started) * 1000
    graph_started = perf_counter()
    graph_rows = retrieve_graph_context(
        query=query,
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        limit=candidate_limit,
    )
    check_deadline()
    graph_search_ms = (perf_counter() - graph_started) * 1000
    fused = reciprocal_rank_fusion({"vector_rag": vector_rows, "graph_rag": graph_rows})[
        :limit
    ]
    expand_started = perf_counter()
    context_by_chunk = {
        row["chunk_id"]: row
        for row in expand_chunk_context(
            [str(row["chunk_id"]) for row in fused],
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )
    }
    check_deadline()
    graph_expand_ms = (perf_counter() - expand_started) * 1000
    for row in fused:
        check_deadline()
        graph = context_by_chunk.get(str(row["chunk_id"]), {})
        row["title"] = graph.get("title", row.get("title", ""))
        # Keep deterministic table-range facts produced during graph search;
        # graph expansion enriches them but must never overwrite them.
        row["graph_context"] = list(
            dict.fromkeys(
                [
                    *[str(value) for value in row.get("graph_context", [])],
                    *[str(value) for value in graph.get("graph_context", [])],
                ]
            )
        )
        row["source"] = {
            "source_id": row["source_id"],
            "chunk_id": row["chunk_id"],
            "heading_path": row.get("heading_path", []),
        }
    check_deadline()
    return {
        "query": query,
        "results": fused,
        "diagnostics": {
            "vector_candidates": len(vector_rows),
            "graph_candidates": len(graph_rows),
            "returned": len(fused),
            "fusion": "application-level RRF(vector hybrid, graph retrieval)",
            "vector_fusion": "Qdrant RRF(dense, sparse)",
            "timings_ms": {
                "vector": round(vector_ms, 3),
                "graph_search": round(graph_search_ms, 3),
                "graph_expand": round(graph_expand_ms, 3),
                "total": round((perf_counter() - total_started) * 1000, 3),
            },
        },
    }
