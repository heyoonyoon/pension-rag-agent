from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import yaml

from pension_graph.semantic_kg import validate_candidates

EXTRACTION_PROMPT = """
/no_think
당신은 한국 연금 문서에서 지식 그래프 후보를 추출하는 정보 추출기다.
반드시 제공된 schema에 있는 노드 타입, 관계 타입, 패턴만 사용한다.

핵심 규칙:
1. 문서에 명시된 사실만 추출하고 상식이나 추측을 추가하지 않는다.
2. 서로 다른 조건, 법률 관점, 세금 관점을 합치지 않는다.
3. O/X 표기는 해당 열 제목의 관점에서만 해석한다.
4. 수치는 값과 단위를 분리하고 적용 대상 및 조건을 연결한다.
5. 모든 관계 properties에 source_id, chunk_id, evidence_quote, confidence를 넣는다.
6. evidence_quote는 입력 text에 실제로 연속해서 존재하는 짧은 원문이어야 한다.
7. 확실하지 않은 내용은 추출하지 않는다.
8. ID는 영문 소문자 kebab-case의 안정적인 의미 ID를 사용한다.
9. 최상위에는 nodes와 relationships만 둔다. name과 업무 속성은 반드시 properties 안에 둔다.

출력은 설명이나 Markdown 없이 반드시 다음 JSON 모양이어야 한다.
{{
  "nodes": [
    {{
      "id": "irp",
      "label": "PensionScheme",
      "properties": {{"id": "irp", "name": "개인형 퇴직연금"}}
    }},
    {{
      "id": "early-withdrawal",
      "label": "FinancialEvent",
      "properties": {{"id": "early-withdrawal", "name": "중도인출"}}
    }}
  ],
  "relationships": [
    {{
      "start_node_id": "irp",
      "end_node_id": "early-withdrawal",
      "type": "ALLOWS_EVENT",
      "properties": {{
        "source_id": "입력에 표시된 source_id",
        "chunk_id": "입력에 표시된 chunk_id",
        "evidence_quote": "입력에 연속해서 존재하는 정확한 원문",
        "confidence": 0.95
      }}
    }}
  ]
}}

Schema:
{schema}

Examples:
{examples}

Text:
{text}
""".strip()


def _build_schema(schema_payload: dict[str, Any]) -> Any:
    try:
        from neo4j_graphrag.components.schema import (
            GraphSchema,
            NodeType,
            Pattern,
            PropertyType,
            RelationshipType,
        )
    except ImportError as exc:  # pragma: no cover - installation guidance
        raise RuntimeError("pip install -e '.[kg-builder]'가 필요합니다") from exc

    common = [
        PropertyType(name="id", type="STRING", required=True),
        PropertyType(name="name", type="STRING", required=True),
    ]
    node_types = []
    for label, spec in schema_payload["node_types"].items():
        properties = list(common)
        if label == "Rule":
            properties.extend(
                [
                    PropertyType(name="statement", type="STRING", required=True),
                    PropertyType(name="source_id", type="STRING", required=True),
                    PropertyType(name="chunk_id", type="STRING", required=True),
                    PropertyType(name="evidence_quote", type="STRING", required=True),
                ]
            )
        elif label == "QuantifiedValue":
            properties.extend(
                [
                    PropertyType(name="value", type="FLOAT", required=True),
                    PropertyType(name="unit", type="STRING", required=True),
                ]
            )
        node_types.append(
            NodeType(
                label=label,
                description=spec["description"],
                properties=properties,
                additional_properties=False,
            )
        )

    evidence_properties = [
        PropertyType(name="source_id", type="STRING", required=True),
        PropertyType(name="chunk_id", type="STRING", required=True),
        PropertyType(name="evidence_quote", type="STRING", required=True),
        PropertyType(name="confidence", type="FLOAT", required=True),
    ]
    relationship_types = [
        RelationshipType(
            label=label,
            description="원문 근거가 필수인 연금 도메인 관계",
            properties=evidence_properties,
            additional_properties=False,
        )
        for label in schema_payload["relationship_types"]
    ]
    patterns = [
        Pattern(source=source, relationship=label, target=target)
        for label, spec in schema_payload["relationship_types"].items()
        for source in spec["from"]
        for target in spec["to"]
    ]
    return GraphSchema(
        node_types=tuple(node_types),
        relationship_types=tuple(relationship_types),
        patterns=tuple(patterns),
        additional_node_types=False,
        additional_relationship_types=False,
        additional_patterns=False,
    )


async def extract_candidates(
    project_root: Path,
    *,
    provider: str = "auto",
    model: str | None = None,
    limit: int = 0,
    max_concurrency: int = 3,
) -> dict[str, int | str]:
    try:
        from neo4j_graphrag.components.entity_relation_extractor import LLMEntityRelationExtractor
        from neo4j_graphrag.components.types import TextChunk, TextChunks
        from neo4j_graphrag.llm import OllamaLLM, OpenAILLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install -e '.[kg-builder]'가 필요합니다") from exc

    processed = project_root / "data" / "processed"
    chunks = [
        json.loads(line)
        for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if limit > 0:
        chunks = chunks[:limit]
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    schema_payload = yaml.safe_load(
        (project_root / "config" / "kg_schema.yaml").read_text(encoding="utf-8")
    )
    graph_schema = _build_schema(schema_payload)
    selected_provider = provider
    if selected_provider == "auto":
        selected_provider = "openai" if os.getenv("OPENAI_API_KEY") else "ollama"
    if selected_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("provider=openai지만 OPENAI_API_KEY가 설정되지 않았습니다")
        selected_model = model or "gpt-5.6-sol"
        llm = OpenAILLM(
            model_name=selected_model,
            model_params={"reasoning_effort": "high", "max_completion_tokens": 16000},
        )
        structured_output = True
    elif selected_provider == "ollama":
        selected_model = model or "qwen3:14b"
        llm = OllamaLLM(
            model_name=selected_model,
            model_params={"options": {}},
        )
        # neo4j-graphrag 1.19 sync/async compatibility bug nests options twice.
        # The async path expects the direct Ollama options mapping.
        llm.model_params = {"temperature": 0, "num_ctx": 8192, "num_predict": 900}
        structured_output = False
    else:
        raise RuntimeError("provider는 auto, openai, ollama 중 하나여야 합니다")
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        prompt_template=EXTRACTION_PROMPT,
        create_lexical_graph=False,
        max_concurrency=max_concurrency,
        use_structured_output=structured_output,
    )
    text_chunks = TextChunks(
        chunks=[
            TextChunk(
                index=index,
                uid=str(chunk["chunk_id"]),
                metadata={"source_id": chunk["source_id"], "chunk_id": chunk["chunk_id"]},
                text=(
                    f"[source_id={chunk['source_id']}]\n[chunk_id={chunk['chunk_id']}]\n"
                    f"{chunk['content']}"
                ),
            )
            for index, chunk in enumerate(chunks)
        ]
    )
    result = await extractor.run(chunks=text_chunks, schema=graph_schema)
    payload = result.model_dump(mode="json")
    validation = validate_candidates(payload, chunks_by_id, schema_payload)

    candidate_dir = processed / "kg-builder"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "raw-graph.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (candidate_dir / "accepted.json").write_text(
        json.dumps(validation.accepted_relationships, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (candidate_dir / "rejected.json").write_text(
        json.dumps(validation.rejected_relationships, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "provider": selected_provider,
        "model": selected_model,
        "chunks": len(chunks),
        "nodes": len(payload.get("nodes", [])),
        "relationships": len(payload.get("relationships", [])),
        "accepted_relationships": len(validation.accepted_relationships),
        "rejected_relationships": len(validation.rejected_relationships),
    }


def run_extract_candidates(**kwargs: Any) -> dict[str, int | str]:
    return asyncio.run(extract_candidates(**kwargs))
