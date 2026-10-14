from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROMPT_VERSION = "2026-09-05.3"
VALIDATOR_VERSION = "2026-09-05.3"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class CandidateNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=64)
    properties: dict[str, Any]


class EvidenceProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)


class CandidateRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_node_id: str = Field(min_length=1)
    end_node_id: str = Field(min_length=1)
    type: str = Field(min_length=1, max_length=64)
    properties: EvidenceProperties


class CandidateGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[CandidateNode] = Field(default_factory=list, max_length=60)
    relationships: list[CandidateRelationship] = Field(default_factory=list, max_length=40)


class CompactFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_name: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    target_type: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)


class CompactExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: list[CompactFact] = Field(max_length=20)


class CuratedExtraction(CompactExtraction):
    source_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


@dataclass(frozen=True)
class ParsedPayload:
    payload: dict[str, Any]
    repaired: bool
    repair_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateValidation:
    accepted_relationships: list[dict[str, Any]]
    rejected_relationships: list[dict[str, Any]]
    review_relationships: list[dict[str, Any]] = field(default_factory=list)


def _json_object_candidates(raw: str) -> list[str]:
    value = raw.strip()
    candidates = [value]
    candidates.extend(
        item.strip()
        for item in re.findall(r"```(?:json)?\s*([\s\S]*?)```", value, re.IGNORECASE)
    )
    without_thought = re.sub(r"<think>[\s\S]*?</think>", "", value, flags=re.IGNORECASE)
    if without_thought.strip() != value:
        candidates.append(without_thought.strip())
    decoder = json.JSONDecoder()
    for index, char in enumerate(without_thought):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(without_thought[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(without_thought[index : index + end])
        break
    return list(dict.fromkeys(item for item in candidates if item))


def parse_candidate_payload(raw: str | dict[str, Any]) -> ParsedPayload:
    """Parse strict JSON, then perform only safe syntax repairs—never add facts."""
    if isinstance(raw, dict):
        model = CandidateGraph.model_validate(raw)
        return ParsedPayload(model.model_dump(mode="json"), repaired=False)
    errors: list[str] = []
    for candidate in _json_object_candidates(raw):
        attempts = [(candidate, ())]
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        if repaired != candidate:
            attempts.append((repaired, ("removed_trailing_comma",)))
        for text, notes in attempts:
            try:
                decoded = json.loads(text, strict=False)
                model = CandidateGraph.model_validate(decoded)
                return ParsedPayload(
                    model.model_dump(mode="json"), bool(notes), tuple(notes)
                )
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                errors.append(str(exc))
    last_error = errors[-1] if errors else "JSON object not found"
    raise ValueError(f"후보 JSON을 복구할 수 없습니다: {last_error}")


def parse_compact_payload(raw: str | dict[str, Any]) -> ParsedPayload:
    candidates = [json.dumps(raw, ensure_ascii=False)] if isinstance(raw, dict) else _json_object_candidates(raw)
    errors: list[str] = []
    for candidate in candidates:
        attempts = [(candidate, ())]
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        if repaired != candidate:
            attempts.append((repaired, ("removed_trailing_comma",)))
        for text, notes in attempts:
            try:
                decoded = json.loads(text, strict=False)
                model = CompactExtraction.model_validate(decoded)
                return ParsedPayload(model.model_dump(mode="json"), bool(notes), tuple(notes))
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                errors.append(str(exc))
    last_error = errors[-1] if errors else "JSON object not found"
    raise ValueError(f"간소화 후보 JSON을 복구할 수 없습니다: {last_error}")


def compact_json_schema(schema_payload: dict[str, Any]) -> dict[str, Any]:
    schema = CompactExtraction.model_json_schema()
    fact_properties = schema["$defs"]["CompactFact"]["properties"]
    node_types = sorted(_node_types(schema_payload))
    relationship_types = sorted(schema_payload["relationship_types"])
    fact_properties["source_type"]["enum"] = node_types
    fact_properties["target_type"]["enum"] = node_types
    fact_properties["relationship_type"]["enum"] = relationship_types
    return schema


def _local_node_id(label: str, name: str) -> str:
    digest = hashlib.sha256(f"{label}\0{_normalize_text(name)}".encode()).hexdigest()[:20]
    return f"node:{digest}"


def _quantified_properties(name: str) -> dict[str, Any]:
    match = re.search(r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*([^\d\s,]+)", name)
    if not match:
        return {}
    return {"value": float(match.group(1).replace(",", "")), "unit": match.group(2)}


def compact_to_candidate_graph(
    compact_payload: dict[str, Any], chunk: dict[str, Any]
) -> dict[str, Any]:
    """Build IDs and provenance deterministically instead of asking the LLM to copy them."""
    nodes: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    for fact in compact_payload.get("facts", []):
        endpoints: list[tuple[str, str]] = [
            (fact["source_type"], fact["source_name"]),
            (fact["target_type"], fact["target_name"]),
        ]
        endpoint_ids: list[str] = []
        for label, name in endpoints:
            node_id = _local_node_id(label, name)
            properties: dict[str, Any] = {"id": node_id, "name": name}
            if label == "Rule":
                properties.update(
                    {
                        "statement": fact["evidence_quote"],
                        "source_id": chunk["source_id"],
                        "chunk_id": chunk["chunk_id"],
                        "evidence_quote": fact["evidence_quote"],
                    }
                )
            elif label == "QuantifiedValue":
                properties.update(_quantified_properties(name))
            nodes[node_id] = {"id": node_id, "label": label, "properties": properties}
            endpoint_ids.append(node_id)
        relationships.append(
            {
                "start_node_id": endpoint_ids[0],
                "end_node_id": endpoint_ids[1],
                "type": fact["relationship_type"],
                "properties": {
                    "source_id": chunk["source_id"],
                    "chunk_id": chunk["chunk_id"],
                    "evidence_quote": fact["evidence_quote"],
                    "confidence": fact["confidence"],
                },
            }
        )
    return {"nodes": list(nodes.values()), "relationships": relationships}


def _node_types(schema_payload: dict[str, Any]) -> set[str]:
    value = schema_payload["node_types"]
    return set(value if isinstance(value, list) else value.keys())


def _required_properties(schema_payload: dict[str, Any], label: str) -> list[str]:
    value = schema_payload["node_types"]
    if not isinstance(value, dict):
        return []
    return list((value.get(label) or {}).get("required", []))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _strict_alias_matches(label: str, name: str, policy: dict[str, Any]) -> bool:
    aliases = (policy.get("strict_entity_aliases") or {}).get(label)
    if not aliases:
        return True
    normalized_name = _normalize_text(name)
    return normalized_name in {_normalize_text(str(alias)) for alias in aliases}


def align_quote_to_source(quote: str, content: str) -> str | None:
    """Return the exact source slice when only Markdown whitespace differs."""
    if quote in content:
        return quote
    tokens = re.findall(r"\S+", quote)
    if not tokens:
        return None
    match = re.search(r"\s+".join(re.escape(token) for token in tokens), content)
    return content[match.start() : match.end()] if match else None


def _fill_rule_evidence(
    node: dict[str, Any] | None,
    *,
    source_id: str,
    chunk_id: str,
    quote: str,
) -> None:
    """Copy existing relationship evidence into a Rule node; no new fact is inferred."""
    if node is None or node.get("label") != "Rule" or not quote:
        return
    properties = node.setdefault("properties", {})
    properties.setdefault("statement", quote)
    properties.setdefault("source_id", source_id)
    properties.setdefault("chunk_id", chunk_id)
    properties.setdefault("evidence_quote", quote)


def validate_candidates(
    graph_payload: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    schema_payload: dict[str, Any],
) -> CandidateValidation:
    """Check allowlists, required fields, provenance, exact quotes, and confidence."""
    allowed_node_types = _node_types(schema_payload)
    relationship_specs = schema_payload["relationship_types"]
    policy = schema_payload.get("extraction_policy", {})
    minimum_quote_length = int(policy.get("minimum_evidence_characters", 8))
    auto_accept_confidence = float(policy.get("auto_accept_confidence", 0.8))

    nodes: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for node in graph_payload.get("nodes", []):
        node_id = str(node.get("id", ""))
        if node_id in nodes:
            duplicate_ids.add(node_id)
        elif node_id:
            nodes[node_id] = node

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    seen_claims: set[str] = set()
    for original_relation in graph_payload.get("relationships", []):
        reasons: list[str] = []
        relation = copy.deepcopy(original_relation)
        source = copy.deepcopy(nodes.get(str(relation.get("start_node_id", ""))))
        target = copy.deepcopy(nodes.get(str(relation.get("end_node_id", ""))))
        relation_type = str(relation.get("type", ""))
        spec = relationship_specs.get(relation_type)
        properties = relation.get("properties") or {}
        source_id = str(properties.get("source_id", ""))
        chunk_id = str(properties.get("chunk_id", ""))
        quote = str(properties.get("evidence_quote", ""))
        confidence = properties.get("confidence")
        chunk = chunks_by_id.get(chunk_id)
        if chunk is not None and quote:
            aligned_quote = align_quote_to_source(quote, str(chunk["content"]))
            if aligned_quote is not None:
                quote = aligned_quote
                properties["evidence_quote"] = aligned_quote
        _fill_rule_evidence(
            source, source_id=source_id, chunk_id=chunk_id, quote=quote
        )
        _fill_rule_evidence(
            target, source_id=source_id, chunk_id=chunk_id, quote=quote
        )

        if source is None or target is None:
            reasons.append("관계의 시작 또는 끝 노드가 없음")
        for role, node in (("시작", source), ("끝", target)):
            if node is None:
                continue
            node_id = str(node.get("id", ""))
            label = str(node.get("label", ""))
            node_properties = node.get("properties") or {}
            if node_id in duplicate_ids:
                reasons.append(f"중복된 {role} 노드 ID")
            if not ID_PATTERN.fullmatch(node_id):
                reasons.append(f"허용되지 않은 {role} 노드 ID")
            if label not in allowed_node_types:
                reasons.append(f"허용되지 않은 {role} 노드 타입: {label}")
            name = str(node_properties.get("name", ""))
            if name and not _strict_alias_matches(label, name, policy):
                reasons.append(f"{role} 노드가 {label} 통제 어휘에 없음: {name}")
            for required in _required_properties(schema_payload, label):
                if node_properties.get(required) in (None, ""):
                    reasons.append(f"{role} 노드 필수 속성 누락: {required}")
            if "id" in node_properties and str(node_properties["id"]) != node_id:
                reasons.append(f"{role} 노드 id 속성과 노드 ID가 다름")

        if spec is None:
            reasons.append(f"허용되지 않은 관계 타입: {relation_type}")
        elif source and target:
            if source.get("label") not in spec["from"]:
                reasons.append("스키마상 허용되지 않은 시작 노드 타입")
            if target.get("label") not in spec["to"]:
                reasons.append("스키마상 허용되지 않은 끝 노드 타입")

        if not source_id or not chunk_id or not quote:
            reasons.append("source_id/chunk_id/evidence_quote 중 필수 근거가 누락됨")
        elif chunk is None:
            reasons.append("존재하지 않는 chunk_id")
        else:
            content = str(chunk["content"])
            if source_id != chunk["source_id"]:
                reasons.append("source_id와 청크 소유 문서가 다름")
            if quote not in content:
                reasons.append("evidence_quote가 원문 청크에 존재하지 않음: 연속 문자열 불일치")
            if len(_normalize_text(quote)) < minimum_quote_length:
                reasons.append("evidence_quote가 의미 검증에 너무 짧음")
            for role, node in (("시작", source), ("끝", target)):
                if node is None or node.get("label") in {"Rule", "QuantifiedValue"}:
                    continue
                name = str((node.get("properties") or {}).get("name", ""))
                if name and _normalize_text(name) not in _normalize_text(content):
                    reasons.append(f"{role} 노드 name이 원문 청크에 없음")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            reasons.append("confidence가 0~1 범위 숫자가 아님")
        conditioned_direct = set(policy.get("conditioned_direct_relationships", []))
        condition_cues = [str(item) for item in policy.get("condition_cues", [])]
        if relation_type in conditioned_direct and any(cue in quote for cue in condition_cues):
            reasons.append("조건부 허용·금지는 Rule 노드를 경유해야 함")

        record = {"relationship": relation, "source_node": source, "target_node": target}
        if reasons:
            record["rejection_reasons"] = list(dict.fromkeys(reasons))
            rejected.append(record)
            continue
        claim_id = make_claim_id(source, relation, target)
        if claim_id in seen_claims:
            record["rejection_reasons"] = ["동일한 근거의 중복 관계"]
            rejected.append(record)
            continue
        seen_claims.add(claim_id)
        record["claim_id"] = claim_id
        if float(confidence) < auto_accept_confidence:
            record["review_reason"] = (
                f"confidence {float(confidence):.3f} < "
                f"auto_accept_confidence {auto_accept_confidence:.3f}"
            )
            review.append(record)
        else:
            accepted.append(record)
    return CandidateValidation(accepted, rejected, review)


def make_entity_id(node: dict[str, Any]) -> str:
    label = str(node["label"])
    name = _normalize_text(str((node.get("properties") or {}).get("name", node["id"])))
    digest = hashlib.sha256(f"{label}\0{name}".encode()).hexdigest()[:20]
    return f"entity:{label.lower()}:{digest}"


def make_claim_id(
    source: dict[str, Any], relation: dict[str, Any], target: dict[str, Any]
) -> str:
    properties = relation.get("properties") or {}
    parts = (
        make_entity_id(source),
        str(relation.get("type", "")),
        make_entity_id(target),
        str(properties.get("chunk_id", "")),
        str(properties.get("evidence_quote", "")),
    )
    return "claim:" + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def promote_accepted(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for record in records:
        relation = record["relationship"]
        properties = relation["properties"]
        source = record["source_node"]
        target = record["target_node"]
        source_id = make_entity_id(source)
        target_id = make_entity_id(target)
        edge_digest = hashlib.sha256(
            f"{source_id}\0{relation['type']}\0{target_id}".encode()
        ).hexdigest()[:24]
        promoted.append(
            {
                "claim_id": record.get("claim_id") or make_claim_id(source, relation, target),
                "edge_key": f"edge:{edge_digest}",
                "relationship_type": relation["type"],
                "source": {
                    "entity_id": source_id,
                    "label": source["label"],
                    **source["properties"],
                },
                "target": {
                    "entity_id": target_id,
                    "label": target["label"],
                    **target["properties"],
                },
                **properties,
                "status": "validated",
                "extractor_prompt_version": PROMPT_VERSION,
            }
        )
    return promoted


def build_extraction_prompt(chunk: dict[str, Any], schema_payload: dict[str, Any]) -> str:
    node_lines = [
        f"- {label}: {spec.get('description', '')}"
        for label, spec in schema_payload["node_types"].items()
    ]
    relation_lines = [
        f"- {label}: {','.join(spec['from'])} -> {','.join(spec['to'])}"
        for label, spec in schema_payload["relationship_types"].items()
    ]
    return f"""
당신은 한국 연금 문서의 정보 추출기다. JSON Schema에 맞는 facts 배열만 출력한다.

절대 규칙:
1. 아래 원문에 명시된 사실만 추출하고 상식이나 추측을 추가하지 마라.
2. 노드 label과 관계 type은 아래 허용 목록만 사용하라.
3. source_name과 target_name은 원문에 실제로 있는 짧은 표현을 그대로 복사하라.
4. evidence_quote는 원문에 그대로 존재하는 완전한 문장 또는 표 행이어야 한다.
5. source_type, relationship_type, target_type은 허용 목록에서만 고르라.
6. ID, source_id, chunk_id는 출력하지 마라. 프로그램이 확정한다.
7. 서로 다른 조건·법률·세금 관점을 합치지 마라. O/X는 열 제목 관점으로만 해석하라.
8. 관계가 없으면 facts를 빈 배열로 반환하라. 최대 20개 관계만 추출하라.
9. 조건·예외·연령·기간·사유가 붙은 허용 또는 금지는 PensionScheme의 직접 관계로 만들지 마라.
   Rule을 주어로 만들고 APPLIES_TO로 제도에, HAS_CONDITION으로 조건에,
   PERMITS_EVENT 또는 FORBIDS_EVENT로 사건에 연결하라.
10. Tax 타입은 연금소득세·퇴직소득세·기타소득세·이자소득세처럼 실제 세금 이름에만 사용하라.
    '세율', '저율', '세금'처럼 종류가 불명확한 표현은 Tax 노드로 만들지 마라.

허용 노드:
{chr(10).join(node_lines)}

허용 관계:
{chr(10).join(relation_lines)}

원문:
<chunk>
{chunk['content']}
</chunk>
""".strip()


def _checkpoint_key(chunk: dict[str, Any], model: str) -> str:
    raw = f"{PROMPT_VERSION}\0{model}\0{chunk['chunk_id']}\0{chunk['sha256']}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


async def extract_with_ollama(
    project_root: Path,
    *,
    model: str = "qwen3:14b",
    limit: int = 0,
    source_id: str | None = None,
    chunk_id: str | None = None,
    max_concurrency: int = 1,
    retries: int = 2,
    resume: bool = True,
) -> dict[str, Any]:
    try:
        from ollama import AsyncClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install -e '.[kg-builder]'가 필요합니다") from exc

    processed = project_root / "data" / "processed"
    chunks = [
        json.loads(line)
        for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if source_id:
        chunks = [item for item in chunks if item["source_id"] == source_id]
    if chunk_id:
        chunks = [item for item in chunks if item["chunk_id"] == chunk_id]
    if limit > 0:
        chunks = chunks[:limit]
    if not chunks:
        raise RuntimeError("선택된 청크가 없습니다")

    schema_payload = yaml.safe_load(
        (project_root / "config" / "kg_schema.yaml").read_text(encoding="utf-8")
    )
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    output_dir = processed / "semantic-kg"
    checkpoint_dir = output_dir / "checkpoints"
    client = AsyncClient(timeout=900)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process(chunk: dict[str, Any]) -> dict[str, Any]:
        checkpoint = checkpoint_dir / f"{_checkpoint_key(chunk, model)}.json"
        if resume and checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if cached.get("status") == "processed" and cached.get("payload"):
                validation = validate_candidates(cached["payload"], chunks_by_id, schema_payload)
                cached["accepted"] = validation.accepted_relationships
                cached["review"] = validation.review_relationships
                cached["rejected"] = validation.rejected_relationships
                cached["validator_version"] = VALIDATOR_VERSION
                _write_json(checkpoint, cached)
            return cached
        prompt = build_extraction_prompt(chunk, schema_payload)
        last_error = ""
        async with semaphore:
            for attempt in range(1, retries + 2):
                started = datetime.now(UTC)
                try:
                    response = await client.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        format=compact_json_schema(schema_payload),
                        options={
                            "temperature": 0,
                            "num_ctx": 8192,
                            "num_predict": 1800,
                            "seed": 42,
                        },
                        keep_alive="30m",
                    )
                    raw = response["message"]["content"]
                    parsed = parse_compact_payload(raw)
                    candidate_payload = compact_to_candidate_graph(parsed.payload, chunk)
                    validation = validate_candidates(candidate_payload, chunks_by_id, schema_payload)
                    record = {
                        "status": "processed",
                        "chunk_id": chunk["chunk_id"],
                        "source_id": chunk["source_id"],
                        "chunk_sha256": chunk["sha256"],
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        "validator_version": VALIDATOR_VERSION,
                        "attempt": attempt,
                        "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                        "json_repaired": parsed.repaired,
                        "repair_notes": list(parsed.repair_notes),
                        "raw_response": raw,
                        "compact_payload": parsed.payload,
                        "payload": candidate_payload,
                        "accepted": validation.accepted_relationships,
                        "review": validation.review_relationships,
                        "rejected": validation.rejected_relationships,
                    }
                    _write_json(checkpoint, record)
                    return record
                except Exception as exc:  # noqa: BLE001 - isolate and retry any per-chunk failure
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt <= retries:
                        prompt += "\n이전 응답은 검증에 실패했다. 설명 없이 유효한 JSON만 다시 출력하라."
            record = {
                "status": "error",
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "chunk_sha256": chunk["sha256"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "validator_version": VALIDATOR_VERSION,
                "error": last_error,
                "accepted": [],
                "review": [],
                "rejected": [],
            }
            _write_json(checkpoint, record)
            return record

    results = await asyncio.gather(*(process(chunk) for chunk in chunks))
    accepted = [item for result in results for item in result.get("accepted", [])]
    review = [item for result in results for item in result.get("review", [])]
    rejected = [item for result in results for item in result.get("rejected", [])]
    errors = [result for result in results if result["status"] == "error"]
    promoted = promote_accepted(accepted)
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "review.jsonl", review)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    _write_jsonl(output_dir / "errors.jsonl", errors)
    _write_jsonl(output_dir / "promoted.jsonl", promoted)
    summary = {
        "provider": "ollama",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "selected_chunks": len(chunks),
        "processed_chunks": len(results) - len(errors),
        "error_chunks": len(errors),
        "accepted_relationships": len(accepted),
        "review_relationships": len(review),
        "rejected_relationships": len(rejected),
        "promoted_claims": len(promoted),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_dir / "run-summary.json", summary)
    return summary


def run_staged_extraction(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(extract_with_ollama(**kwargs))


def validate_curated_extraction(project_root: Path, curated_path: Path) -> dict[str, Any]:
    """Validate an agent-curated fact file through the production gates."""
    allowed_dir = (project_root / "data" / "curated" / "semantic-kg").resolve()
    resolved = curated_path.resolve()
    if resolved.parent != allowed_dir:
        raise ValueError(f"수동 의미 관계 입력은 {allowed_dir} 바로 아래 파일만 허용됩니다")
    curated = CuratedExtraction.model_validate_json(resolved.read_text(encoding="utf-8"))
    processed = project_root / "data" / "processed"
    chunks = [
        json.loads(line)
        for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    chunk = chunks_by_id.get(curated.chunk_id)
    if chunk is None or chunk["source_id"] != curated.source_id:
        raise ValueError("curated source_id/chunk_id가 원문 청크와 일치하지 않습니다")
    schema_payload = yaml.safe_load(
        (project_root / "config" / "kg_schema.yaml").read_text(encoding="utf-8")
    )
    compact_payload = CompactExtraction(facts=curated.facts).model_dump(mode="json")
    candidate_payload = compact_to_candidate_graph(compact_payload, chunk)
    validation = validate_candidates(candidate_payload, chunks_by_id, schema_payload)
    promoted = promote_accepted(validation.accepted_relationships)
    output_dir = processed / "semantic-kg" / "curated" / resolved.stem
    _write_json(output_dir / "candidate-graph.json", candidate_payload)
    _write_jsonl(output_dir / "accepted.jsonl", validation.accepted_relationships)
    _write_jsonl(output_dir / "review.jsonl", validation.review_relationships)
    _write_jsonl(output_dir / "rejected.jsonl", validation.rejected_relationships)
    _write_jsonl(output_dir / "promoted.jsonl", promoted)
    summary = {
        "curated_file": str(resolved.relative_to(project_root)),
        "source_id": curated.source_id,
        "chunk_id": curated.chunk_id,
        "candidate_relationships": len(candidate_payload["relationships"]),
        "accepted_relationships": len(validation.accepted_relationships),
        "review_relationships": len(validation.review_relationships),
        "rejected_relationships": len(validation.rejected_relationships),
        "promoted_claims": len(promoted),
        "validator_version": VALIDATOR_VERSION,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
