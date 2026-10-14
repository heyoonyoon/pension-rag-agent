from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pension_graph.semantic_kg import (
    CompactExtraction,
    CuratedExtraction,
    _write_json,
    _write_jsonl,
    compact_to_candidate_graph,
    promote_accepted,
    validate_candidates,
)

DIRECT_EXTRACTOR_VERSION = "codex-direct-2026-09-05.5"


@dataclass(frozen=True)
class Mention:
    name: str
    start: int
    end: int


SCHEMES = (
    "개인형 퇴직연금",
    "개인형퇴직연금",
    "기업형IRP",
    "개인형IRP",
    "연금저축계좌",
    "개인연금저축",
    "퇴직연금",
    "연금계좌",
    "IRP",
    "DC형",
    "DC 형",
    "DB형",
    "DB 형",
    "DC",
    "DB",
)

ROLES = (
    "개인 거주자",
    "부양가족",
    "전문투자자",
    "신규가입자",
    "가입자",
    "근로자",
    "사용자",
    "퇴직자",
    "수급자",
    "배우자",
    "임원",
    "자영업자",
    "회사",
)

EVENTS = (
    "과세재원 확정",
    "개인회생절차개시",
    "개인회생절차",
    "부담금 납입",
    "연금지급개시",
    "연금 지급",
    "연금지급",
    "연금수령",
    "연금개시",
    "중도 인출",
    "중도인출",
    "일부 인출",
    "계좌 해지",
    "계좌해지",
    "계약 해지",
    "계약해지",
    "특별해지",
    "해지상환",
    "계약 이전",
    "계약이전",
    "계좌이체",
    "가입자이전",
    "추가 납입",
    "추가납입",
    "신규 개설",
    "신규개설",
    "계좌개설",
    "리밸런싱",
    "운용지시",
    "자동 매수",
    "자동매수",
    "직접 매수",
    "현물 이전",
    "현금 이전",
    "교체매매",
    "매매",
    "유상청약",
    "반대매매",
    "담보융자",
    "증권담보융자",
    "서류 제출",
    "제출",
    "확인",
    "작성",
    "등록",
    "변경",
    "처리",
    "게시",
    "통지",
    "발송",
    "이용",
    "조회",
    "선택",
    "현금상환",
    "상환매도",
    "연금교육",
    "가입자교육",
    "압류",
    "매수",
    "매도",
    "환매",
    "인출",
    "해지",
    "이전",
    "납입",
    "입금",
    "지급",
    "수령",
    "신청",
    "취소",
    "가입",
    "개설",
    "교육",
    "청약",
    "상환",
)

PRODUCTS = (
    "미래에셋증권현금성자산",
    "상장인프라펀드",
    "원리금보장상품",
    "원리금 지급 운용상품",
    "집합투자증권",
    "주식혼합형 펀드",
    "채권혼합형펀드",
    "현금성자산",
    "디폴트옵션",
    "사전지정운용방법",
    "해외상장 ETF",
    "레버리지 ETF",
    "인버스 ETF",
    "상장 주식",
    "비상장 주식",
    "발행어음",
    "국공채",
    "전환사채",
    "신주인수권부 사채",
    "교환사채",
    "후순위채권",
    "ETF",
    "ETN",
    "TDF",
    "MMF",
    "리츠",
    "REIT",
    "펀드",
    "채권",
    "예금",
    "주식",
)

TAXES = (
    "이연퇴직소득세",
    "연금소득세",
    "퇴직소득세",
    "기타소득세",
    "이자소득세",
)

CHANNELS = (
    "사무담당자웹",
    "M-STOCK",
    "홈페이지",
    "영업점",
    "모바일 App",
    "모바일 앱",
    "모바일",
    "온라인",
    "카이로스",
    "내점",
    "유선",
    "MTS",
    "HTS",
    "FAX",
    "E-mail",
)

LAWS = (
    "근로자퇴직급여 보장법 시행령",
    "근로자퇴직급여보장법 시행령",
    "근로자퇴직급여 보장법",
    "근로자퇴직급여보장법",
    "조세특례제한법",
    "소득세법시행령",
    "소득세법 시행령",
    "소득세법",
    "재난 및 안전관리 기본법",
    "퇴직연금 감독규정",
    "퇴직연금감독규정",
    "근로기준법",
    "산업재해보상보험법",
    "국민연금법",
    "주택법",
    "의료법",
    "민법",
    "상법",
)

DOCUMENT_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9·ㆍ()\[\]-]{0,24}"
    r"(?:신청서|증명서|확인서|계약서|진단서|소견서|신고서|결정문|등록증|등본|명부)"
)
QUANTITY_PATTERN = re.compile(
    r"(?:연\s*|월\s*|매\s*)?"
    r"(?:만\s*)?[0-9][0-9,]*(?:\.[0-9]+)?\s*"
    r"(?:천만원|백만원|만원|억원|원|%|퍼센트|세|년|개월|주|영업일|일|시간|시|회|종목)"
    r"(?:\s*(?:이상|이하|이내|초과|미만|이전|이후|경과|까지|한도))?"
)
CONDITION_PATTERN = re.compile(
    r"([^|。.!?\n]{2,80}?(?:경우|사유에 해당(?:하는|한다면)|때))"
)

NEGATIVE_CUES = ("불가능", "불가", "금지", "할 수 없다", "할 수 없습니다", "못")
REQUIRED_CUES = ("반드시", "필수", "의무", "해야", "하여야")
POSITIVE_CUES = ("가능", "할 수 있", "허용", "신청할 수", "이용할 수")


def _ordered_mentions(text: str, terms: Iterable[str]) -> list[Mention]:
    mentions: list[Mention] = []
    occupied: list[tuple[int, int]] = []
    for term in sorted(set(terms), key=len, reverse=True):
        for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
            span = (match.start(), match.end())
            if any(span[0] < end and span[1] > start for start, end in occupied):
                continue
            occupied.append(span)
            mentions.append(Mention(match.group(0), *span))
    return sorted(mentions, key=lambda item: item.start)


def _evidence_units(content: str) -> list[str]:
    units: list[str] = []
    for line in content.splitlines():
        value = line.strip()
        if not value or re.fullmatch(r"\|?[\s:|-]+\|?", value):
            continue
        html_rows = re.findall(r"<tr[\s\S]*?</tr>", value, flags=re.IGNORECASE)
        if html_rows:
            for row in html_rows:
                pieces = re.split(r"<br\s*/?>", row, flags=re.IGNORECASE)
                units.extend(
                    piece for piece in pieces if len(re.sub(r"<[^>]+>", "", piece)) >= 8
                )
            without_rows = re.sub(r"<tr[\s\S]*?</tr>", "", value, flags=re.IGNORECASE)
            if len(re.sub(r"<[^>]+>", "", without_rows).strip()) >= 8:
                units.append(without_rows)
            continue
        if value.startswith("#") and len(value.lstrip("# ")) < 45:
            continue
        visible = re.sub(r"<[^>]+>", "", value).strip(" -*#|")
        if len(visible) < 30 and visible.endswith(
            ("시기", "절차", "방법", "조건", "대상", "기준", "한도", "유형", "종류", "사항", "안내")
        ):
            continue
        if value.startswith("|"):
            pieces = re.split(r"<br\s*/?>", value, flags=re.IGNORECASE)
        else:
            pieces = re.split(r"(?<=[.!?])\s+(?=[-가-힣A-Za-z0-9])", value)
        units.extend(
            piece.strip()
            for piece in pieces
            if len(re.sub(r"<[^>]+>", "", piece).strip()) >= 8
        )
    return list(dict.fromkeys(units))


def _cue_mentions(text: str) -> list[tuple[str, int, str]]:
    result: list[tuple[str, int, str]] = []
    for kind, cues in (
        ("negative", NEGATIVE_CUES),
        ("required", REQUIRED_CUES),
        ("positive", POSITIVE_CUES),
    ):
        for cue in cues:
            for match in re.finditer(re.escape(cue), text):
                start, end = match.span()
                before = text[max(0, start - 5) : start]
                after = text[end : end + 3]
                if cue == "가능" and (
                    before.endswith("불")
                    or after.startswith(
                        (
                            "금액",
                            "일",
                            "여부",
                            "자산",
                            "수량",
                            "시간",
                            "매체",
                            "종목",
                            "조건",
                            "시기",
                            "대상",
                        )
                    )
                ):
                    continue
                if cue == "불가" and after.startswith("피"):
                    continue
                if cue == "금지":
                    attached_valid = before.endswith(("투자", "압류", "매매", "인출", "가입"))
                    if (start > 0 and text[start - 1].isalnum() and not attached_valid) or after.startswith(
                        ("급", "연")
                    ):
                        continue
                result.append((kind, start, cue))
    return result


def _modality_assignments(text: str, mentions: list[Mention]) -> list[tuple[Mention, str]]:
    """Assign each modal cue to one nearest event/product, preventing relation fan-out."""
    assignments: list[tuple[Mention, str]] = []
    for kind, position, cue in _cue_mentions(text):
        if cue == "가능" and text[position : position + 4] in {"가능금액", "가능 금"}:
            continue
        if cue == "할 수 있" and text[max(0, position - 6) : position].rstrip().endswith(
            ("발생", "확인", "불가")
        ):
            continue
        candidates: list[tuple[int, int, Mention]] = []
        for mention in mentions:
            if mention.end <= position:
                distance = position - mention.end
                direction = 0
            else:
                distance = mention.start - (position + len(cue))
                direction = 1
            if kind == "positive" and direction == 1:
                continue
            if kind == "required" and direction == 1 and cue not in {"반드시", "의무"}:
                continue
            radius = 32 if direction == 0 else 24
            if 0 <= distance <= radius:
                between = text[min(mention.start, position) : max(mention.end, position + len(cue))]
                required_object_cues = (
                    "사유",
                    "조건",
                    "시기",
                    "한도",
                    "방법",
                    "계획",
                    "서류",
                    "요건",
                )
                if kind == "required" and any(item in between for item in required_object_cues):
                    continue
                if "?" not in between and "없이" not in between and between.count("|") <= 1:
                    candidates.append((distance, direction, mention))
        if candidates:
            assignments.append((min(candidates, key=lambda item: (item[0], item[1]))[2], kind))
    return assignments


def _is_question(text: str) -> bool:
    visible = re.sub(r"<[^>]+>", "", text).strip(" |")
    return "?" in visible or visible.endswith(("나요", "까요")) or "어떻게 해야 하나요" in visible


def _event_mentions(text: str) -> list[Mention]:
    result: list[Mention] = []
    for mention in _ordered_mentions(text, EVENTS):
        before = text[max(0, mention.start - 8) : mention.start].rstrip()
        after = text[mention.end : mention.end + 8].lstrip()
        if mention.name == "가입" and after.startswith(("자", "기간", "일", "대상", "조건")):
            continue
        if mention.name == "이전" and before.endswith(("소유권", "주소", "근무지")):
            continue
        if mention.name == "매수" and after.startswith(("형", "가능금액", "상품")):
            continue
        if mention.name == "인출" and after.startswith(
            ("기", "재원", "순서", "금액", "한도", "방법", "전략", "계획")
        ):
            continue
        if mention.name in {"납입", "입금"} and after.startswith(("가능한도", "한도", "금액", "일")):
            continue
        result.append(mention)
    return result


def _supports_product_modality(text: str) -> bool:
    return any(
        cue in text
        for cue in (
            "투자 가능",
            "투자가능",
            "투자금지",
            "투자 금지",
            "매매 가능",
            "매매가능",
            "거래 가능",
            "거래가능",
            "운용이 가능",
            "운용 가능",
            "편입 가능",
            "편입 불가",
            "투자를 허용",
            "투자 허용",
        )
    )


def _channel_is_available(text: str, channel: Mention) -> bool:
    window = text[max(0, channel.start - 15) : min(len(text), channel.end + 18)]
    return not any(cue in window for cue in ("불가", "불가능", "제외", "금지"))


def _rule_name(chunk_id: str, unit_index: int) -> str:
    return f"{chunk_id}:rule:{unit_index:04d}"


def _fact(
    *,
    rule_name: str,
    relationship_type: str,
    target_name: str,
    target_type: str,
    quote: str,
    confidence: float = 0.96,
) -> dict[str, Any]:
    return {
        "source_name": rule_name,
        "source_type": "Rule",
        "relationship_type": relationship_type,
        "target_name": target_name,
        "target_type": target_type,
        "evidence_quote": quote,
        "confidence": confidence,
    }


def _context_mentions(chunk: dict[str, Any], terms: Iterable[str]) -> list[Mention]:
    heading = " > ".join(str(item) for item in chunk.get("heading_path", []))
    prefix = f"{heading}\n{chunk['content'][:240]}"
    content = str(chunk["content"])
    return [
        mention
        for mention in _ordered_mentions(prefix, terms)
        if re.search(re.escape(mention.name), content, flags=re.IGNORECASE)
    ]


def extract_chunk_facts(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    context_schemes = _context_mentions(chunk, SCHEMES)
    for unit_index, quote in enumerate(_evidence_units(str(chunk["content"]))):
        unit_facts: list[dict[str, Any]] = []
        rule_name = _rule_name(str(chunk["chunk_id"]), unit_index)
        events = _event_mentions(quote)
        products = _ordered_mentions(quote, PRODUCTS)
        schemes = _ordered_mentions(quote, SCHEMES)
        roles = _ordered_mentions(quote, ROLES)

        modal_events = [] if _is_question(quote) else _modality_assignments(quote, events)
        for event, modality in modal_events:
            relation = {
                "negative": "FORBIDS_EVENT",
                "required": "REQUIRES_EVENT",
                "positive": "PERMITS_EVENT",
            }.get(modality)
            if relation:
                unit_facts.append(
                    _fact(
                        rule_name=rule_name,
                        relationship_type=relation,
                        target_name=event.name,
                        target_type="FinancialEvent",
                        quote=quote,
                    )
                )

        modal_products = (
            _modality_assignments(quote, products)
            if not _is_question(quote) and _supports_product_modality(quote)
            else []
        )
        for product, modality in modal_products:
            relation = {
                "negative": "FORBIDS_INVESTMENT",
                "positive": "PERMITS_INVESTMENT",
            }.get(modality)
            if relation:
                unit_facts.append(
                    _fact(
                        rule_name=rule_name,
                        relationship_type=relation,
                        target_name=product.name,
                        target_type="InvestmentProduct",
                        quote=quote,
                    )
                )

        for tax in _ordered_mentions(quote, TAXES):
            window = quote[max(0, tax.start - 60) : min(len(quote), tax.end + 60)]
            if "과세 체계에 관한 사항" in window:
                continue
            if "비과세" in window:
                relation = "EXEMPTS_TAX"
            elif "과세" in window or "%" in window or "세율" in window:
                relation = "IMPOSES_TAX"
            else:
                continue
            unit_facts.append(
                _fact(
                    rule_name=rule_name,
                    relationship_type=relation,
                    target_name=tax.name,
                    target_type="Tax",
                    quote=quote,
                )
            )

        for law in _ordered_mentions(quote, LAWS):
            unit_facts.append(
                _fact(
                    rule_name=rule_name,
                    relationship_type="GOVERNED_BY",
                    target_name=law.name,
                    target_type="LegalInstrument",
                    quote=quote,
                    confidence=0.99,
                )
            )

        document_signal = any(cue in quote for cue in ("필수", "제출", "지참", "준비", "징구"))
        if document_signal:
            for match in DOCUMENT_PATTERN.finditer(quote):
                unit_facts.append(
                    _fact(
                        rule_name=rule_name,
                        relationship_type="REQUIRES_DOCUMENT",
                        target_name=match.group(0),
                        target_type="RequiredDocument",
                        quote=quote,
                        confidence=0.94,
                    )
                )

        channel_signal = any(cue in quote for cue in ("접수", "신청", "가능", "처리", "경로"))
        if channel_signal and not _is_question(quote):
            for channel in _ordered_mentions(quote, CHANNELS):
                if not _channel_is_available(quote, channel):
                    continue
                unit_facts.append(
                    _fact(
                        rule_name=rule_name,
                        relationship_type="AVAILABLE_VIA",
                        target_name=channel.name,
                        target_type="ServiceChannel",
                        quote=quote,
                        confidence=0.94,
                    )
                )

        has_core_relation = bool(unit_facts)
        numeric_signal = any(
            cue in quote
            for cue in ("한도", "세율", "기간", "기한", "연령", "이내", "이상", "이하", "초과", "미만")
        )
        if has_core_relation or numeric_signal:
            for match in list(QUANTITY_PATTERN.finditer(quote))[:8]:
                unit_facts.append(
                    _fact(
                        rule_name=rule_name,
                        relationship_type="HAS_VALUE",
                        target_name=match.group(0),
                        target_type="QuantifiedValue",
                        quote=quote,
                        confidence=0.97,
                    )
                )

        if "<" not in quote and unit_facts and any(item["relationship_type"] in {
            "PERMITS_EVENT", "FORBIDS_EVENT", "REQUIRES_EVENT",
            "PERMITS_INVESTMENT", "FORBIDS_INVESTMENT",
        } for item in unit_facts):
            for condition in CONDITION_PATTERN.finditer(quote):
                name = condition.group(1).strip(" -:;,()")
                if len(name) >= 6 and name not in {"이 경우", "그 경우"}:
                    unit_facts.append(
                        _fact(
                            rule_name=rule_name,
                            relationship_type="HAS_CONDITION",
                            target_name=name,
                            target_type="Condition",
                            quote=quote,
                            confidence=0.9,
                        )
                    )

        if unit_facts:
            applicable_schemes = schemes if len(schemes) == 1 else []
            if not schemes:
                applicable_schemes = context_schemes[:1]
            for scheme in applicable_schemes:
                unit_facts.append(
                    _fact(
                        rule_name=rule_name,
                        relationship_type="APPLIES_TO",
                        target_name=scheme.name,
                        target_type="PensionScheme",
                        quote=quote,
                        confidence=0.9 if schemes else 0.82,
                    )
                )
            for role in roles[:3]:
                unit_facts.append(
                    _fact(
                        rule_name=rule_name,
                        relationship_type="APPLIES_TO",
                        target_name=role.name,
                        target_type="PersonRole",
                        quote=quote,
                        confidence=0.9,
                    )
                )

        facts.extend(unit_facts)

    deduplicated: dict[tuple[str, ...], dict[str, Any]] = {}
    for fact in facts:
        key = (
            fact["source_name"],
            fact["relationship_type"],
            fact["target_type"],
            fact["target_name"].casefold(),
            fact["evidence_quote"],
        )
        deduplicated[key] = fact
    return list(deduplicated.values())


def _manual_extractions(curated_dir: Path) -> dict[str, CuratedExtraction]:
    result: dict[str, CuratedExtraction] = {}
    for path in sorted(curated_dir.glob("*.json")):
        curated = CuratedExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        if curated.chunk_id in result:
            raise ValueError(f"한 청크에 수동 검증 파일이 둘 이상입니다: {curated.chunk_id}")
        result[curated.chunk_id] = curated
    return result


def _hold_scope_conflicts(
    accepted: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    opposing_pairs = (
        frozenset(("PERMITS_EVENT", "FORBIDS_EVENT")),
        frozenset(("PERMITS_INVESTMENT", "FORBIDS_INVESTMENT")),
        frozenset(("IMPOSES_TAX", "EXEMPTS_TAX")),
    )
    types_by_edge: dict[tuple[str, str], set[str]] = {}
    for record in accepted:
        relation = record["relationship"]
        key = (relation["start_node_id"], relation["end_node_id"])
        types_by_edge.setdefault(key, set()).add(relation["type"])
    conflicting_types_by_edge = {
        key: set().union(*(pair for pair in opposing_pairs if pair <= types))
        for key, types in types_by_edge.items()
        if any(pair <= types for pair in opposing_pairs)
    }
    kept: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for record in accepted:
        relation = record["relationship"]
        key = (relation["start_node_id"], relation["end_node_id"])
        if relation["type"] not in conflicting_types_by_edge.get(key, set()):
            kept.append(record)
            continue
        held_record = dict(record)
        held_record["review_reason"] = (
            "같은 근거에서 동일 사건의 허용과 금지가 함께 검출됨: "
            "대상 범위 수식어를 수동 분리하기 전까지 승격 보류"
        )
        held.append(held_record)
    return kept, held


def run_direct_extraction(project_root: Path) -> dict[str, Any]:
    """Run the deterministic, no-LLM pass across every canonical Markdown chunk."""
    processed = project_root / "data" / "processed"
    chunks = [
        json.loads(line)
        for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    schema_payload = yaml.safe_load(
        (project_root / "config" / "kg_schema.yaml").read_text(encoding="utf-8")
    )
    manual = _manual_extractions(project_root / "data" / "curated" / "semantic-kg")

    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for chunk in chunks:
        curated = manual.get(str(chunk["chunk_id"]))
        mode = "codex-manual" if curated else "codex-direct-rule"
        facts = (
            CompactExtraction(facts=curated.facts).model_dump(mode="json")["facts"]
            if curated
            else extract_chunk_facts(chunk)
        )
        graph = compact_to_candidate_graph({"facts": facts}, chunk)
        validation = validate_candidates(graph, chunks_by_id, schema_payload)
        accepted.extend(validation.accepted_relationships)
        review.extend(validation.review_relationships)
        rejected.extend(validation.rejected_relationships)
        coverage.append(
            {
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "chunk_sha256": chunk["sha256"],
                "mode": mode,
                "status": "relations_extracted" if facts else "no_high_precision_relation",
                "candidate_relationships": len(graph["relationships"]),
                "accepted_relationships": len(validation.accepted_relationships),
                "review_relationships": len(validation.review_relationships),
                "rejected_relationships": len(validation.rejected_relationships),
            }
        )

    accepted, scope_conflicts = _hold_scope_conflicts(accepted)
    review.extend(scope_conflicts)
    held_by_chunk = {
        chunk_id: sum(
            item["relationship"]["properties"]["chunk_id"] == chunk_id
            for item in scope_conflicts
        )
        for chunk_id in {
            item["relationship"]["properties"]["chunk_id"] for item in scope_conflicts
        }
    }
    for item in coverage:
        held_count = held_by_chunk.get(item["chunk_id"], 0)
        item["accepted_relationships"] -= held_count
        item["review_relationships"] += held_count

    promoted = promote_accepted(accepted)
    for row in promoted:
        row["extractor_prompt_version"] = DIRECT_EXTRACTOR_VERSION
    output_dir = processed / "semantic-kg" / "direct"
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "review.jsonl", review)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    _write_jsonl(output_dir / "promoted.jsonl", promoted)
    _write_jsonl(output_dir / "coverage.jsonl", coverage)
    input_digest = hashlib.sha256(
        "\n".join(f"{item['chunk_id']}:{item['sha256']}" for item in chunks).encode()
    ).hexdigest()
    summary = {
        "provider": "codex-direct-no-llm",
        "extractor_version": DIRECT_EXTRACTOR_VERSION,
        "input_scope": "raw_markdown/doc*.md via validated chunks.jsonl",
        "input_digest": input_digest,
        "selected_chunks": len(chunks),
        "covered_chunks": len(coverage),
        "chunks_with_relationships": sum(bool(item["candidate_relationships"]) for item in coverage),
        "chunks_without_high_precision_relationships": sum(
            not item["candidate_relationships"] for item in coverage
        ),
        "manual_override_chunks": sum(item["mode"] == "codex-manual" for item in coverage),
        "accepted_relationships": len(accepted),
        "review_relationships": len(review),
        "scope_conflict_relationships_held": len(scope_conflicts),
        "rejected_relationships": len(rejected),
        "promoted_claims": len(promoted),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_dir / "run-summary.json", summary)
    return summary
