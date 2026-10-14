from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel

from pension_graph.agent_models import (
    AnswerSection,
    EvidenceAssessment,
    EvidenceCitation,
    EvidenceItem,
    GeneratedAnswer,
    QueryPlan,
    VerificationReport,
)
from pension_graph.retrieval import graph_query_terms
from pension_graph.semantic_kg import align_quote_to_source
from pension_graph.structured_rules import (
    extract_measures,
    parse_table_range_hint,
    resolve_table_row,
    row_matches_query_range,
)

SPACE = re.compile(r"\s+")
MONEY_WITH_UNIT = re.compile(
    r"(?:\d[\d,]*(?:\.\d+)?\s*억(?:\s*\d+\s*(?:천|백|십)?\s*만)?|"
    r"\d[\d,]*(?:\.\d+)?\s*(?:천|백|십)?\s*만|"
    r"\d[\d,]*(?:\.\d+)?)\s*원"
)
NUMBER_WITH_UNIT = re.compile(
    MONEY_WITH_UNIT.pattern
    + r"|\d[\d,]*(?:\.\d+)?\s*(?:%|퍼센트|세|년|개월|월|일|회)"
)
HTML_ROW = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
# Applied to stripped individual lines: detect an internal whitespace run
# directly. A greedy whole-line prefix retries every suffix on long non-tables.
LAYOUT_ROW = re.compile(r"\S\s{3,}\S")


class EvidenceSpan(BaseModel):
    span_id: str
    source_id: str
    chunk_id: str
    quote: str


def build_evidence_pack(
    payload: dict[str, Any],
    *,
    plan: QueryPlan | None = None,
    max_items: int = 5,
    max_chars: int = 10_000,
    max_graph_items: int = 3,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    used = 0
    seen_chunks: set[str] = set()
    by_content_hash: dict[str, EvidenceItem] = {}
    query_terms = _query_terms(plan, str(payload.get("query", "")))
    for row in payload.get("results", []):
        chunk_id = str(row.get("chunk_id", ""))
        content = str(row.get("content", "")).strip()
        if not chunk_id or chunk_id in seen_chunks or len(content) < 40:
            continue
        seen_chunks.add(chunk_id)
        content_hash = hashlib.sha256(
            SPACE.sub(" ", content).strip().casefold().encode("utf-8")
        ).hexdigest()
        graph_context = _compact_graph_context(
            [str(value) for value in row.get("graph_context", [])],
            query_terms=query_terms,
            max_items=max_graph_items,
        )
        structured_context = _structured_context(row)
        existing = by_content_hash.get(content_hash)
        if existing is not None:
            existing.retrieval_channels = sorted(
                set(existing.retrieval_channels)
                | {str(value) for value in row.get("retrieval_channels", [])}
            )
            existing.graph_context = _compact_graph_context(
                [*existing.graph_context, *graph_context],
                query_terms=query_terms,
                max_items=max_graph_items,
            )
            existing.structured_context = list(
                dict.fromkeys([*existing.structured_context, *structured_context])
            )[:3]
            continue
        if items and used + len(content) > max_chars:
            continue
        item = EvidenceItem(
            source_id=str(row.get("source_id", "")),
            chunk_id=chunk_id,
            title=str(row.get("title", "")),
            heading_path=[str(value) for value in row.get("heading_path", [])],
            content=content,
            graph_context=graph_context,
            structured_context=structured_context,
            retrieval_channels=[str(value) for value in row.get("retrieval_channels", [])],
            fusion_score=float(row.get("fusion_score", 0.0)),
        )
        items.append(item)
        by_content_hash[content_hash] = item
        used += len(content)
        if len(items) >= max_items:
            break
    return items


def assess_evidence(plan: QueryPlan, evidence: list[EvidenceItem]) -> EvidenceAssessment:
    if not evidence:
        return EvidenceAssessment(
            sufficient=False,
            coverage_score=0.0,
            reasons=["검색된 실질 근거 청크가 없습니다."],
        )
    haystack = SPACE.sub(
        " ",
        " ".join(
            item.content
            + " "
            + " ".join(item.graph_context)
            + " "
            + " ".join(item.structured_context)
            for item in evidence
        ),
    ).casefold()
    covered: list[str] = []
    missing: list[str] = []
    for link in plan.ontology_links:
        if link.label.casefold() in haystack or link.matched_alias.casefold() in haystack:
            covered.append(link.label)
        else:
            missing.append(link.label)

    terms = [term.casefold() for term in graph_query_terms(plan.normalized_question).split()]
    term_hits = sum(term in haystack for term in terms)
    term_score = term_hits / len(terms) if terms else 1.0
    concept_score = len(covered) / len(plan.ontology_links) if plan.ontology_links else term_score
    coverage = round((term_score + concept_score) / 2, 4)
    # Lexical coverage is diagnostic, never a Korean wording-based truth gate.
    sufficient = True
    reasons = ["검색 후보 존재; 사실 충족 여부는 의미 검증에서 판정합니다."]
    return EvidenceAssessment(
        sufficient=sufficient,
        coverage_score=coverage,
        covered_concepts=covered,
        missing_concepts=missing,
        reasons=reasons,
    )


def build_evidence_spans(
    evidence: list[EvidenceItem],
    *,
    plan: QueryPlan | None = None,
    max_spans_per_item: int = 16,
    max_total_chars: int = 6_000,
) -> list[EvidenceSpan]:
    """Keep every source citable; budget only optional fine-grained quotations.

    Every nonempty chunk gets an immutable full-source span outside the optional
    character budget. ``max_spans_per_item`` limits additional granular spans,
    not source visibility. The caller must budget whole evidence bundles before
    this stage: a quotation budget must never silently discard original data.
    """
    if max_spans_per_item < 0 or max_total_chars < 0:
        raise ValueError("Evidence span budgets must be nonnegative")
    query_terms = _query_terms(plan, plan.normalized_question if plan else "")
    source_candidates: list[list[tuple[int, str]]] = []
    full_spans: list[EvidenceSpan] = []
    for item in evidence:
        if not item.content.strip():
            raise ValueError("Empty evidence source cannot have a valid citation span")
        quotes = _source_spans(item.content)
        # Preserve the existing granular ID when the source already is one exact
        # span. Otherwise reserve a budget-independent whole-chunk ID. Integrity
        # verification rebuilds these same IDs with a different optional budget.
        exact_index = next((index for index, quote in enumerate(quotes) if quote == item.content), None)
        full_spans.append(EvidenceSpan(
            span_id=f"{item.chunk_id}:evidence:{exact_index:03d}" if exact_index is not None
                    else f"{item.chunk_id}:evidence:full",
            source_id=item.source_id, chunk_id=item.chunk_id, quote=item.content,
        ))
        ranked_quotes = sorted(
            ((index, quote) for index, quote in enumerate(quotes) if index != exact_index),
            key=lambda value: (
                -_span_relevance_score(value[1], query_terms, plan),
                value[0],
            ),
        )[:max_spans_per_item]
        source_candidates.append(ranked_quotes)

    # Give each source an equal first-pass allowance, then share unused allowance
    # in rounds. A large early chunk can no longer consume every later source's
    # optional quote budget, and full source spans remain available in all cases.
    selected: list[list[tuple[int, str]]] = [[] for _ in evidence]
    pending: list[list[tuple[int, str]]] = [[] for _ in evidence]
    used_chars = 0
    fair_share = max_total_chars // len(evidence) if evidence else 0
    for index, candidates in enumerate(source_candidates):
        item_used = 0
        for candidate in candidates:
            if item_used + len(candidate[1]) <= fair_share:
                selected[index].append(candidate)
                item_used += len(candidate[1])
            else:
                pending[index].append(candidate)
        used_chars += item_used
    while any(pending):
        for index, candidates in enumerate(pending):
            if not candidates:
                continue
            candidate = candidates.pop(0)
            if used_chars + len(candidate[1]) <= max_total_chars:
                selected[index].append(candidate)
                used_chars += len(candidate[1])

    spans: list[EvidenceSpan] = []
    for item, granular, full_span in zip(evidence, selected, full_spans, strict=True):
        for source_index, quote in sorted(granular):
            spans.append(EvidenceSpan(
                    span_id=f"{item.chunk_id}:evidence:{source_index:03d}",
                    source_id=item.source_id, chunk_id=item.chunk_id, quote=quote,
            ))
        spans.append(full_span)
    return spans


def format_evidence_for_model(evidence: list[EvidenceItem], spans: list[EvidenceSpan]) -> str:
    spans_by_chunk: dict[str, list[EvidenceSpan]] = {}
    for span in spans:
        spans_by_chunk.setdefault(span.chunk_id, []).append(span)
    blocks: list[str] = []
    for item in evidence:
        item_spans = spans_by_chunk.get(item.chunk_id, [])
        full_span = next((span for span in item_spans
                          if span.quote == item.content and span.source_id == item.source_id), None)
        if full_span is None:
            # Defensive formatting for callers supplying only granular spans. The
            # normal generator obtains this same ID from build_evidence_spans.
            full_span = build_evidence_spans([item], max_spans_per_item=0)[0]
        graph = "\n".join(f"- {value}" for value in item.graph_context)
        structured = "\n".join(f"- {value}" for value in item.structured_context)
        source_spans = "\n".join(
            f'<span id="{span.span_id}">{span.quote}</span>'
            for span in item_spans if span.span_id != full_span.span_id
        )
        blocks.append(
            f'<evidence source_id="{item.source_id}" chunk_id="{item.chunk_id}">\n'
            f"제목: {item.title}\n"
            f"제목 경로: {' > '.join(item.heading_path)}\n"
            "필수 문맥(표 머리글/각주/적용조건 포함 전체 원문; 전체 구간도 인용 가능):\n"
            f'<span id="{full_span.span_id}">{item.content}</span>\n'
            f"서버 검증 원문 구간:\n{source_spans}\n"
            f"표 구조 힌트(검색용이며 직접 인용 금지):\n{structured}\n"
            f"그래프 검색 힌트(원문 대조 필요):\n{graph}\n"
            "</evidence>"
        )
    return "\n\n".join(blocks)


def resolve_structured_table_answer(
    plan: QueryPlan, evidence: list[EvidenceItem]
) -> GeneratedAnswer | None:
    """Resolve an explicit interval-to-column lookup without domain Q&A rules."""
    question = plan.normalized_question
    # Multiple input measures can refer to different clauses or compared limits.
    # Binding any one of them to an arbitrary range row would be unsafe.
    if len(extract_measures(question)) != 1:
        return None
    candidates: list[tuple[int, str, str, str, str, str, str]] = []
    for item in evidence:
        quotes = _source_spans(item.content)
        for hint in item.graph_context:
            row = parse_table_range_hint(hint)
            if row is None:
                continue
            for resolution in resolve_table_row(question, row):
                quote = _find_source_row_quote(
                    quotes,
                    condition=resolution.condition.value,
                    output=resolution.output.value,
                )
                if quote is None:
                    continue
                candidates.append(
                    (
                        resolution.target_score,
                        resolution.output.header,
                        resolution.output.value,
                        resolution.query_measure.raw,
                        item.source_id,
                        item.chunk_id,
                        quote,
                    )
                )
    if not candidates:
        return None
    best_score = max(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_score]
    outputs = {_normalize_lookup_value(candidate[2]) for candidate in best}
    if len(outputs) != 1:
        return None
    _, target, output, query_value, source_id, chunk_id, quote = best[0]
    return GeneratedAnswer(
        sections=[
            AnswerSection(
                topic=target,
                text=f"질의값 {query_value}에 해당하는 {target}은 {output}입니다.",
                citations=[
                    EvidenceCitation(
                        source_id=source_id,
                        chunk_id=chunk_id,
                        evidence_quote=quote,
                    )
                ],
            )
        ]
    )


def verify_generated_answer(
    generated: GeneratedAnswer,
    evidence: list[EvidenceItem],
    *,
    plan: QueryPlan | None = None,
) -> VerificationReport:
    """Source integrity only. A pass here is NOT a factual/semantic verdict."""
    by_chunk = {item.chunk_id: item for item in evidence}
    allowed_spans = {
        span.span_id: span for span in build_evidence_spans(
            evidence, plan=plan, max_spans_per_item=100_000, max_total_chars=10**9,
        )
    }
    errors: list[str] = []
    repair: list[int] = []
    verified = 0
    if not generated.sections:
        # Empty drafts are reviewed semantically for clarification or absent evidence.
        return VerificationReport(passed=True, action="SEMANTIC_REVIEW")
    for index, section in enumerate(generated.sections):
        section_errors: list[str] = []
        if not section.text.strip():
            section_errors.append("본문이 비어 있습니다.")
        if not section.citations:
            section_errors.append("인용이 없습니다.")
        for citation in section.citations:
            item = by_chunk.get(citation.chunk_id)
            if item is None:
                section_errors.append(f"허용되지 않은 chunk_id: {citation.chunk_id}")
                continue
            if citation.source_id != item.source_id:
                section_errors.append(f"source_id와 chunk_id가 불일치합니다: {citation.chunk_id}")
                continue
            aligned = align_quote_to_source(citation.evidence_quote, item.content)
            if aligned is None or not citation.evidence_quote.strip():
                section_errors.append(f"원문에 없는 인용문입니다: {citation.chunk_id}")
                continue
            if citation.span_id is not None:
                span = allowed_spans.get(citation.span_id)
                if (
                    span is None or span.chunk_id != citation.chunk_id
                    or span.source_id != citation.source_id or span.quote != aligned
                ):
                    section_errors.append(f"원문 구간 ID/인용문 불일치: {citation.span_id}")
                    continue
            citation.evidence_quote = aligned
            verified += 1
        if section_errors:
            repair.append(index)
            errors.extend(f"섹션 {index}: {error}" for error in section_errors)
    return VerificationReport(
        passed=not errors, errors=errors, verified_citations=verified,
        stage="integrity", action="REPAIR" if errors else "SEMANTIC_REVIEW",
        repair_sections=repair,
    )


def compose_answer(generated: GeneratedAnswer) -> tuple[str, list[EvidenceCitation]]:
    paragraphs: list[str] = []
    citations: list[EvidenceCitation] = []
    for section in generated.sections:
        ids = list(dict.fromkeys(citation.chunk_id for citation in section.citations))
        topic = f"{section.topic.strip()}\n" if section.topic.strip() else ""
        paragraphs.append(f"{topic}{section.text.strip()} [{', '.join(ids)}]")
        citations.extend(section.citations)
    unique = {(item.source_id, item.chunk_id, item.evidence_quote): item for item in citations}
    return "\n\n".join(paragraphs), list(unique.values())


def render_retrieved_context(evidence: list[EvidenceItem]) -> str:
    """Preserve the budgeted evidence pack, including late quotes and table notes.

    The retrieval stage chooses whole evidence items within its context budget.
    Truncating them again here would hide sources used to generate and verify the
    answer from the evaluation API's retrieved_context field.
    """
    return "\n\n".join(
        f"[{item.chunk_id}] {item.title}\n{item.content}" for item in evidence
    )


def derive_verified_calculations(
    plan: QueryPlan, evidence: list[EvidenceItem]
) -> list[str]:
    """Derive only arithmetic explicitly licensed by a retrieved source formula."""
    question = plan.normalized_question
    evidence_text = SPACE.sub(" ", " ".join(item.content for item in evidence))
    question_money = MONEY_WITH_UNIT.findall(question)
    values: list[str] = []

    year_match = re.search(r"(?:연금)?수령연차(?:가|는|\s)*([0-9]+)\s*년?", question)
    has_pension_limit_formula = (
        "11" in evidence_text
        and "연금수령연차" in evidence_text
        and "120%" in evidence_text
        and "평가액" in evidence_text
    )
    if year_match and question_money and has_pension_limit_formula:
        principal = parse_korean_won(question_money[0])
        year = int(year_match.group(1))
        denominator = 11 - year
        if principal is not None and denominator > 0:
            result = principal * 120 // (denominator * 100)
            values.extend(_won_representations(result))

    formula_compact = re.sub(r"\s+", "", evidence_text).casefold()
    percent_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", question)
    if (
        question_money
        and percent_match
        and any(marker in formula_compact for marker in ("r=a×b", "r=a*b", "예상액(a)"))
    ):
        principal = parse_korean_won(question_money[0])
        if principal is not None:
            rate = float(percent_match.group(1))
            result = round(principal * rate / 100)
            values.extend(_won_representations(result))

    return list(dict.fromkeys(values))


def _won_representations(value: int) -> list[str]:
    values = [f"{value:,}원"]
    if value % 10_000 == 0:
        values.append(f"{value // 10_000:,}만원")
    return values


def parse_korean_won(value: str) -> int | None:
    compact = SPACE.sub("", value).replace(",", "")
    if not compact.endswith("원"):
        return None
    compact = compact[:-1]
    if not compact:
        return None
    try:
        total = Decimal(0)
        if "억" in compact:
            billion, compact = compact.split("억", 1)
            total += Decimal(billion) * 100_000_000
        if "만" in compact:
            ten_thousand, compact = compact.split("만", 1)
            total += _parse_small_korean_number(ten_thousand) * 10_000
        if compact:
            total += _parse_small_korean_number(compact)
        return int(total) if total.is_finite() and total == total.to_integral_value() else None
    except (TypeError, ValueError, InvalidOperation):
        return None


def _parse_small_korean_number(value: str) -> Decimal:
    if not value:
        return Decimal(1)
    if value.replace(".", "", 1).isdigit():
        return Decimal(value)
    total = Decimal(0)
    remaining = value
    for unit, multiplier in (("천", 1_000), ("백", 100), ("십", 10)):
        if unit not in remaining:
            continue
        prefix, remaining = remaining.split(unit, 1)
        total += (Decimal(prefix) if prefix else Decimal(1)) * multiplier
    if remaining:
        total += Decimal(remaining)
    return total


def _span_relevance_score(
    quote: str, query_terms: tuple[str, ...], plan: QueryPlan | None
) -> int:
    score = sum(term in quote.casefold() for term in query_terms)
    if plan is not None and row_matches_query_range(plan.normalized_question, quote):
        score += 10
    return score


def _find_source_row_quote(
    quotes: list[str], *, condition: str, output: str
) -> str | None:
    normalized_condition = _normalize_lookup_value(condition)
    normalized_output = _normalize_lookup_value(output)
    for quote in quotes:
        normalized_quote = _normalize_lookup_value(quote)
        if normalized_condition in normalized_quote and normalized_output in normalized_quote:
            return quote
    return None


def _normalize_lookup_value(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _source_spans(content: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", content):
        block = block.strip()
        if not block:
            continue
        candidates: list[str]
        if block.startswith("|"):
            rows = [
                line.strip()
                for line in block.splitlines()
                if line.strip() and not re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())
            ]
            candidates = rows
        elif "<table" in block.casefold():
            rows = [match.group(0).strip() for match in HTML_ROW.finditer(block)]
            candidates = rows or [block]
        else:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            layout_rows = sum(bool(LAYOUT_ROW.search(line)) for line in lines)
            candidates = lines if layout_rows >= 2 else [block]
        for candidate in candidates:
            if len(candidate) < 8 or candidate in seen:
                continue
            seen.add(candidate)
            spans.append(candidate)
    return spans


def _query_terms(plan: QueryPlan | None, fallback_query: str) -> tuple[str, ...]:
    values: list[str] = []
    query = plan.normalized_question if plan is not None else fallback_query
    values.extend(graph_query_terms(query).split())
    if plan is not None:
        for link in plan.ontology_links:
            values.extend((link.label, link.matched_alias))
    normalized = [value.casefold().strip() for value in values if len(value.strip()) >= 2]
    return tuple(dict.fromkeys(normalized))


def _compact_graph_context(
    values: list[str], *, query_terms: tuple[str, ...], max_items: int
) -> list[str]:
    unique: list[tuple[int, str]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        normalized = SPACE.sub(" ", value).strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append((index, normalized))
    ranked = sorted(
        unique,
        key=lambda value: (
            -sum(term in value[1].casefold() for term in query_terms),
            value[0],
        ),
    )[:max_items]
    return [value for _, value in ranked]


def _structured_context(row: dict[str, Any]) -> list[str]:
    if row.get("segment_kind") != "table_row":
        return []
    value = SPACE.sub(" ", str(row.get("matched_segment", ""))).strip()
    return [value] if value else []
