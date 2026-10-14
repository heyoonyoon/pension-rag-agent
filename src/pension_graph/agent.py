from __future__ import annotations

import json
import logging
import math
import os
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from time import perf_counter
from typing import Any, Protocol, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import ValidationError

from pension_graph.agent_models import (
    AgentResult,
    AnswerStatus,
    EvidenceItem,
    GeneratedAnswer,
    QueryPlan,
    VerificationReport,
)
from pension_graph.clova import ClovaAnswerGenerator
from pension_graph.deadline import bind_deadline, check_deadline
from pension_graph.evidence import (
    assess_evidence,
    build_evidence_pack,
    compose_answer,
    render_retrieved_context,
    resolve_structured_table_answer,
)
from pension_graph.hcx_metrics import verification_error_category
from pension_graph.hybrid_retrieval import retrieve_hybrid_context
from pension_graph.query_planner import (
    analyze_question,
    contrast_query,
    retry_query,
    semantic_skeleton_query,
)
from pension_graph.retrieval import graph_query_terms
from pension_graph.vector_store import VectorSettings
from pension_graph.verification import SemanticAnswerVerifier


class AnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        plan: QueryPlan,
        evidence: list[EvidenceItem],
        feedback: list[str] | None = None,
    ) -> GeneratedAnswer: ...


Retriever = Callable[[str, int], dict[str, Any]]


class AnswerVerifier(Protocol):
    def verify(
        self, *, question: str, plan: QueryPlan, evidence: list[EvidenceItem],
        generated: GeneratedAnswer,
    ) -> VerificationReport: ...


class AgentState(TypedDict, total=False):
    question_id: str
    question: str
    plan: dict[str, Any]
    active_query: str
    retrieval_payload: dict[str, Any]
    evidence: list[dict[str, Any]]
    assessment: dict[str, Any]
    generated: dict[str, Any]
    verification: dict[str, Any]
    retry_count: int
    generation_retry_count: int
    status: str
    answer: str
    citations: list[dict[str, Any]]
    trace: list[str]
    timings_ms: dict[str, float]


class PensionSystemAgent:
    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
        verifier: AnswerVerifier | None = None,
        planner: Callable[[str], QueryPlan] | None = None,
        max_retries: int = 2,
        max_generation_retries: int = 1,
    ) -> None:
        self.retriever = retriever or _environment_retriever
        self.planner = planner or analyze_question
        self.generator = generator or ClovaAnswerGenerator()
        self.verifier = verifier or SemanticAnswerVerifier(
            self.generator if isinstance(self.generator, ClovaAnswerGenerator)
            else ClovaAnswerGenerator()
        )
        self.max_retries = max_retries
        self.max_generation_retries = max_generation_retries
        self.graph = self._build_graph()

    def answer(self, *, question_id: str, question: str) -> AgentResult:
        check_deadline()
        started = perf_counter()
        state = self.graph.invoke(
            {
                "question_id": question_id,
                "question": question,
                "retry_count": 0,
                "generation_retry_count": 0,
                "trace": [],
                "timings_ms": {},
            },
            config={"configurable": {"thread_id": f"{question_id}:{uuid.uuid4()}"}},
        )
        evidence = [
            EvidenceItem.model_validate(item) for item in state.get("evidence", [])
        ]
        check_deadline()
        timings = dict(state.get("timings_ms", {}))
        timings["total"] = _milliseconds_since(started)
        timing_trace = "지연시간(" + ", ".join(
            f"{name}={value:.1f}ms" for name, value in timings.items()
        ) + ")"
        retrieval_retries = state.get("retry_count", 0)
        generation_retries = state.get("generation_retry_count", 0)
        return AgentResult(
            question_id=question_id,
            question=question,
            answer_status=AnswerStatus(state["status"]),
            answer=state["answer"],
            retrieved_context=render_retrieved_context(evidence),
            think_trace=" → ".join([*state.get("trace", []), timing_trace]),
            citations=state.get("citations", []),
            retry_count=retrieval_retries + generation_retries,
            retrieval_retry_count=retrieval_retries,
            generation_retry_count=generation_retries,
            timings_ms=timings,
            error_code=(
                "VERIFICATION_ERROR" if state.get("verification", {}).get("action") == "ERROR"
                else "NONE"
            ),
        )

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node(
            "retrieve",
            self._retrieve,
            retry_policy=RetryPolicy(max_attempts=2, retry_on=ConnectionError),
        )
        workflow.add_node(
            "assess",
            self._assess,
        )
        workflow.add_node("structured_lookup", self._structured_lookup)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node(
            "generate",
            self._generate,
            retry_policy=RetryPolicy(
                initial_interval=1.0,
                backoff_factor=2.0,
                max_interval=8.0,
                max_attempts=2,
                retry_on=_is_transient_hcx_error,
            ),
        )
        workflow.add_node("verify", self._verify)
        workflow.add_node("prepare_generation_retry", self._prepare_generation_retry)
        workflow.add_node("finalize", self._finalize)
        workflow.add_node("clarify", self._clarify)
        workflow.add_node("out_of_scope", self._out_of_scope)
        workflow.add_node("abstain", self._abstain)

        workflow.add_edge(START, "analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analysis,
            {
                "retrieve": "retrieve",
                "clarify": "clarify",
                "out_of_scope": "out_of_scope",
                "abstain": "abstain",
            },
        )
        workflow.add_edge("retrieve", "assess")
        workflow.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {
                "structured_lookup": "structured_lookup",
                "rewrite": "rewrite",
                "out_of_scope": "out_of_scope",
                "abstain": "abstain",
            },
        )
        workflow.add_conditional_edges(
            "structured_lookup",
            self._route_after_structured_lookup,
            {"verify": "verify", "generate": "generate"},
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {
                "finalize": "finalize",
                "regenerate": "prepare_generation_retry",
                "rewrite": "rewrite",
                "clarify": "clarify",
                "out_of_scope": "out_of_scope",
                "abstain": "abstain",
            },
        )
        workflow.add_edge("prepare_generation_retry", "generate")
        for terminal in ("finalize", "clarify", "out_of_scope", "abstain"):
            workflow.add_edge(terminal, END)
        return workflow.compile()

    def _analyze(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        plan = self.planner(state["question"])
        concepts = ",".join(link.label for link in plan.ontology_links) or "없음"
        intents = ",".join(intent.value for intent in plan.intents)
        scope = plan.scope_decision.value
        return {
            "plan": plan.model_dump(mode="json"),
            "active_query": plan.normalized_question,
            "trace": [
                *state.get("trace", []),
                f"질의분석({intents}; 온톨로지={concepts}; 범위={scope})",
            ],
            "timings_ms": _add_timing(state, "analyze", started),
        }

    def _route_after_analysis(self, state: AgentState) -> str:
        # Lexical planning is a retrieval hint, never authority to refuse an
        # unfamiliar spelling or infer missing personal inputs before evidence.
        # Semantic review owns scope/clarification decisions after retrieval.
        return "retrieve"

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        plan = QueryPlan.model_validate(state["plan"])
        queries = [state["active_query"]]
        focused = state.get("verification", {}).get("search_queries", [])
        if state.get("retry_count", 0) > 0 and focused:
            queries = focused[:2]
        if state.get("retry_count", 0) == 0 and len(plan.answer_requirements) > 1:
            concept_prefix = " ".join(link.label for link in plan.ontology_links)
            queries = [
                contrast_query(f"{concept_prefix} {item}".strip())
                for item in plan.answer_requirements[:3]
            ]
        elif state.get("retry_count", 0) == 0 and plan.premise_check_required:
            counter_query = contrast_query(state["active_query"])
            if counter_query != state["active_query"]:
                skeleton_query = semantic_skeleton_query(plan)
                queries = list(dict.fromkeys([counter_query, skeleton_query]))
        if len(queries) == 1:
            payload = _rerank_payload(queries[0], self.retriever(queries[0], 10))
        else:
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                payloads = list(executor.map(
                    bind_deadline(lambda query: self.retriever(query, 10)), queries,
                ))
            payloads = [
                _rerank_payload(query, payload)
                for query, payload in zip(queries, payloads, strict=True)
            ]
            payload = _merge_retrieval_payloads(plan.normalized_question, payloads, limit=10)
        evidence = build_evidence_pack(
            payload,
            plan=plan,
            max_items=min(8, max(5, len(queries) * 3)),
        )
        if state.get("generated") and state.get("verification"):
            # Preserve source identity for locked, previously supported answer blocks.
            retained = {
                item["chunk_id"]: EvidenceItem.model_validate(item)
                for item in state.get("evidence", [])
            }
            retained.update({item.chunk_id: item for item in evidence})
            evidence = list(retained.values())
        elapsed = _milliseconds_since(started)
        diagnostics = payload.get("diagnostics", {})
        channel_timings = diagnostics.get("timings_ms", {})
        timing_detail = ""
        if isinstance(channel_timings, dict) and channel_timings:
            timing_detail = "; " + ", ".join(
                f"{name}={float(value):.1f}ms"
                for name, value in channel_timings.items()
            )
        return {
            "retrieval_payload": payload,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "trace": [
                *state.get("trace", []),
                (
                    f"하이브리드검색(질의={len(queries)}; "
                    f"근거={len(evidence)}; {elapsed:.1f}ms{timing_detail})"
                ),
            ],
            "timings_ms": _add_timing(state, "retrieve", started),
        }

    def _assess(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        plan = QueryPlan.model_validate(state["plan"])
        evidence = [
            EvidenceItem.model_validate(item) for item in state.get("evidence", [])
        ]
        assessment = assess_evidence(plan, evidence)
        return {
            "assessment": assessment.model_dump(mode="json"),
            "trace": [
                *state.get("trace", []),
                (
                    "근거게이트("
                    f"충족={assessment.sufficient}; coverage={assessment.coverage_score:.2f})"
                ),
            ],
            "timings_ms": _add_timing(state, "assess", started),
        }

    def _route_after_assessment(self, state: AgentState) -> str:
        if state["assessment"]["sufficient"]:
            return "structured_lookup"
        if state.get("retry_count", 0) < self.max_retries:
            return "rewrite"
        # Empty retrieval cannot establish that the question is out of scope.
        return "abstain"

    def _structured_lookup(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        plan = QueryPlan.model_validate(state["plan"])
        evidence = [
            EvidenceItem.model_validate(item) for item in state.get("evidence", [])
        ]
        # An existing failed draft must be repaired, never overwritten by a table shortcut.
        generated = None if state.get("verification") else resolve_structured_table_answer(
            plan, evidence
        )
        trace_step = "구조형표조회(적중)" if generated is not None else "구조형표조회(해당없음)"
        result: dict[str, Any] = {
            "trace": [*state.get("trace", []), trace_step],
            "timings_ms": _add_timing(state, "structured_lookup", started),
        }
        if generated is not None:
            result["generated"] = generated.model_dump(mode="json")
        return result

    def _route_after_structured_lookup(self, state: AgentState) -> str:
        return "verify" if state.get("generated") and not state.get("verification") else "generate"

    def _rewrite(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        attempt = state.get("retry_count", 0) + 1
        plan = QueryPlan.model_validate(state["plan"])
        focused = state.get("verification", {}).get("search_queries", [])
        rewritten = focused[0] if focused else retry_query(plan, attempt)
        return {
            "retry_count": attempt,
            "active_query": rewritten,
            "trace": [*state.get("trace", []), f"재검색질의({attempt})"],
            "timings_ms": _add_timing(state, "rewrite", started),
        }

    def _generate(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        plan = QueryPlan.model_validate(state["plan"])
        evidence = [
            EvidenceItem.model_validate(item) for item in state.get("evidence", [])
        ]
        feedback = state.get("verification", {}).get("errors", [])
        if state.get("generated") and state.get("verification") and hasattr(self.generator, "repair"):
            generated = self.generator.repair(
                question=state["question"], plan=plan, evidence=evidence,
                previous=GeneratedAnswer.model_validate(state["generated"]),
                repair_sections=state["verification"].get("repair_sections", []),
                feedback=feedback,
            )
        else:
            generated = self.generator.generate(
                question=state["question"], plan=plan, evidence=evidence, feedback=feedback,
            )
        return {
            "generated": generated.model_dump(mode="json"),
            "trace": [
                *state.get("trace", []),
                f"HCX답변생성({_milliseconds_since(started):.1f}ms)",
            ],
            "timings_ms": _add_timing(state, "generate", started),
        }

    def _verify(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        generated = GeneratedAnswer.model_validate(state["generated"])
        evidence = [
            EvidenceItem.model_validate(item) for item in state.get("evidence", [])
        ]
        plan = QueryPlan.model_validate(state["plan"])
        report = self.verifier.verify(
            question=state["question"], generated=generated, evidence=evidence, plan=plan,
        )
        if report.action == "ERROR":
            logging.getLogger("uvicorn.error").warning(json.dumps({
                "event": "verification_error",
                "category": verification_error_category(report.errors),
                "model_calls": report.model_calls,
                "elapsed_ms": _milliseconds_since(started),
            }))
        return {
            "verification": report.model_dump(mode="json"),
            "trace": [
                *state.get("trace", []),
                (
                    f"근거의미검증(v2; 단계={report.stage}; 통과={report.passed}; "
                    f"조치={report.action}; 인용={report.verified_citations}; "
                    f"{_milliseconds_since(started):.1f}ms)"
                ),
                *([f"검증판정요약({'; '.join(report.errors)[:450]})"] if report.errors else []),
            ],
            "timings_ms": _add_timing(state, "verify", started),
        }

    def _route_after_verification(self, state: AgentState) -> str:
        if (
            state["verification"]["passed"]
            and state["verification"].get("stage") == "semantic"
            and state["verification"].get("action") == "ACCEPT"
        ):
            return "finalize"
        action = state["verification"].get("action")
        if action == "CLARIFY":
            return "clarify"
        if action == "OUT_OF_SCOPE":
            return "out_of_scope"
        if action == "RETRIEVE" and state.get("retry_count", 0) < self.max_retries:
            return "rewrite"
        if action == "REPAIR" and (
            state.get("generation_retry_count", 0) < self.max_generation_retries
        ):
            return "regenerate"
        return "abstain"

    def _prepare_generation_retry(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        attempt = state.get("generation_retry_count", 0) + 1
        errors = state.get("verification", {}).get("errors", [])
        return {
            "generation_retry_count": attempt,
            "trace": [
                *state.get("trace", []),
                (
                    f"답변교정({attempt}; 검색근거재사용; "
                    f"오류={len(errors)})"
                ),
            ],
            "timings_ms": _add_timing(state, "generation_retry_prepare", started),
        }

    def _finalize(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        generated = GeneratedAnswer.model_validate(state["generated"])
        answer, citations = compose_answer(generated)
        return {
            "status": AnswerStatus.ANSWERED.value,
            "answer": answer,
            "citations": [item.model_dump(mode="json") for item in citations],
            "trace": [*state.get("trace", []), "최종응답"],
            "timings_ms": _add_timing(state, "finalize", started),
        }

    def _clarify(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        plan = QueryPlan.model_validate(state["plan"])
        questions = state.get("verification", {}).get("clarification_questions", [])
        needed = ", ".join(plan.missing_information)
        answer = (
            "정확한 안내를 위해 추가 정보가 필요합니다.\n" + "\n".join(questions)
            if questions else f"정확히 계산하려면 다음 정보가 필요합니다: {needed}."
        )
        citations = []
        safe = state.get("verification", {}).get("safe_sections", [])
        if safe and state.get("generated"):
            draft = GeneratedAnswer.model_validate(state["generated"])
            partial = GeneratedAnswer(sections=[draft.sections[index] for index in safe])
            supported, cited = compose_answer(partial)
            answer = "확인된 범위의 부분 안내입니다.\n\n" + supported + "\n\n" + answer
            citations = [item.model_dump(mode="json") for item in cited]
        return {
            "status": AnswerStatus.NEEDS_CLARIFICATION.value,
            "answer": answer,
            "citations": citations,
            "trace": [*state.get("trace", []), "추가정보요청"],
            "timings_ms": _add_timing(state, "clarify", started),
        }

    def _out_of_scope(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        return {
            "status": AnswerStatus.OUT_OF_SCOPE.value,
            "answer": "이 Agent는 연금제도·계좌·세금·가입 및 업무 절차 질문만 답변합니다.",
            "citations": [],
            "trace": [*state.get("trace", []), "범위외응답"],
            "timings_ms": _add_timing(state, "out_of_scope", started),
        }

    def _abstain(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        verification_failed = bool(state.get("verification"))
        report = state.get("verification", {})
        status = AnswerStatus.INSUFFICIENT_EVIDENCE.value
        citations = []
        if report.get("action") == "ERROR":
            answer = "근거 의미 검증 서비스에 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            trace_step = "의미검증서비스오류"
            status = AnswerStatus.SYSTEM_ERROR.value
        elif verification_failed:
            answer = (
                "검색 근거는 확인했지만 안전한 인용 검증을 통과한 답변을 "
                "생성하지 못했습니다."
            )
            trace_step = "검증실패응답"
        else:
            answer = (
                "제공된 연금제도 자료에서 답변을 뒷받침할 충분한 근거를 "
                "확인하지 못했습니다."
            )
            trace_step = "근거부족응답"
        safe = report.get("safe_sections", [])
        if safe and state.get("generated") and report.get("action") not in ("ERROR", "OUT_OF_SCOPE"):
            draft = GeneratedAnswer.model_validate(state["generated"])
            partial = GeneratedAnswer(sections=[draft.sections[index] for index in safe])
            supported, cited = compose_answer(partial)
            answer = "아래는 확인된 범위의 부분 답변입니다. 전체 질문은 해결되지 않았습니다.\n\n"
            answer += supported
            answer += "\n\n나머지 쟁점은 현재 근거와 검증 결과만으로 확정할 수 없습니다."
            citations = [item.model_dump(mode="json") for item in cited]
            trace_step = "검증된독립블록부분응답"
        return {
            "status": status,
            "answer": answer,
            "citations": citations,
            "trace": [*state.get("trace", []), trace_step],
            "timings_ms": _add_timing(state, "abstain", started),
        }


def _environment_retriever(query: str, limit: int) -> dict[str, Any]:
    return retrieve_hybrid_context(
        query=query,
        qdrant_settings=VectorSettings.from_env(),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7688"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        limit=limit,
    )


def _merge_retrieval_payloads(
    query: str, payloads: list[dict[str, Any]], *, limit: int
) -> dict[str, Any]:
    """Fuse requirement-specific rankings while preserving all provenance fields."""
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}
    for payload_index, payload in enumerate(payloads):
        for rank, raw_row in enumerate(payload.get("results", []), start=1):
            row = dict(raw_row)
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            if chunk_id not in merged:
                merged[chunk_id] = row
                merged[chunk_id]["matched_requirements"] = [payload_index]
                continue
            existing = merged[chunk_id]
            existing["matched_requirements"] = list(
                dict.fromkeys([*existing.get("matched_requirements", []), payload_index])
            )
            for field in ("graph_context", "structured_context", "retrieval_channels"):
                existing[field] = list(
                    dict.fromkeys([*existing.get(field, []), *row.get(field, [])])
                )
    rows_by_id: dict[str, dict[str, Any]] = {}
    for chunk_id, row in merged.items():
        result = dict(row)
        result["fusion_score"] = scores[chunk_id]
        rows_by_id[chunk_id] = result
    ordered_ids: list[str] = []
    max_rank = max((len(payload.get("results", [])) for payload in payloads), default=0)
    for rank in range(max_rank):
        for payload in payloads:
            payload_rows = payload.get("results", [])
            if rank >= len(payload_rows):
                continue
            chunk_id = str(payload_rows[rank].get("chunk_id", ""))
            if chunk_id and chunk_id not in ordered_ids:
                ordered_ids.append(chunk_id)
    rows = [rows_by_id[chunk_id] for chunk_id in ordered_ids if chunk_id in rows_by_id]
    return {
        "query": query,
        "results": rows[:limit],
        "diagnostics": {
            "retrieval_queries": len(payloads),
            "fusion": "requirement-level reciprocal rank fusion",
        },
    }


def _rerank_payload(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Second-stage lexical rerank over hybrid candidates.

    Dense/sparse/graph RRF is optimized for recall. This small deterministic stage
    moves candidates containing the question's discriminative terms and phrases to
    the front without discarding any retrieved provenance.
    """
    rows = [dict(row) for row in payload.get("results", [])]
    if len(rows) < 2:
        return payload
    terms = [
        term.casefold()
        for term in graph_query_terms(query).split()
        if len(term) >= 2 and not re.fullmatch(r"[0-9.,%]+", term)
    ]
    terms = list(dict.fromkeys(terms))
    if not terms:
        return payload

    blobs = [
        " ".join(
            [
                str(row.get("title", "")),
                str(row.get("content", "")),
                *[str(value) for value in row.get("graph_context", [])],
                *[str(value) for value in row.get("structured_context", [])],
            ]
        ).casefold()
        for row in rows
    ]
    document_frequency = {
        term: sum(term in blob for blob in blobs)
        for term in terms
    }
    phrase_terms = [
        f"{left} {right}"
        for left, right in pairwise(terms)
        if len(left) + len(right) >= 5
    ]

    scored: list[tuple[float, int, dict[str, Any]]] = []
    total = len(rows)
    for rank, (row, blob) in enumerate(zip(rows, blobs, strict=True), start=1):
        lexical = sum(
            math.log((total + 1) / (document_frequency[term] + 0.5)) + 1.0
            for term in terms
            if term in blob
        )
        phrase_bonus = sum(2.0 for phrase in phrase_terms if phrase in blob)
        # The original hybrid rank remains a tie-breaker and a weak semantic prior.
        score = lexical + phrase_bonus + (1.0 / (10 + rank))
        row["rerank_score"] = round(score, 6)
        scored.append((score, rank, row))

    reranked = [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))]
    result = dict(payload)
    result["results"] = reranked
    diagnostics = dict(result.get("diagnostics", {}))
    diagnostics["reranker"] = "query-idf lexical over hybrid candidates"
    result["diagnostics"] = diagnostics
    return result


def _is_transient_hcx_error(exc: Exception) -> bool:
    if isinstance(exc, (json.JSONDecodeError, ValidationError)):
        return True
    if isinstance(exc, (ConnectionError, httpx.TransportError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


def _milliseconds_since(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _add_timing(state: AgentState, name: str, started: float) -> dict[str, float]:
    timings = dict(state.get("timings_ms", {}))
    timings[name] = round(timings.get(name, 0.0) + _milliseconds_since(started), 3)
    return timings
