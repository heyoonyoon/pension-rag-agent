from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Intent(StrEnum):
    DEFINITION = "DEFINITION"
    COMPARISON = "COMPARISON"
    ELIGIBILITY = "ELIGIBILITY"
    PROCEDURE = "PROCEDURE"
    TAX = "TAX"
    CALCULATION = "CALCULATION"
    WITHDRAWAL = "WITHDRAWAL"
    TRANSFER = "TRANSFER"
    CONTRIBUTION = "CONTRIBUTION"
    INVESTMENT_RULE = "INVESTMENT_RULE"
    GENERAL = "GENERAL"


class AnswerStatus(StrEnum):
    ANSWERED = "ANSWERED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class ScopeDecision(StrEnum):
    """Whether scope is known now or must be established from corpus evidence."""

    CONFIRMED = "CONFIRMED"
    EVIDENCE_REQUIRED = "EVIDENCE_REQUIRED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class OntologyLink(BaseModel):
    concept_id: str
    label: str
    ontology_class: str
    matched_alias: str


class QueryPlan(BaseModel):
    normalized_question: str
    intents: list[Intent]
    ontology_links: list[OntologyLink] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    answer_requirements: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    unavailable_capability: str = ""
    premise_check_required: bool = False
    scope_decision: ScopeDecision = ScopeDecision.CONFIRMED
    # Kept for compatibility with callers created before the three-state scope gate.
    in_scope: bool = True


class EvidenceItem(BaseModel):
    source_id: str
    chunk_id: str
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    content: str
    graph_context: list[str] = Field(default_factory=list)
    structured_context: list[str] = Field(default_factory=list)
    retrieval_channels: list[str] = Field(default_factory=list)
    fusion_score: float = 0.0


class EvidenceCitation(BaseModel):
    source_id: str
    chunk_id: str
    evidence_quote: str
    span_id: str | None = None


class AnswerSection(BaseModel):
    topic: str = ""
    text: str
    citations: list[EvidenceCitation]


class GeneratedAnswer(BaseModel):
    sections: list[AnswerSection]


class EvidenceAssessment(BaseModel):
    sufficient: bool
    coverage_score: float = Field(ge=0.0, le=1.0)
    covered_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)
    verified_citations: int = 0
    stage: str = "integrity"
    action: str = "REPAIR"
    repair_sections: list[int] = Field(default_factory=list)
    safe_sections: list[int] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    semantic_review: dict = Field(default_factory=dict)
    model_calls: int = 0


class AgentResult(BaseModel):
    question_id: str
    question: str
    answer_status: AnswerStatus
    answer: str
    retrieved_context: str
    think_trace: str
    citations: list[EvidenceCitation] = Field(default_factory=list)
    retry_count: int = 0
    retrieval_retry_count: int = 0
    generation_retry_count: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error_code: Literal[
        "NONE", "VERIFICATION_ERROR", "PRODUCT_CITATION_ERROR", "PRODUCT_PROTOCOL_ERROR",
        "PRODUCT_PROVIDER_ERROR", "PRODUCT_INTERNAL_ERROR",
        "PREFLIGHT_RETRIEVAL_ERROR",
    ] = "NONE"
