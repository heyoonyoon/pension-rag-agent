from __future__ import annotations

import re

from pension_graph.agent_models import Intent, OntologyLink, QueryPlan, ScopeDecision
from pension_graph.concepts import Concept, load_query_concepts
from pension_graph.config import Paths

SPACE = re.compile(r"\s+")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

INTENT_TERMS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (Intent.COMPARISON, ("차이", "비교", "다른", "장단점", "vs")),
    (Intent.ELIGIBILITY, ("가입", "대상", "자격", "가능한 사람", "누가")),
    (Intent.PROCEDURE, ("방법", "절차", "신청", "서류")),
    (Intent.TAX, ("세금", "세율", "과세", "비과세", "세액공제", "소득세")),
    (Intent.CALCULATION, ("계산", "얼마", "공제액", "수령액", "몇 년")),
    (Intent.WITHDRAWAL, ("중도인출", "중도 인출", "해지", "인출")),
    (Intent.TRANSFER, ("계약이전", "계좌이체", "이전", "이수관")),
    (Intent.CONTRIBUTION, ("납입", "부담금", "입금")),
    (Intent.INVESTMENT_RULE, ("위험자산", "운용", "투자 가능", "매수 가능")),
    (Intent.DEFINITION, ("뭐야", "뜻", "정의", "개념", "설명해")),
)

DOMAIN_TERMS = (
    "연금",
    "퇴직",
    "IRP",
    "DB",
    "DC",
    "세액공제",
    "중도인출",
    "디폴트옵션",
    "부담금",
    "퇴직소득세",
    "연금소득세",
)
PRODUCT_RECOMMENDATION_TERMS = (
    "상품 추천",
    "펀드 추천",
    "종목 추천",
    "하나 골라",
    "뭘 살",
    "무엇을 살",
    "어디에 투자",
    "수익률 좋은",
    "수익률 높은",
    "어떤 펀드",
    "어느 펀드",
    "포트폴리오 추천",
    "포트폴리오를 짜",
    "포트폴리오 짜",
)
PRODUCT_FACT_TERMS = (
    "합성총보수",
    "총보수",
    "증권거래비용",
    "운용설정액",
    "운용규모",
    "수익률 순위",
    "위험등급",
    "몇 종목",
    "상품 구성",
    "어떤 두 상품",
    "구성해",
    "발행사",
    "원금 이상",
)
INTRINSIC_PRODUCT_FACT_TERMS = (
    "합성총보수",
    "총보수",
    "증권거래비용",
    "운용설정액",
    "운용규모",
    "수익률 순위",
    "위험등급",
)
PRODUCT_NOUNS = ("펀드", "ETF", "ETN", "리츠", "상품", "포트폴리오")
RECOMMENDATION_CUES = ("추천", "골라", "선택해", "짜줘", "구성해")
PREMISE_CUES = ("무조건", "아무 때나", "항상", "전부", "절대", "당연히")
PERSONAL_CALCULATION_CUES = ("내가", "제가", "저는", "받을 수", "돌려받", "실제 공제")
PERSONAL_CUES = ("내 ", "내가", "제 ", "제가", "저는", "저도")
PERSONAL_DECISION_CUES = ("판단해", "확인해", "봐줘", "가능한지")
PRIVATE_DATA_TERMS = ("현재 잔액", "계좌번호", "거래내역을 조회", "내 계좌 조회")
LIVE_EXTERNAL_TERMS = ("최신 판례", "판결문 원문", "사건번호", "실시간 판례")
CURRENT_APP_OPERATION_TERMS = ("현재 장애", "복구 예정", "서비스 장애 현황")
CONTRAST_TERM_PAIRS = (
    (("가능", "할 수 있", "되나요", "돼요"), "불가 금지 제한 예외"),
    (("불가능", "할 수 없", "안 되"), "가능 허용"),
    (("계속", "유지"), "중단 중지"),
    (("전액", "전체", "전부"), "일부 제외 한도"),
    (("한 종목", "일부"), "전체 전 종목"),
    (("포함", "적용"), "제외 미적용 예외"),
    (("자동",), "직접 승인 수동"),
    (("휴일", "공휴일", "비영업일"), "전 영업일 익 영업일 월 연도 변경"),
)
RELATION_QUERY_TERMS = (
    "소급",
    "변경",
    "전환",
    "중도인출",
    "인출",
    "지급",
    "가입",
    "납입",
    "압류",
    "상환",
    "해지",
    "과세",
    "세금",
    "운용",
    "매수",
    "이전",
    "신청",
    "발급",
    "보호",
    "담보",
    "지정",
    "적용",
    "제외",
)


def _match_alias(text: str, concept: Concept) -> str | None:
    for alias in sorted(concept.aliases, key=len, reverse=True):
        if alias.isascii() and alias.replace("-", "").isalnum():
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                text,
                re.IGNORECASE,
            ):
                return alias
        elif alias.casefold() in text.casefold():
            return alias
    return None


def _has_intent_term(text: str, term: str) -> bool:
    # Avoid treating the middle of "정해지는" as an account termination.
    if term == "해지":
        return re.search(r"(?<!정)해지", text) is not None
    if term == "인출" and any(
        phrase in text for phrase in ("비정기 인출", "연금 인출", "한도인출")
    ):
        return False
    if term == "운용" and "운용설정액" in text:
        return False
    return term in text


def _is_definition_question(text: str) -> bool:
    """Recognize definitions without treating every object-list question as one."""
    return bool(
        re.search(r"(?:이란|란)(?:\s|\?|$)", text)
        or re.search(r"무엇(?:인가요|입니까|인가|인지)(?:\?|\.|$)", text)
    )


def _extract_answer_requirements(text: str) -> list[str]:
    """Keep independently requested clauses visible to generation and verification."""
    requirements: list[str] = []
    for sentence in re.split(r"[?!]+", text):
        sentence = sentence.strip(" .,")
        if not sentence:
            continue
        if any(cue in sentence for cue in ("각각", "한꺼번에")) and "," in sentence:
            clauses = [part.strip(" .,") for part in sentence.split(",") if part.strip()]
            requirements.extend(clauses)
        elif "," in sentence:
            clauses = [part.strip(" .,") for part in sentence.split(",") if part.strip()]
            predicate_cues = ("해야", "가능", "되", "생겨", "알려", "다르", "제외")
            if len(clauses) == 2 and all(
                any(cue in clause for cue in predicate_cues) for clause in clauses
            ):
                requirements.extend(clauses)
            else:
                requirements.append(sentence)
        else:
            requirements.append(sentence)
    return list(dict.fromkeys(requirements)) or [text]


def _product_agent_request(text: str) -> bool:
    if any(term in text for term in PRODUCT_RECOMMENDATION_TERMS):
        return True
    if any(term in text for term in INTRINSIC_PRODUCT_FACT_TERMS):
        return True
    has_product_noun = any(term.casefold() in text.casefold() for term in PRODUCT_NOUNS)
    if has_product_noun and any(cue in text for cue in RECOMMENDATION_CUES):
        return True
    return has_product_noun and any(term in text for term in PRODUCT_FACT_TERMS)


def _personal_missing_information(text: str, intents: list[Intent]) -> list[str]:
    """Return missing decision inputs by task type, never a domain Q&A answer."""
    missing: list[str] = []
    personalized = any(cue in text for cue in PERSONAL_CUES) or any(
        cue in text for cue in PERSONAL_DECISION_CUES
    )

    if Intent.CALCULATION in intents:
        if "연금수령한도" in text and not (
            re.search(r"평가액[^\d]*(?:\d|[일이삼사오육칠팔구십백천만억])", text)
            and re.search(r"수령연차[^\d]*\d+", text)
        ):
            missing.append("계좌 평가액과 연금수령연차 정보")
        elif any(term in text for term in ("환급액", "돌려받", "실제 얼마")) and not re.search(
            r"(?:납입|퇴직금|이전)[^\d]*\d", text
        ):
            missing.append("계산 대상 금액과 적용 세율을 정할 소득·재원 정보")
        elif "몇 년 동안" in text and "계좌" not in text:
            missing.append("연금계좌 잔액과 지급기간 계산에 필요한 운용 조건 정보")
        elif personalized and "전환할 수 있는 금액" in text:
            missing.append("ISA 만기 지급액과 이미 전환한 금액 정보")
        elif personalized and not NUMBER.search(text):
            missing.append("계산에 필요한 소득·납입액·수령액 등의 숫자 정보")

    if (
        personalized
        and Intent.ELIGIBILITY in intents
        and not (re.search(r"\d+\s*시간", text) and re.search(r"\d+\s*년", text))
    ):
        missing.append("주당 근로시간과 계속근로기간 정보")

    if personalized and Intent.WITHDRAWAL in intents:
        missing.append("계좌 유형과 중도인출 사유·발생시점·증빙 가능 여부 정보")

    return list(dict.fromkeys(missing))


def _unavailable_capability(text: str) -> str:
    if any(term in text for term in PRIVATE_DATA_TERMS):
        return "이 Agent에는 사용자의 실제 계좌·개인정보 조회 권한이 없습니다."
    if any(term in text for term in LIVE_EXTERNAL_TERMS):
        return "이 Agent는 고정된 raw_markdown 자료 밖의 최신 외부 원문을 조회하지 않습니다."
    if (
        any(term in text for term in ("법", "법령", "세법", "규정"))
        and "미리" in text
        and any(term in text for term in ("확정", "예측", "바뀔", "개정"))
    ):
        return "고정된 raw_markdown 자료만으로 미래 법령 변경을 확정하거나 예측할 수 없습니다."
    return ""


def contrast_query(text: str) -> str:
    """Add generic opposite predicates so false premises retrieve refuting evidence."""
    contrasts = [
        opposite
        for cues, opposite in CONTRAST_TERM_PAIRS
        if any(cue in text for cue in cues)
    ]
    if not contrasts:
        return text
    return SPACE.sub(" ", f"{text} {' '.join(contrasts)}").strip()


def semantic_skeleton_query(plan: QueryPlan) -> str:
    """Compress a question to ontology subjects, relation words, and both polarities."""
    concepts = [link.label for link in plan.ontology_links]
    relations = [term for term in RELATION_QUERY_TERMS if term in plan.normalized_question]
    expanded = contrast_query(plan.normalized_question)
    contrasts = expanded.removeprefix(plan.normalized_question).strip()
    return SPACE.sub(" ", " ".join([*concepts, *relations, contrasts])).strip()


def _is_yes_no_question(text: str) -> bool:
    return bool(
        re.search(
            r"(?:수 있(?:나요|어|습니까)|가능(?:한가요|해|하죠|한지)|"
            r"되나요|돼요|인가요|맞나요|거죠|하죠|있죠)(?:\?|\.|$)",
            text,
        )
    )


def analyze_question(question: str, *, paths: Paths | None = None) -> QueryPlan:
    normalized = SPACE.sub(" ", question).strip()
    paths = paths or Paths()
    links: list[OntologyLink] = []
    for concept in load_query_concepts(
        paths.project_root / "config" / "concepts.yaml",
        paths.project_root / "ontology",
    ):
        matched = _match_alias(normalized, concept)
        if matched:
            links.append(
                OntologyLink(
                    concept_id=concept.concept_id,
                    label=concept.label,
                    ontology_class=concept.ontology_class,
                    matched_alias=matched,
                )
            )

    intents = [
        intent
        for intent, terms in INTENT_TERMS
        if any(_has_intent_term(normalized, term) for term in terms)
    ]
    if _is_definition_question(normalized) and Intent.DEFINITION not in intents:
        intents.append(Intent.DEFINITION)
    # A tax-credit limit question is one closed calculation task, not a request for
    # a second explanation of taxation. Keeping TAX here caused needless subqueries.
    other_tax_terms = ("세금", "세율", "과세", "비과세", "소득세")
    if (
        Intent.CALCULATION in intents
        and Intent.TAX in intents
        and "세액공제" in normalized
        and not any(term in normalized for term in other_tax_terms)
    ):
        intents.remove(Intent.TAX)
    if (
        Intent.COMPARISON in intents
        and Intent.INVESTMENT_RULE in intents
        and all(term in normalized.upper() for term in ("DB", "DC"))
        and not any(term in normalized for term in ("위험자산", "투자 가능", "매수 가능"))
    ):
        intents.remove(Intent.INVESTMENT_RULE)
    if not intents:
        intents = [Intent.GENERAL]

    has_domain_anchor = bool(links) or any(
        term.casefold() in normalized.casefold() for term in DOMAIN_TERMS
    )
    product_only = _product_agent_request(normalized)
    # A finite keyword list can confirm a known pension term, but it cannot prove
    # that an unknown term is outside the corpus. Unknowns therefore go through
    # retrieval and are classified from source evidence. Only an explicit
    # unsupported capability request is rejected before retrieval.
    if any(term in normalized for term in CURRENT_APP_OPERATION_TERMS) or product_only:
        scope_decision = ScopeDecision.OUT_OF_SCOPE
    elif has_domain_anchor:
        scope_decision = ScopeDecision.CONFIRMED
    else:
        scope_decision = ScopeDecision.EVIDENCE_REQUIRED

    missing = _personal_missing_information(normalized, intents)
    personalized_calculation = Intent.CALCULATION in intents and any(
        cue in normalized for cue in PERSONAL_CALCULATION_CUES
    )
    if personalized_calculation and not NUMBER.search(normalized) and not missing:
        missing.append("계산에 필요한 소득·납입액·수령액 등의 숫자 정보")

    canonical = [link.label for link in links]
    subqueries = [normalized]
    if canonical:
        expanded = " ".join(dict.fromkeys([*canonical, *normalized.split()]))
        if expanded != normalized:
            subqueries.append(expanded)
    if len(intents) > 1:
        for intent in intents[:3]:
            subquery = " ".join([*canonical, _intent_hint(intent)]).strip()
            if subquery and subquery not in subqueries:
                subqueries.append(subquery)

    return QueryPlan(
        normalized_question=normalized,
        intents=intents,
        ontology_links=links,
        subqueries=subqueries[:5],
        answer_requirements=_extract_answer_requirements(normalized),
        missing_information=missing,
        unavailable_capability=_unavailable_capability(normalized),
        premise_check_required=(
            any(cue in normalized for cue in PREMISE_CUES)
            or _is_yes_no_question(normalized)
        ),
        scope_decision=scope_decision,
        in_scope=scope_decision != ScopeDecision.OUT_OF_SCOPE,
    )


def retry_query(plan: QueryPlan, attempt: int) -> str:
    concepts = " ".join(link.label for link in plan.ontology_links)
    intents = " ".join(_intent_hint(intent) for intent in plan.intents)
    if attempt <= 1:
        return SPACE.sub(
            " ", f"{plan.normalized_question} {concepts} {intents}"
        ).strip()
    return SPACE.sub(
        " ",
        f"{plan.normalized_question} {concepts} 적용 조건 예외 법령 세율 필요 서류 절차",
    ).strip()


def _intent_hint(intent: Intent) -> str:
    return {
        Intent.DEFINITION: "정의 개념",
        Intent.COMPARISON: "차이 비교",
        Intent.ELIGIBILITY: "가입 대상 자격 조건",
        Intent.PROCEDURE: "신청 절차 필요 서류",
        Intent.TAX: "과세 세율 세액공제 예외",
        Intent.CALCULATION: "계산 한도 공제율",
        Intent.WITHDRAWAL: "중도인출 허용 사유 제한",
        Intent.TRANSFER: "계약이전 조건 절차",
        Intent.CONTRIBUTION: "납입 한도 부담금",
        Intent.INVESTMENT_RULE: "투자 가능 금지 한도",
        Intent.GENERAL: "제도 설명",
    }[intent]
