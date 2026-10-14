from __future__ import annotations

import hashlib
import logging
import math
import os
from functools import lru_cache
from threading import Lock
from typing import Annotated, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from pension_graph.agent import PensionSystemAgent
from pension_graph.agent_models import AgentResult, AnswerStatus
from pension_graph.deadline import (
    RequestDeadlineExceeded,
    check_deadline,
    request_deadline,
)
from pension_graph.hybrid_retrieval import retrieve_hybrid_context
from pension_graph.vector_store import VectorSettings, make_client

logger = logging.getLogger("uvicorn.error")
_inflight_lock = Lock()
_inflight_requests: set[str] = set()

app = FastAPI(
    title="Pension Agent API",
    version="0.2.0",
    description="근거 중심 연금 Agent 및 제출용 GET /answer API",
)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=30)


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, str] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str


class AnswerAgent(Protocol):
    def answer(self, *, question_id: str, question: str) -> AgentResult: ...


@lru_cache(maxsize=1)
def get_pension_agent() -> AnswerAgent:
    mode = os.getenv("SUBMISSION_MODE", "system").strip().lower()
    if mode == "integrated":
        from pension_graph.submission import IntegratedPensionAgent

        return IntegratedPensionAgent()
    if mode != "system":
        raise ValueError("Invalid submission mode")
    return PensionSystemAgent()


def _error_response(
    *, question_id: str, question: str, code: str, message: str, status_code: int = 503,
    retrieved_context: str = "",
) -> JSONResponse:
    body = AnswerResponse(
        question_id=question_id,
        question=question,
        retrieved_context=retrieved_context,
        think_trace=f"status=SYSTEM_ERROR; error={code}",
        answer=message,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path == "/answer":
        return _error_response(
            question_id=request.query_params.get("question_id", ""),
            question=request.query_params.get("question", ""),
            code="INVALID_REQUEST",
            message="question_id와 question을 허용된 길이의 문자열로 전달해 주세요.",
            status_code=422,
        )
    # Use FastAPI's standard validation handler on non-submission endpoints.
    from fastapi.exception_handlers import request_validation_exception_handler

    return await request_validation_exception_handler(request, exc)


@app.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health/ready", response_model=HealthResponse)
def ready(response: Response) -> HealthResponse:
    dependencies: dict[str, str] = {}
    try:
        settings = VectorSettings.from_env()
    except (KeyError, ValueError, TypeError):
        response.status_code = 503
        return HealthResponse(status="not_ready", dependencies={"configuration": "invalid"})
    try:
        client = make_client(settings)
        dependencies["qdrant"] = (
            "ok" if client.collection_exists(settings.collection) else "collection_missing"
        )
    except (OSError, ValueError, ResponseHandlingException, UnexpectedResponse) as exc:
        dependencies["qdrant"] = f"error:{type(exc).__name__}"
    try:
        embedding_response = httpx.get(f"{settings.dense_url.rstrip('/')}/health", timeout=3)
        dependencies["dense_embedding"] = "ok" if embedding_response.is_success else "not_ready"
    except httpx.HTTPError as exc:
        dependencies["dense_embedding"] = f"error:{type(exc).__name__}"
    try:
        with GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7688"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.environ["NEO4J_PASSWORD"],
            ),
        ) as driver:
            driver.verify_connectivity()
        dependencies["neo4j"] = "ok"
    except (KeyError, OSError, Neo4jError, ServiceUnavailable) as exc:
        dependencies["neo4j"] = f"error:{type(exc).__name__}"
    dependencies["hcx"] = (
        "configured" if os.getenv("CLOVA_STUDIO_API_KEY", "").strip() else "not_configured"
    )
    integrated = os.getenv("SUBMISSION_MODE", "system").strip().lower() == "integrated"
    if integrated:
        product_url = os.getenv("PRODUCT_AGENT_URL", "http://localhost:8090").rstrip("/")
        try:
            product_response = httpx.get(f"{product_url}/health", timeout=3)
            product_response.raise_for_status()
            product_health = product_response.json()
            dependencies["product_agent"] = (
                "ok" if product_health.get("ready") is True else "not_ready"
            )
            dense_complete = product_health.get("dense_complete")
            dependencies["product_dense"] = (
                "complete" if dense_complete is True
                else "partial" if dense_complete is False else "unknown"
            )
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            dependencies["product_agent"] = f"error:{type(exc).__name__}"
            dependencies["product_dense"] = "unknown"
    status = (
        "ok"
        if dependencies.get("qdrant") == "ok"
        and dependencies.get("dense_embedding") == "ok"
        and dependencies.get("neo4j") == "ok"
        and dependencies.get("hcx") == "configured"
        and (not integrated or dependencies.get("product_agent") == "ok")
        else "not_ready"
    )
    if status != "ok":
        response.status_code = 503
    return HealthResponse(status=status, dependencies=dependencies)


@app.post("/v1/retrieve")
def retrieve(request: RetrieveRequest) -> dict[str, object]:
    try:
        return retrieve_hybrid_context(
            query=request.query,
            qdrant_settings=VectorSettings.from_env(),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7688"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ["NEO4J_PASSWORD"],
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            limit=request.limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=503, detail="NEO4J_PASSWORD가 없습니다.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"검색 의존성 오류: {type(exc).__name__}") from exc


@app.get("/v1/retrieve")
def retrieve_get(
    q: Annotated[str, Query(min_length=1, max_length=1000)],
    limit: Annotated[int, Query(ge=1, le=30)] = 10,
) -> dict[str, object]:
    return retrieve(RetrieveRequest(query=q, limit=limit))


@app.get(
    "/answer", response_model=AnswerResponse,
    responses={503: {"model": AnswerResponse}, 422: {"model": AnswerResponse}},
)
def answer(
    question_id: Annotated[str, Query(min_length=1, max_length=200)],
    question: Annotated[str, Query(min_length=1, max_length=1000)],
) -> AnswerResponse | JSONResponse:
    """과제 명세의 GET /answer 계약을 제공하고 숨겨진 사고과정은 노출하지 않습니다."""
    request_key = hashlib.sha256(f"{question_id}\0{question}".encode()).hexdigest()
    # A caller-side timeout must not start the same request again while a previous
    # non-cooperative call is still running. This admission guard is per process.
    with _inflight_lock:
        if request_key in _inflight_requests:
            return _error_response(
                question_id=question_id, question=question, code="REQUEST_IN_PROGRESS",
                message="동일한 요청을 처리 중입니다. 잠시 후 다시 시도해 주세요.",
            )
        _inflight_requests.add(request_key)
    try:
        seconds = float(os.getenv("PENSION_REQUEST_TIMEOUT_SECONDS", "270"))
        if not math.isfinite(seconds) or not 0 < seconds <= 270:
            raise ValueError("Request budget must be between zero and 270 seconds")
        # Deliberately execute inline: future.result(timeout=...) leaves an active
        # thread behind and would permit retry work to accumulate invisibly.
        with request_deadline(seconds):
            result = get_pension_agent().answer(question_id=question_id, question=question)
            check_deadline()
            if result.answer_status == AnswerStatus.SYSTEM_ERROR:
                return _error_response(
                    question_id=question_id, question=question,
                    code=result.error_code if result.error_code != "NONE" else "AGENT_UNAVAILABLE",
                    message="현재 답변 처리에 일시적인 문제가 있습니다. 다시 시도해 주세요.",
                    retrieved_context=result.retrieved_context,
                )
            body = AnswerResponse(
                question_id=question_id,
                question=question,
                retrieved_context=result.retrieved_context,
                think_trace=(
                    f"status={result.answer_status.value}; retries={result.retry_count}; "
                    f"retrieval_retries={result.retrieval_retry_count}; "
                    f"generation_retries={result.generation_retry_count}; "
                    f"{result.think_trace}"
                ),
                answer=result.answer,
            )
            check_deadline()
            return body
    except RequestDeadlineExceeded:
        return _error_response(
            question_id=question_id, question=question, code="DEADLINE_EXCEEDED",
            message="요청 처리 시간 예산을 초과했습니다. 다시 시도해 주세요.",
        )
    except Exception as exc:  # noqa: BLE001 - public boundary must preserve the error contract
        # No exception text, headers, provider bodies, locals or full traceback.
        # Log only internal code locations to distinguish otherwise opaque failures.
        from pension_graph.hcx_metrics import exception_locations

        logger.warning("Agent request failed type=%s locations=%s",
                       type(exc).__name__, exception_locations(exc))
        return _error_response(
            question_id=question_id, question=question, code="DEPENDENCY_FAILURE",
            message="현재 답변 처리에 일시적인 문제가 있습니다. 다시 시도해 주세요.",
        )
    finally:
        with _inflight_lock:
            _inflight_requests.discard(request_key)
