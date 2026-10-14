"""Allowlisted HCX usage metrics: never copy prompts, completions or thinking text."""
from __future__ import annotations

from pathlib import Path
from traceback import walk_tb
from typing import Any


def exception_locations(exc: Exception) -> list[dict[str, int | str]]:
    """Code locations only; never exception values, locals, source lines or paths."""
    return [
        {"file": Path(frame.f_code.co_filename).name,
         "function": frame.f_code.co_name, "line": line}
        for frame, line in walk_tb(exc.__traceback__)
        if "pension_graph" in Path(frame.f_code.co_filename).parts
    ][-4:]


def verification_error_category(errors: list[str]) -> str:
    """Map engine-owned error prefixes to fixed labels, never log arbitrary text."""
    categories = {
        "HcxOutputTruncated": "OUTPUT_TRUNCATED", "HcxJsonConflict": "JSON_CONFLICT",
        "JSONDecodeError": "INVALID_JSON", "ValidationError": "SCHEMA_INVALID",
        "ReviewProtocolError": "REVIEW_CONTRACT", "HTTPStatusError": "PROVIDER_HTTP",
        "ReadTimeout": "PROVIDER_TIMEOUT", "ConnectTimeout": "PROVIDER_TIMEOUT",
        "RequestDeadlineExceeded": "DEADLINE_EXCEEDED",
    }
    prefix = "의미 검증 서비스/응답 오류: "
    for error in errors:
        if error.startswith(prefix):
            detail = error[len(prefix):]
            for name, category in categories.items():
                if detail == name or detail.startswith((name + ":", name + "(")):
                    return category
    return "UNKNOWN_VERIFICATION_ERROR"


def completion_budget(max_tokens: int, thinking_effort: str, override: str | None = None) -> int:
    """Thinking and final JSON share one budget; apply documented effort defaults.

    An explicit experiment override affects thinking calls only, never generation
    with effort=none. This is a token cap, not a latency or success guarantee.
    """
    defaults = {"low": 5120, "medium": 10240, "high": 20480}
    if thinking_effort not in {"none", *defaults}:
        raise ValueError("Invalid thinking effort")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ValueError("Invalid completion token budget")
    if thinking_effort == "none":
        return max_tokens
    budget = int(override) if override is not None else max(defaults[thinking_effort], max_tokens)
    if not 1 <= budget <= 32768:
        raise ValueError("Invalid thinking token budget")
    return budget


def safe_usage_metrics(payload: Any) -> dict[str, int | str]:
    """Read the documented v3 result.usage shape; unknown metadata is discarded."""
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        return {}
    result = payload["result"]
    metrics: dict[str, int | str] = {}
    usage = result.get("usage")
    if isinstance(usage, dict):
        for source, output in (
            ("promptTokens", "prompt_tokens"),
            ("completionTokens", "completion_tokens"),
            ("totalTokens", "total_tokens"),
        ):
            value = usage.get(source)
            if type(value) is int and value >= 0:
                metrics[output] = value
        details = usage.get("completionTokensDetails")
        if isinstance(details, dict):
            value = details.get("thinkingTokens")
            if type(value) is int and value >= 0:
                metrics["thinking_tokens"] = value
    reason = result.get("finishReason") or result.get("stopReason")
    if isinstance(reason, str) and reason in {"length", "stop", "end_token", "stop_before"}:
        metrics["finish_reason"] = reason
    return metrics


def safe_quota_metrics(headers: Any) -> dict[str, int]:
    """Only numeric quota capacities/remaining counts; no arbitrary HTTP headers."""
    fields = {
        "x-ratelimit-limit-requests": "request_limit",
        "x-ratelimit-limit-tokens": "token_limit",
        "x-ratelimit-remaining-requests": "requests_remaining",
        "x-ratelimit-remaining-tokens": "tokens_remaining",
    }
    values = {str(key).lower(): str(value).strip() for key, value in headers.items()}
    return {output: int(values[name]) for name, output in fields.items()
            if name in values and values[name].isascii() and values[name].isdigit()
            and len(values[name]) <= 15}
