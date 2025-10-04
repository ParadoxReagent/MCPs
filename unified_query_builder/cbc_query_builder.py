from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .cbc_schema_loader import normalise_search_type


DEFAULT_SEARCH_TYPE = "process_search"
DEFAULT_BOOLEAN_OPERATOR = "AND"
SUPPORTED_BOOLEAN_OPERATORS = {"AND", "OR"}
MAX_LIMIT = 5000

_MD5_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")
_SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\b")
_PORT_RE = re.compile(r"\bport\s*(?:=|is)?\s*(\d{1,5})\b", re.IGNORECASE)
_QUOTED_VALUE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')

_PROCESS_NAME_RE = re.compile(
    r"(?:process|binary)\s+name(?:\s+(?:is|=|equals|was))?\s*[\"']?([A-Za-z0-9_.-]+)[\"']?",
    re.IGNORECASE,
)
_CMDLINE_RE = re.compile(
    r"(?:cmdline|command\s+line)\s+(?:contains|includes|with)?\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?:path|file\s+path)\s+(?:is|=|equals)?\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_USERNAME_RE = re.compile(
    r"user(?:name)?\s+(?:is|=|equals|running\s+as)?\s*[\"']?([A-Za-z0-9_@.-]+)[\"']?",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"domain\s+(?:is|=|equals|contains)?\s*[\"']?([A-Za-z0-9_.-]+)[\"']?",
    re.IGNORECASE,
)

_STOPWORDS = {
    "find",
    "show",
    "me",
    "all",
    "process",
    "processes",
    "with",
    "that",
    "where",
    "which",
    "the",
    "a",
    "an",
    "for",
    "running",
    "binary",
    "alerts",
    "alert",
    "search",
}


class QueryBuildError(ValueError):
    """Raised when we cannot construct a valid query."""


def _quote_if_needed(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    cleaned = cleaned.replace("\"", r"\"")
    if any(ch.isspace() for ch in cleaned) or ":" in cleaned:
        return f'"{cleaned}"'
    return cleaned


def _sanitise_term(term: str) -> str:
    cleaned = term.strip()
    if not cleaned:
        return ""
    if any(ch in cleaned for ch in [";", "|", "(", ")", "{" ,"}"]):
        raise QueryBuildError(f"Unsafe characters detected in term '{term}'")
    # Escape backslashes for Windows paths and domain-qualified usernames
    cleaned = cleaned.replace("\\", "\\\\")
    return cleaned


def _field_if_available(candidates: Sequence[str], available_fields: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in available_fields:
            return candidate
    return None


def _collect_fields(field_map: Dict[str, Dict[str, Any]]) -> List[str]:
    return list(field_map.keys())


def _extract_patterns(intent: str, field_map: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[Tuple[int, int]], List[Dict[str, Any]]]:
    expressions: List[str] = []
    spans: List[Tuple[int, int]] = []
    metadata: List[Dict[str, Any]] = []
    available_fields = _collect_fields(field_map)

    pattern_definitions = [
        ("md5", _MD5_RE, ["process_md5", "md5"]),
        ("sha256", _SHA256_RE, ["process_sha256", "sha256"]),
        ("ipv4", _IPV4_RE, ["ipaddr", "remote_ip", "sensor_ip"]),
        ("ipv6", _IPV6_RE, ["ipv6addr", "remote_ipv6"]),
    ]

    for label, regex, candidates in pattern_definitions:
        for match in regex.finditer(intent):
            field = _field_if_available(candidates, available_fields)
            if not field:
                continue
            value = match.group(0)
            expressions.append(f"{field}:{_sanitise_term(value)}")
            spans.append(match.span())
            metadata.append({"type": label, "field": field, "value": value})

    # Explicit constructs
    explicit_patterns = [
        ("process_name", _PROCESS_NAME_RE, ["process_name", "observed_filename", "binary", "parent_name"]),
        ("cmdline", _CMDLINE_RE, ["cmdline"]),
        ("path", _PATH_RE, ["path", "observed_filename"]),
        ("username", _USERNAME_RE, ["username", "user", "logon_user"]),
        ("domain", _DOMAIN_RE, ["domain", "hostname"]),
    ]

    for label, regex, candidates in explicit_patterns:
        for match in regex.finditer(intent):
            field = _field_if_available(candidates, available_fields)
            if not field:
                continue
            value = match.group(1)
            if not value:
                continue
            formatted = _quote_if_needed(_sanitise_term(value))
            expressions.append(f"{field}:{formatted}")
            spans.append(match.span())
            metadata.append({"type": label, "field": field, "value": value})

    # Ports
    for match in _PORT_RE.finditer(intent):
        field = _field_if_available(["ipport", "port"], available_fields)
        if not field:
            continue
        value = match.group(1)
        expressions.append(f"{field}:{value}")
        spans.append(match.span())
        metadata.append({"type": "port", "field": field, "value": value})

    return expressions, spans, metadata


def _residual_terms(intent: str, spans: List[Tuple[int, int]]) -> List[str]:
    if not intent:
        return []

    chars = list(intent)
    for start, end in spans:
        for idx in range(start, min(end, len(chars))):
            chars[idx] = " "

    residual = re.sub(r"\s+", " ", "".join(chars)).strip()
    if not residual:
        return []

    terms: List[str] = []
    for token in re.split(r"[;,]", residual):
        token = token.strip()
        if not token:
            continue
        # Remove quoted substrings to avoid duplication
        token = _QUOTED_VALUE_RE.sub(lambda m: m.group(1) or m.group(2) or "", token)
        words = [w for w in re.split(r"[^A-Za-z0-9_.-]+", token) if w]
        filtered = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
        for word in filtered:
            terms.append(word)
    return terms


def _compose_query(expressions: List[str], boolean_operator: str) -> str:
    if not expressions:
        raise QueryBuildError("No search terms could be derived from the provided input")
    return f" {boolean_operator} ".join(expressions)


def build_cbc_query(
    schema: Dict[str, Any],
    *,
    search_type: str | None = None,
    terms: Sequence[str] | None = None,
    natural_language_intent: str | None = None,
    boolean_operator: str = DEFAULT_BOOLEAN_OPERATOR,
    limit: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build a Carbon Black Cloud EDR query string and return metadata."""

    search_types = schema.get("search_types", {})
    chosen_search_type, normalisation_log = normalise_search_type(
        search_type or DEFAULT_SEARCH_TYPE, search_types.keys()
    )

    field_map = {}
    if hasattr(schema, "field_map_for"):
        # Support callers who pass CBCSchemaCache.load()
        field_map = schema.field_map_for(chosen_search_type)  # type: ignore[attr-defined]
    else:
        # schema may already be the payload
        mapping_key = {
            "process_search": "process_search_fields",
            "binary_search": "binary_search_fields",
            "alert_search": "alert_search_fields",
            "threat_report_search": "threat_report_search_fields",
        }.get(chosen_search_type)
        if mapping_key:
            raw_fields = schema.get(mapping_key, {})
            if isinstance(raw_fields, dict):
                field_map = raw_fields

    expressions: List[str] = []
    recognised: List[Dict[str, Any]] = []

    if terms:
        for term in terms:
            cleaned = _sanitise_term(term)
            if not cleaned:
                continue
            expressions.append(cleaned)
            recognised.append({"type": "structured", "value": cleaned})

    if natural_language_intent:
        nl_expressions, spans, meta = _extract_patterns(natural_language_intent, field_map)
        expressions.extend(nl_expressions)
        recognised.extend(meta)

        for token in _residual_terms(natural_language_intent, spans):
            sanitised = _sanitise_term(token)
            if not sanitised:
                continue
            expressions.append(sanitised)
            recognised.append({"type": "keyword", "value": sanitised})

    if not expressions:
        raise QueryBuildError("No expressions provided. Supply terms or natural language intent.")

    operator = boolean_operator.upper().strip()
    if operator not in SUPPORTED_BOOLEAN_OPERATORS:
        raise QueryBuildError(
            f"Unsupported boolean operator '{boolean_operator}'. Use one of: {', '.join(SUPPORTED_BOOLEAN_OPERATORS)}"
        )

    # Clamp limit if provided
    limit_value: int | None = None
    if limit is not None:
        if limit <= 0:
            raise QueryBuildError("Limit must be positive")
        limit_value = min(limit, MAX_LIMIT)

    query = _compose_query(expressions, operator)

    metadata = {
        "search_type": chosen_search_type,
        "normalisation": normalisation_log,
        "boolean_operator": operator,
        "recognised": recognised,
    }

    if limit_value is not None:
        metadata["limit"] = limit_value
        if limit_value != limit:
            metadata["limit_clamped"] = MAX_LIMIT

    return query, metadata
