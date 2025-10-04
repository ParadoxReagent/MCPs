from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from cbc_builder.schema_loader import CBCSchemaCache, normalise_search_type
from cbc_builder.query_builder import (
    build_cbc_query,
    QueryBuildError,
    DEFAULT_BOOLEAN_OPERATOR,
    MAX_LIMIT,
)
from cbc_builder.rag import RAGService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SCHEMA_FILE = Path(__file__).with_name("cb_edr_schema.json")
cache = CBCSchemaCache(SCHEMA_FILE)
DATA_DIR = Path(".cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)
rag_service = RAGService(schema_cache=cache, cache_dir=DATA_DIR)
mcp = FastMCP(name="cbc-query-builder")


class FieldsParams(BaseModel):
    search_type: str = Field(..., description="Carbon Black search type (process, binary, alert, threat)")


class ExampleQueryParams(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Optional example category: process_search, binary_search, alert_search, etc.",
    )


class BuildQueryParams(BaseModel):
    search_type: Optional[str] = Field(default=None, description="Desired search type (defaults to process_search)")
    terms: Optional[List[str]] = Field(default=None, description="Pre-built expressions such as field:value pairs")
    natural_language_intent: Optional[str] = Field(
        default=None,
        description="High-level description of what to search for",
    )
    boolean_operator: str = Field(default=DEFAULT_BOOLEAN_OPERATOR, description="Boolean operator between expressions")
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_LIMIT,
        description="Optional record limit hint for downstream consumers",
    )


class RetrieveContextParams(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)


@mcp.tool
def list_search_types() -> Dict[str, Any]:
    """List Carbon Black Cloud search types with their descriptions."""

    schema = cache.load()
    search_types = schema.get("search_types", {})
    logger.info("Listing %d search types", len(search_types))
    return {"search_types": search_types}


@mcp.tool
def get_fields(params: FieldsParams) -> Dict[str, Any]:
    """Return available fields for a given search type."""

    schema = cache.load()
    search_type, log_entries = normalise_search_type(params.search_type, schema.get("search_types", {}).keys())
    fields = cache.list_fields(search_type)
    logger.info("Resolved search type %s (%s) with %d fields", params.search_type, search_type, len(fields))
    return {"search_type": search_type, "fields": fields, "normalisation": log_entries}


@mcp.tool
def get_operator_reference() -> Dict[str, Any]:
    """Return the logical, wildcard, and field operator reference."""

    operators = cache.operator_reference()
    logger.info("Returning operator reference with categories: %s", list(operators.keys()))
    return {"operators": operators}


@mcp.tool
def get_best_practices() -> Dict[str, Any]:
    """Return documented query-building best practices."""

    best = cache.best_practices()
    logger.info("Returning %s best practice entries", len(best) if isinstance(best, list) else "structured")
    return {"best_practices": best}


@mcp.tool
def get_example_queries(params: ExampleQueryParams) -> Dict[str, Any]:
    """Return example queries, optionally filtered by category."""

    examples = cache.example_queries()
    if params.category:
        key = params.category
        if key not in examples:
            available = ", ".join(sorted(examples.keys()))
            logger.warning("Unknown example category %s", key)
            return {"error": f"Unknown category '{key}'. Available: {available}"}
        return {"category": key, "examples": examples[key]}
    return {"examples": examples}


@mcp.tool
def retrieve_context(params: RetrieveContextParams) -> Dict[str, Any]:
    """Return relevant schema passages for a natural language query."""

    try:
        results = rag_service.search(params.query, k=params.k)
        logger.info("RAG returned %d matches for query", len(results))
        return {"matches": results}
    except Exception as exc:
        logger.warning("Failed to retrieve RAG context: %s", exc)
        return {"error": str(exc)}


@mcp.tool
def build_query(params: BuildQueryParams) -> Dict[str, Any]:
    """Build a Carbon Black Cloud query from structured parameters or natural language."""

    schema = cache.load()
    payload = params.model_dump()
    try:
        query, metadata = build_cbc_query(schema, **payload)
        logger.info("Built CBC query for search_type=%s", metadata.get("search_type"))

        intent = payload.get("natural_language_intent")
        if intent:
            try:
                context = rag_service.search(intent, k=5)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Unable to attach RAG context: %s", exc)
            else:
                if context:
                    metadata = {**metadata, "rag_context": context}

        return {"query": query, "metadata": metadata}
    except QueryBuildError as exc:
        logger.warning("Failed to build query: %s", exc)
        return {"error": str(exc)}


if __name__ == "__main__":
    logger.info("Starting CBC query builder MCP server")
    mcp.run()
