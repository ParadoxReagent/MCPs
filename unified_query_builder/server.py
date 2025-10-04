from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from unified_query_builder.cbc_schema_loader import CBCSchemaCache, normalise_search_type
from unified_query_builder.cbc_query_builder import (
    build_cbc_query,
    QueryBuildError,
    DEFAULT_BOOLEAN_OPERATOR,
    MAX_LIMIT,
)
from unified_query_builder.kql_schema_loader import SchemaCache
from unified_query_builder.kql_query_builder import (
    build_kql_query,
    suggest_columns,
    example_queries_for_table,
)
from unified_query_builder.rag import (
    UnifiedRAGService,
    SchemaSource,
    build_cbc_documents,
    build_kql_documents,
)
from fastmcp import FastMCP
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


mcp = FastMCP(name="unified-query-builder")
DATA_DIR = Path(".cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CBC_SCHEMA_FILE = Path(__file__).with_name("cb_edr_schema.json")
cbc_cache = CBCSchemaCache(CBC_SCHEMA_FILE)

KQL_SCHEMA_DIR = Path(__file__).with_name("defender_xdr_kql_schema_fuller")
kql_cache = SchemaCache(schema_path=KQL_SCHEMA_DIR / "schema_index.json")


def _cbc_version(cache: CBCSchemaCache) -> Optional[str]:
    try:
        data = cache.load()
    except Exception:  # pragma: no cover - defensive
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return str(version) if version else None


def _kql_version(cache: SchemaCache) -> Optional[int]:
    try:
        return cache.version
    except Exception:  # pragma: no cover - defensive
        return None


def _load_kql_schema(cache: SchemaCache, force: bool = False) -> Dict[str, Any]:
    if force:
        cache.refresh(force=True)
    return cache.load_or_refresh()


rag_service = UnifiedRAGService(
    sources=[
        SchemaSource(
            name="cbc",
            schema_cache=cbc_cache,
            loader=lambda cache, force=False: cache.load(force_refresh=force),
            document_builder=build_cbc_documents,
            version_getter=_cbc_version,
        ),
        SchemaSource(
            name="kql",
            schema_cache=kql_cache,
            loader=_load_kql_schema,
            document_builder=build_kql_documents,
            version_getter=_kql_version,
        ),
    ],
    cache_dir=DATA_DIR,
)


# ---------------------------------------------------------------------------
# Pydantic parameter models
# ---------------------------------------------------------------------------


class CBCFieldsParams(BaseModel):
    search_type: str = Field(
        ..., description="Carbon Black search type (process, binary, alert, threat)"
    )


class CBCExampleQueryParams(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Optional example category: process_search, binary_search, alert_search, etc.",
    )


class CBCBuildQueryParams(BaseModel):
    search_type: Optional[str] = Field(
        default=None, description="Desired search type (defaults to process_search)"
    )
    terms: Optional[List[str]] = Field(
        default=None, description="Pre-built expressions such as field:value pairs"
    )
    natural_language_intent: Optional[str] = Field(
        default=None,
        description="High-level description of what to search for",
    )
    boolean_operator: str = Field(
        default=DEFAULT_BOOLEAN_OPERATOR,
        description="Boolean operator between expressions",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_LIMIT,
        description="Optional record limit hint for downstream consumers",
    )


class KQLListTablesParams(BaseModel):
    keyword: Optional[str] = Field(None, description="Substring filter")


class KQLGetSchemaParams(BaseModel):
    table: str


class KQLSuggestColumnsParams(BaseModel):
    table: str
    keyword: Optional[str] = None


class KQLBuildQueryParams(BaseModel):
    table: Optional[str] = None
    select: Optional[List[str]] = None
    where: Optional[List[str]] = None
    time_window: Optional[str] = None
    summarize: Optional[str] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None
    natural_language_intent: Optional[str] = None


class RetrieveContextParams(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    query_type: Optional[Literal["cbc", "kql"]] = Field(
        default=None,
        description="Optionally restrict retrieval to CBC or KQL schema entries",
    )


# ---------------------------------------------------------------------------
# CBC tools
# ---------------------------------------------------------------------------


@mcp.tool
def cbc_list_search_types() -> Dict[str, Any]:
    """List Carbon Black Cloud search types with their descriptions."""

    schema = cbc_cache.load()
    search_types = schema.get("search_types", {})
    logger.info("Listing %d CBC search types", len(search_types))
    return {"search_types": search_types}


@mcp.tool
def cbc_get_fields(params: CBCFieldsParams) -> Dict[str, Any]:
    """Return available fields for a given search type."""

    schema = cbc_cache.load()
    search_type, log_entries = normalise_search_type(
        params.search_type, schema.get("search_types", {}).keys()
    )
    fields = cbc_cache.list_fields(search_type)
    logger.info(
        "Resolved CBC search type %s (%s) with %d fields",
        params.search_type,
        search_type,
        len(fields),
    )
    return {"search_type": search_type, "fields": fields, "normalisation": log_entries}


@mcp.tool
def cbc_get_operator_reference() -> Dict[str, Any]:
    """Return the logical, wildcard, and field operator reference."""

    operators = cbc_cache.operator_reference()
    logger.info("Returning CBC operator reference with categories: %s", list(operators.keys()))
    return {"operators": operators}


@mcp.tool
def cbc_get_best_practices() -> Dict[str, Any]:
    """Return documented query-building best practices."""

    best = cbc_cache.best_practices()
    logger.info(
        "Returning %s best practice entries", len(best) if isinstance(best, list) else "structured"
    )
    return {"best_practices": best}


@mcp.tool
def cbc_get_example_queries(params: CBCExampleQueryParams) -> Dict[str, Any]:
    """Return example queries, optionally filtered by category."""

    examples = cbc_cache.example_queries()
    if params.category:
        key = params.category
        if key not in examples:
            available = ", ".join(sorted(examples.keys()))
            logger.warning("Unknown CBC example category %s", key)
            return {"error": f"Unknown category '{key}'. Available: {available}"}
        return {"category": key, "examples": examples[key]}
    return {"examples": examples}


@mcp.tool
def cbc_build_query(params: CBCBuildQueryParams) -> Dict[str, Any]:
    """Build a Carbon Black Cloud query from structured parameters or natural language."""

    schema = cbc_cache.load()
    payload = params.model_dump()
    try:
        query, metadata = build_cbc_query(schema, **payload)
        logger.info("Built CBC query for search_type=%s", metadata.get("search_type"))

        intent = payload.get("natural_language_intent")
        if intent:
            try:
                context = rag_service.search(intent, k=5, source_filter="cbc")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Unable to attach CBC RAG context: %s", exc)
            else:
                if context:
                    metadata = {**metadata, "rag_context": context}

        return {"query": query, "metadata": metadata}
    except QueryBuildError as exc:
        logger.warning("Failed to build CBC query: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# KQL tools
# ---------------------------------------------------------------------------


@mcp.tool
def kql_list_tables(params: KQLListTablesParams) -> Dict[str, Any]:
    """List available Advanced Hunting tables (optionally filter by keyword)."""

    schema = kql_cache.load_or_refresh()
    names = list(schema.keys())
    if params.keyword:
        kw = params.keyword.lower()
        names = [n for n in names if kw in n.lower()]
    result = {"tables": sorted(names)}
    logger.info("Found %d KQL tables matching filter", len(names))
    return result


@mcp.tool
def kql_get_table_schema(params: KQLGetSchemaParams) -> Dict[str, Any]:
    """Return columns and docs URL for a given table."""

    schema = kql_cache.load_or_refresh()
    table = params.table
    if table not in schema:
        try:
            from rapidfuzz import process

            choice, score, _ = process.extractOne(table, schema.keys())
            logger.warning(
                "KQL table '%s' not found, suggesting '%s' with score %s",
                table,
                choice,
                score,
            )
            return {"error": f"Unknown table '{table}'. Did you mean '{choice}' (score {score})?"}
        except ImportError:
            logger.error("rapidfuzz not available for fuzzy matching")
            return {"error": f"Unknown table '{table}'"}

    logger.info(
        "Retrieved schema for KQL table '%s' with %d columns",
        table,
        len(schema[table]["columns"]),
    )
    return {"table": table, "columns": schema[table]["columns"], "url": schema[table]["url"]}


@mcp.tool
def kql_suggest_columns(params: KQLSuggestColumnsParams) -> Dict[str, Any]:
    """Suggest columns for a table, optionally filtered by keyword."""

    schema = kql_cache.load_or_refresh()
    suggestions = suggest_columns(schema, params.table, params.keyword)
    logger.info(
        "Found %d KQL column suggestions for table '%s'",
        len(suggestions),
        params.table,
    )
    return {"suggestions": suggestions}


@mcp.tool
def kql_examples(params: KQLGetSchemaParams) -> Dict[str, Any]:
    """Return example KQL for a given table."""

    schema = kql_cache.load_or_refresh()
    examples = example_queries_for_table(schema, params.table)
    logger.info("Generated %d KQL examples for table '%s'", len(examples), params.table)
    return {"examples": examples}


@mcp.tool
def kql_build_query(params: KQLBuildQueryParams) -> Dict[str, Any]:
    """Build a KQL query from structured params or natural-language intent."""

    schema = kql_cache.load_or_refresh()
    payload = params.model_dump()
    try:
        kql, meta = build_kql_query(schema=schema, **payload)

        if payload.get("natural_language_intent"):
            try:
                context = rag_service.search(payload["natural_language_intent"], k=5, source_filter="kql")
                if context:
                    meta = {**meta, "rag_context": context}
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to retrieve KQL RAG context: %s", exc)

        logger.info("Successfully built KQL query for table '%s'", meta.get("table", "unknown"))
        return {"kql": kql, "meta": meta}
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to build KQL query: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@mcp.tool
def retrieve_context(params: RetrieveContextParams) -> Dict[str, Any]:
    """Return relevant schema passages for a natural language query."""

    try:
        results = rag_service.search(params.query, k=params.k, source_filter=params.query_type)
        logger.info(
            "RAG returned %d matches for query with filter=%s",
            len(results),
            params.query_type,
        )
        return {"matches": results}
    except Exception as exc:
        logger.warning("Failed to retrieve RAG context: %s", exc)
        return {"error": str(exc)}


if __name__ == "__main__":
    logger.info("Starting unified query builder MCP server")
    mcp.run()
