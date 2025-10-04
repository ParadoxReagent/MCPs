"""Unified retrieval service shared across CBC and KQL schemas."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled at runtime
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - handled at runtime
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    from rapidfuzz import process as rapidfuzz_process
except ImportError:  # pragma: no cover - handled at runtime
    rapidfuzz_process = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


LoaderFn = Callable[[Any, bool], Dict[str, Any]]
VersionFn = Callable[[Any], Optional[str | int]]
DocumentBuilder = Callable[[Dict[str, Any]], Sequence[Dict[str, Any]]]


@dataclass
class SchemaSource:
    """Description of a schema source the RAG service should index."""

    name: str
    schema_cache: Any
    loader: LoaderFn
    document_builder: DocumentBuilder
    version_getter: Optional[VersionFn] = None

    def load_schema(self, force: bool = False) -> Dict[str, Any]:
        return self.loader(self.schema_cache, force)

    def version(self) -> Optional[str | int]:
        if self.version_getter is None:
            return None
        return self.version_getter(self.schema_cache)


def _ensure_sentence_transformers() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for retrieval. Install the optional dependencies.",
        ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR
    return SentenceTransformer  # type: ignore[return-value]


@dataclass
class UnifiedRAGService:
    """Build and reuse embeddings for multiple schema sources."""

    sources: Sequence[SchemaSource]
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[Any] = None
        self._documents: List[Dict[str, Any]] = []
        self._dimension: Optional[int] = None
        self._metadata_path = self.cache_dir / "rag_metadata.json"
        self._index_path = self.cache_dir / "rag_index.faiss"
        self._mode: str = "uninitialized"
        self._source_versions: Dict[str, Optional[str | int]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            SentenceModel = _ensure_sentence_transformers()
            logger.info("Loading sentence transformer model '%s'", self.model_name)
            self._model = SentenceModel(self.model_name)
        return self._model

    def _documents_signature(self, documents: Sequence[Dict[str, Any]]) -> str:
        payload = json.dumps(
            [
                {
                    "id": doc.get("id"),
                    "source": doc.get("source"),
                    "text": doc.get("text"),
                }
                for doc in documents
            ],
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_cached_index(self, signature: str) -> bool:
        if not self._metadata_path.exists() or not self._index_path.exists():
            return False

        try:
            with self._metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read RAG metadata cache: %s", exc)
            return False

        if metadata.get("signature") != signature:
            logger.info("Cached embeddings are out of date; rebuilding.")
            return False

        mode = metadata.get("mode", "faiss")
        self._mode = mode
        self._documents = metadata.get("documents", [])
        self._source_versions = metadata.get("source_versions", {})

        if mode == "fuzzy":
            logger.info(
                "Loaded fallback retrieval metadata for %d documents.",
                len(self._documents),
            )
            self._index = None
            self._dimension = None
            return True

        if faiss is None:
            return False

        try:
            self._index = faiss.read_index(str(self._index_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read FAISS index: %s", exc)
            self._index = None
            return False

        self._dimension = metadata.get("dimension")
        if self._dimension is None:
            logger.warning("Cached metadata missing embedding dimension; rebuilding.")
            self._index = None
            self._documents = []
            return False

        logger.info("Loaded cached embeddings for %d documents.", len(self._documents))
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_index(self, force: bool = False) -> None:
        """Ensure the embedding index is built for all registered sources."""

        documents: List[Dict[str, Any]] = []
        versions: Dict[str, Optional[str | int]] = {}

        for source in self.sources:
            schema = source.load_schema(force=force)
            versions[source.name] = source.version()
            for raw_doc in source.document_builder(schema):
                if "text" not in raw_doc:
                    continue
                doc = dict(raw_doc)
                doc.setdefault("id", f"{source.name}:{len(documents)}")
                doc["source"] = source.name
                documents.append(doc)

        signature = self._documents_signature(documents)

        if (
            not force
            and self._mode != "uninitialized"
            and documents
            and self._source_versions == versions
            and self._load_cached_index(signature)
        ):
            self._documents = documents
            self._source_versions = versions
            return

        if not documents:
            logger.warning("No documents available for RAG indexing.")
            self._documents = []
            self._index = None
            self._dimension = None
            self._mode = "fuzzy"
            self._source_versions = versions
            return

        if SentenceTransformer is None or faiss is None or np is None:
            if rapidfuzz_process is None:
                missing = []
                if SentenceTransformer is None:
                    missing.append("sentence-transformers")
                if faiss is None:
                    missing.append("faiss-cpu")
                if np is None:
                    missing.append("numpy")
                raise RuntimeError(
                    "Retrieval dependencies are unavailable: " + ", ".join(missing)
                )
            logger.warning(
                "Embedding libraries unavailable; using rapidfuzz-based fallback retrieval.",
            )
            self._documents = documents
            self._dimension = None
            self._index = None
            self._mode = "fuzzy"
            self._source_versions = versions
            with self._metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "signature": signature,
                        "mode": "fuzzy",
                        "documents": documents,
                        "source_versions": versions,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            if self._index_path.exists():
                self._index_path.unlink()
            return

        self._mode = "faiss"
        model = self._load_model()
        logger.info("Generating embeddings for %d schema documents", len(documents))
        embeddings = model.encode(
            [doc["text"] for doc in documents],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Model returned embeddings with unexpected shape")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        faiss.write_index(index, str(self._index_path))
        with self._metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "signature": signature,
                    "mode": "faiss",
                    "dimension": dimension,
                    "documents": documents,
                    "source_versions": versions,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self._index = index
        self._documents = documents
        self._dimension = dimension
        self._source_versions = versions
        logger.info("Persisted new embeddings cache to %s", self.cache_dir)

    def search(self, query: str, k: int = 5, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the top-k documents matching the query.

        Parameters
        ----------
        query:
            Free-form natural language query.
        k:
            Number of results to return (defaults to 5, capped at available documents).
        source_filter:
            Optional name of a registered schema source to filter results (e.g. "cbc" or "kql").
        """

        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")

        self.ensure_index()

        candidates = [doc for doc in self._documents if not source_filter or doc.get("source") == source_filter]
        if not candidates:
            return []

        top_k = min(k, len(candidates))
        if top_k == 0:
            return []

        if self._mode == "fuzzy":
            if rapidfuzz_process is None:
                raise RuntimeError("rapidfuzz is required for fallback retrieval")
            matches = rapidfuzz_process.extract(
                query,
                [doc["text"] for doc in candidates],
                limit=top_k,
            )
            results: List[Dict[str, Any]] = []
            for _, score, idx in matches:
                doc = candidates[idx]
                results.append(
                    {
                        "source": doc.get("source"),
                        "id": doc.get("id"),
                        "text": doc.get("text"),
                        "metadata": doc.get("metadata", {}),
                        "score": float(score),
                    }
                )
            return results

        if self._index is None or self._dimension is None:
            raise RuntimeError("Embedding index is not initialized")

        model = self._load_model()
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Build matrix of candidate embeddings in the same order as candidates
        # by querying FAISS for more results than strictly needed, then filter.
        scores, indices = self._index.search(query_embedding, len(self._documents))
        ordered: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            if source_filter and doc.get("source") != source_filter:
                continue
            ordered.append(
                {
                    "source": doc.get("source"),
                    "id": doc.get("id"),
                    "text": doc.get("text"),
                    "metadata": doc.get("metadata", {}),
                    "score": float(score),
                }
            )
            if len(ordered) >= top_k:
                break
        return ordered

    def clear_cache(self) -> None:
        """Remove cached embeddings."""

        if self._metadata_path.exists():
            self._metadata_path.unlink()
        if self._index_path.exists():
            self._index_path.unlink()
        self._index = None
        self._documents = []
        self._dimension = None
        self._mode = "uninitialized"
        self._source_versions = {}


def build_cbc_documents(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert the CBC schema JSON into retrieval-friendly documents."""

    documents: List[Dict[str, Any]] = []

    search_types = schema.get("search_types", {})
    field_mapping = {
        "process_search": "process_search_fields",
        "binary_search": "binary_search_fields",
        "alert_search": "alert_search_fields",
        "threat_report_search": "threat_report_search_fields",
    }

    if isinstance(search_types, dict):
        for name in sorted(search_types.keys()):
            meta = search_types.get(name, {})
            description = str(meta.get("description", "")) if isinstance(meta, dict) else ""
            applicable = []
            if isinstance(meta, dict):
                raw_applicable = meta.get("applicable_to")
                if isinstance(raw_applicable, list):
                    applicable = [str(item) for item in raw_applicable]

            field_key = field_mapping.get(name, "")
            raw_fields = schema.get(field_key, {})
            field_lines: List[str] = []
            if isinstance(raw_fields, dict):
                for field_name in sorted(raw_fields.keys()):
                    field_meta = raw_fields.get(field_name)
                    if not isinstance(field_meta, dict):
                        continue
                    field_type = str(field_meta.get("type", ""))
                    description_line = str(field_meta.get("description", ""))
                    values = field_meta.get("values")
                    qualifiers: List[str] = []
                    if field_type:
                        qualifiers.append(field_type)
                    if field_meta.get("default_field"):
                        qualifiers.append("default")
                    header = field_name
                    if qualifiers:
                        header += f" ({', '.join(qualifiers)})"
                    parts = [header]
                    if description_line:
                        parts.append(f"- {description_line}")
                    if isinstance(values, list) and values:
                        preview = ", ".join(str(v) for v in values[:5])
                        if len(values) > 5:
                            preview += ", ..."
                        parts.append(f"Values: {preview}")
                    field_lines.append(" ".join(parts).strip())
            if not field_lines:
                field_lines.append("No field metadata available.")

            applies_to = ", ".join(applicable) if applicable else "General"
            text = "\n".join(
                [
                    f"Search Type: {name}",
                    f"Description: {description or 'Not documented.'}",
                    f"Applies To: {applies_to}",
                    "Fields:",
                    *field_lines,
                ]
            )

            documents.append(
                {
                    "id": f"cbc:search_type:{name}",
                    "text": text,
                    "metadata": {
                        "section": "search_types",
                        "search_type": name,
                        "description": description,
                        "applicable_to": applicable,
                    },
                }
            )

    field_types = schema.get("field_types")
    if isinstance(field_types, dict) and field_types:
        lines: List[str] = []
        for field_type, meta in sorted(field_types.items()):
            if not isinstance(meta, dict):
                continue
            description = str(meta.get("description", ""))
            behavior = str(meta.get("search_behavior", ""))
            example = meta.get("example")
            parts = [f"Type: {field_type}"]
            if description:
                parts.append(f"Description: {description}")
            if behavior:
                parts.append(f"Search behaviour: {behavior}")
            if example:
                parts.append(f"Example: {example}")
            lines.append(" | ".join(parts))
        documents.append(
            {
                "id": "cbc:field_types",
                "text": "\n".join(["Field Type Reference:", *lines]) if lines else "Field Type Reference:",
                "metadata": {"section": "field_types"},
            }
        )

    operators = schema.get("operators")
    if isinstance(operators, dict) and operators:
        lines = ["Operator Reference:"]
        for category, entries in sorted(operators.items()):
            lines.append(f"Category: {category}")
            if isinstance(entries, dict):
                for name, meta in sorted(entries.items()):
                    if not isinstance(meta, dict):
                        continue
                    description = str(meta.get("description", ""))
                    syntax = meta.get("syntax")
                    examples = meta.get("examples")
                    line_parts = [f"- {name}"]
                    if description:
                        line_parts.append(description)
                    if isinstance(syntax, list) and syntax:
                        line_parts.append(f"Syntax: {', '.join(str(s) for s in syntax)}")
                    if isinstance(examples, list) and examples:
                        sample = "; ".join(str(e) for e in examples[:3])
                        if len(examples) > 3:
                            sample += "; ..."
                        line_parts.append(f"Examples: {sample}")
                    lines.append(" ".join(line_parts))
            lines.append("")
        documents.append(
            {
                "id": "cbc:operators",
                "text": "\n".join(lines).strip(),
                "metadata": {"section": "operators"},
            }
        )

    best_practices = schema.get("best_practices")
    if isinstance(best_practices, dict) and best_practices:
        lines = ["Best Practices:"]
        for category, tips in sorted(best_practices.items()):
            lines.append(f"Category: {category}")
            if isinstance(tips, list):
                for tip in tips:
                    lines.append(f"- {tip}")
            lines.append("")
        documents.append(
            {
                "id": "cbc:best_practices",
                "text": "\n".join(lines).strip(),
                "metadata": {"section": "best_practices"},
            }
        )

    guidelines = schema.get("query_building_guidelines")
    if isinstance(guidelines, dict) and guidelines:
        lines = ["Query Building Guidelines:"]
        for step, meta in sorted(guidelines.items()):
            if not isinstance(meta, dict):
                continue
            title = step.replace("_", " ").title()
            description = str(meta.get("description", ""))
            lines.append(f"Step: {title}")
            if description:
                lines.append(f"- {description}")
            for key in ("questions", "considerations", "rules", "validations", "tips"):
                entries = meta.get(key)
                if isinstance(entries, list) and entries:
                    lines.append(f"  {key.title()}:")
                    for entry in entries:
                        lines.append(f"    - {entry}")
            lines.append("")
        documents.append(
            {
                "id": "cbc:guidelines",
                "text": "\n".join(lines).strip(),
                "metadata": {"section": "guidelines"},
            }
        )

    example_queries = schema.get("example_queries")
    if isinstance(example_queries, dict) and example_queries:
        lines = ["Example Queries:"]
        for category, examples in sorted(example_queries.items()):
            lines.append(f"Category: {category}")
            if isinstance(examples, list):
                for example in examples:
                    if isinstance(example, dict):
                        title = str(example.get("title", ""))
                        query = str(example.get("query", ""))
                        description = str(example.get("description", ""))
                        if title:
                            lines.append(f"- {title}")
                        if description:
                            lines.append(f"  Description: {description}")
                        if query:
                            lines.append(f"  Query: {query}")
                    else:
                        lines.append(f"- {example}")
            lines.append("")
        documents.append(
            {
                "id": "cbc:examples",
                "text": "\n".join(lines).strip(),
                "metadata": {"section": "examples"},
            }
        )

    return documents


def build_kql_documents(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert the Defender schema into retrieval-friendly documents."""

    documents: List[Dict[str, Any]] = []
    for table in sorted(schema.keys()):
        table_info = schema.get(table, {})
        if not isinstance(table_info, dict):
            continue
        url = str(table_info.get("url", ""))
        columns = table_info.get("columns", []) or []

        column_lines: List[str] = []
        for column in columns:
            if not isinstance(column, dict):
                continue
            name = str(column.get("name", ""))
            ctype = str(column.get("type", ""))
            description = str(column.get("description", ""))
            parts = [part for part in [name, f"({ctype})" if ctype else "", description] if part]
            if parts:
                column_lines.append(" ".join(parts))

        if not column_lines:
            column_lines.append("No column metadata available.")

        text = "\n".join(
            [
                f"Table: {table}",
                f"Documentation: {url}" if url else "Documentation: (missing)",
                "Columns:",
                *column_lines,
            ]
        )

        documents.append(
            {
                "id": f"kql:{table}",
                "text": text,
                "metadata": {"table": table, "url": url},
            }
        )

    return documents
