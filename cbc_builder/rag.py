"""Retrieval utilities for Carbon Black Cloud schema content."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - import guard
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

try:  # pragma: no cover - import guard
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled at runtime
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:  # pragma: no cover - import guard
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - handled at runtime
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

try:  # pragma: no cover - import guard
    from rapidfuzz import process as rapidfuzz_process
except ImportError:  # pragma: no cover - handled at runtime
    rapidfuzz_process = None  # type: ignore[assignment]

from .schema_loader import CBCSchemaCache

logger = logging.getLogger(__name__)


def _ensure_sentence_transformers() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for retrieval. Install the optional dependencies.",
        ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR
    return SentenceTransformer  # type: ignore[return-value]


@dataclass
class RAGService:
    """Build and reuse embeddings for the Carbon Black Cloud schema documentation."""

    schema_cache: CBCSchemaCache
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
        self._schema_version: Optional[str] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            SentenceModel = _ensure_sentence_transformers()
            logger.info("Loading sentence transformer model '%s'", self.model_name)
            self._model = SentenceModel(self.model_name)
        return self._model

    def _field_summary(self, field_name: str, meta: Dict[str, Any]) -> str:
        field_type = str(meta.get("type", ""))
        description = str(meta.get("description", ""))
        values = meta.get("values")
        default_flag = meta.get("default_field")

        qualifiers: List[str] = []
        if field_type:
            qualifiers.append(field_type)
        if default_flag:
            qualifiers.append("default")

        header = field_name
        if qualifiers:
            header += f" ({', '.join(qualifiers)})"

        lines = [header]
        if description:
            lines.append(f"- {description}")
        if isinstance(values, list) and values:
            preview = ", ".join(str(v) for v in values[:5])
            if len(values) > 5:
                preview += ", ..."
            lines.append(f"Values: {preview}")
        return " ".join(lines).strip()

    def _build_documents(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                        if isinstance(field_meta, dict):
                            field_lines.append(self._field_summary(field_name, field_meta))
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
                        "id": f"search_type:{name}",
                        "section": "search_types",
                        "name": name,
                        "text": text,
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
                    "id": "field_types",
                    "section": "field_types",
                    "text": "\n".join(["Field Type Reference:", *lines]) if lines else "Field Type Reference:",
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
                            line_parts.append(f"{description}")
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
                    "id": "operators",
                    "section": "operators",
                    "text": "\n".join(lines).strip(),
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
                    "id": "best_practices",
                    "section": "best_practices",
                    "text": "\n".join(lines).strip(),
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
                    "id": "guidelines",
                    "section": "guidelines",
                    "text": "\n".join(lines).strip(),
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
                    "id": "example_queries",
                    "section": "examples",
                    "text": "\n".join(lines).strip(),
                }
            )

        return documents

    def _documents_signature(self, documents: List[Dict[str, Any]]) -> str:
        payload = json.dumps(
            [
                {
                    "id": doc.get("id"),
                    "section": doc.get("section"),
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
        """Ensure the embedding index exists for the CBC schema."""

        schema = self.schema_cache.load(force_refresh=force)
        schema_version = str(schema.get("version", "")) if isinstance(schema, dict) else None

        if (
            not force
            and self._mode != "uninitialized"
            and schema_version
            and schema_version == self._schema_version
        ):
            return

        documents = self._build_documents(schema)
        signature = self._documents_signature(documents)

        if not force and self._load_cached_index(signature):
            self._documents = documents
            self._schema_version = schema_version
            return

        if SentenceTransformer is None or faiss is None or np is None:
            if rapidfuzz_process is None:
                raise RuntimeError(
                    "Retrieval dependencies are unavailable and rapidfuzz fallback is missing.",
                )
            logger.warning(
                "Embedding libraries unavailable; using rapidfuzz-based fallback retrieval.",
            )
            self._documents = documents
            self._dimension = None
            self._index = None
            self._mode = "fuzzy"
            self._schema_version = schema_version
            with self._metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "signature": signature,
                        "mode": "fuzzy",
                        "documents": documents,
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
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self._index = index
        self._documents = documents
        self._dimension = dimension
        self._schema_version = schema_version
        logger.info("Persisted new embeddings cache to %s", self.cache_dir)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k documents matching the query."""

        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")

        self.ensure_index()
        top_k = min(k, len(self._documents))
        if top_k == 0:
            return []

        if self._mode == "fuzzy":
            if rapidfuzz_process is None:
                raise RuntimeError("rapidfuzz is required for fallback retrieval")
            matches = rapidfuzz_process.extract(
                query,
                [doc["text"] for doc in self._documents],
                limit=top_k,
            )
            results: List[Dict[str, Any]] = []
            for _, score, idx in matches:
                doc = self._documents[idx]
                results.append(
                    {
                        "id": doc.get("id"),
                        "section": doc.get("section"),
                        "text": doc.get("text"),
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

        scores, indices = self._index.search(query_embedding, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            results.append(
                {
                    "id": doc.get("id"),
                    "section": doc.get("section"),
                    "text": doc.get("text"),
                    "score": float(score),
                }
            )
        return results

    def clear_cache(self) -> None:
        """Remove cached embeddings (primarily for testing)."""

        if self._metadata_path.exists():
            self._metadata_path.unlink()
        if self._index_path.exists():
            self._index_path.unlink()
        self._index = None
        self._documents = []
        self._dimension = None
        self._mode = "uninitialized"
        self._schema_version = None
        logger.info("Cleared RAG embeddings cache.")
