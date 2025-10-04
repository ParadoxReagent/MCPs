from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class CBCSchemaCache:
    """Load and cache the Carbon Black Cloud EDR schema file."""

    def __init__(self, schema_path: Path) -> None:
        self.schema_path = Path(schema_path)
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] | None = None

    def load(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            if force_refresh or self._cache is None:
                raw = self.schema_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Schema root must be a JSON object")
                payload = data.get("carbonblack_edr_query_schema")
                if not isinstance(payload, dict):
                    raise ValueError("Missing 'carbonblack_edr_query_schema' root key")
                self._cache = payload
            return self._cache

    # Convenience helpers -------------------------------------------------

    def search_types(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.load().get("search_types", {}))

    def field_map_for(self, search_type: str) -> Dict[str, Dict[str, Any]]:
        payload = self.load()
        mapping_key = {
            "process_search": "process_search_fields",
            "binary_search": "binary_search_fields",
            "alert_search": "alert_search_fields",
            "threat_report_search": "threat_report_search_fields",
        }.get(search_type)

        if not mapping_key:
            return {}

        fields = payload.get(mapping_key, {})
        return dict(fields) if isinstance(fields, dict) else {}

    def list_fields(self, search_type: str) -> List[Dict[str, Any]]:
        fields = self.field_map_for(search_type)
        output: List[Dict[str, Any]] = []
        for name, meta in sorted(fields.items()):
            if isinstance(meta, dict):
                entry = {"name": name}
                entry.update(meta)
                output.append(entry)
        return output

    def operator_reference(self) -> Dict[str, Any]:
        payload = self.load()
        return payload.get("operators", {})

    def best_practices(self) -> List[str] | Dict[str, Any]:
        payload = self.load()
        best = payload.get("best_practices")
        return best if isinstance(best, (list, dict)) else []

    def example_queries(self) -> Dict[str, Any]:
        payload = self.load()
        examples = payload.get("example_queries", {})
        return examples if isinstance(examples, dict) else {}


def normalise_search_type(name: str | None, available: Iterable[str]) -> Tuple[str, List[str]]:
    """Return a valid search type and a record of the normalisation steps."""

    available_list = [st for st in available]
    log: List[str] = []

    if not name:
        if available_list:
            default = available_list[0]
            log.append(f"defaulted_to:{default}")
            return default, log
        raise ValueError("No search types available in schema")

    cleaned = name.strip().lower().replace(" ", "_")
    candidates = {
        "process": "process_search",
        "process_search": "process_search",
        "binary": "binary_search",
        "binary_search": "binary_search",
        "alert": "alert_search",
        "alert_search": "alert_search",
        "alerts": "alert_search",
        "threat": "threat_report_search",
        "threat_report": "threat_report_search",
        "threat_report_search": "threat_report_search",
        "report": "threat_report_search",
    }

    resolved = candidates.get(cleaned, cleaned)
    if resolved in available_list:
        if resolved != name:
            log.append(f"normalised_from:{name}->{resolved}")
        return resolved, log

    # Attempt fuzzy fallback by prefix
    for candidate in available_list:
        if candidate.startswith(resolved):
            log.append(f"prefix_matched:{candidate}")
            return candidate, log

    raise ValueError(f"Unknown search type '{name}'. Valid options: {', '.join(available_list)}")
