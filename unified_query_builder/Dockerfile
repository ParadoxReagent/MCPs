# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_USER=app \
    APP_UID=10001 \
    APP_GID=10001 \
    APP_HOME=/app \
    CACHE_DIR=/app/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git socat \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${APP_GID} ${APP_USER} \
 && useradd -m -u ${APP_UID} -g ${APP_GID} -s /usr/sbin/nologin ${APP_USER} \
 && mkdir -p ${APP_HOME} ${CACHE_DIR}

WORKDIR ${APP_HOME}

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    python - <<'PY'
import fastmcp, pydantic
print("Deps OK. FastMCP:", getattr(fastmcp, "__version__", "unknown"))
PY

RUN mkdir -p ${APP_HOME}/unified_query_builder

COPY server.py cbc_query_builder.py kql_query_builder.py rag.py \
     cbc_schema_loader.py kql_schema_loader.py __init__.py \
     cb_edr_schema.json entrypoint.sh ./unified_query_builder/
COPY defender_xdr_kql_schema_fuller ./unified_query_builder/defender_xdr_kql_schema_fuller

RUN chmod +x unified_query_builder/entrypoint.sh && \
    chown -R ${APP_UID}:${APP_GID} ${APP_HOME}

USER ${APP_UID}:${APP_GID}

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD test -w ${CACHE_DIR} || exit 1

ENTRYPOINT ["./unified_query_builder/entrypoint.sh"]
CMD ["python", "-m", "unified_query_builder.server"]
