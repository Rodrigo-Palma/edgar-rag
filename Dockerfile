# syntax=docker/dockerfile:1
#
# The answering service, without the models: Ollama stays on the host, where it
# has the GPU (see compose.yaml). The index is not baked in either; it is
# mounted read-only at /data/index, so one image serves any index.
#
# Both base images are pinned by digest, written out in each FROM rather than
# through an ARG so that Dependabot (docker ecosystem) can read and bump them.
# The python image appears twice; the two lines move together.

FROM ghcr.io/astral-sh/uv:0.12.22@sha256:f513a91fc62fe7c17567eee97230dd198e43edb8a9fbecca843714a4358fe1bc AS uv

# --- build: resolve nothing, install exactly what uv.lock pins ---------------
FROM python:3.13-slim@sha256:8296499feed1c18bd8064c279d45e2a1b4b6be586f8b9e16dcf2aaf843480d88 AS build

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /src

# Dependencies first, in their own layer, so a change to the code alone does
# not reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- runtime: the virtual environment and nothing that built it --------------
FROM python:3.13-slim@sha256:8296499feed1c18bd8064c279d45e2a1b4b6be586f8b9e16dcf2aaf843480d88 AS runtime

RUN groupadd --system --gid 10001 edgar \
    && useradd --system --uid 10001 --gid edgar --no-create-home --shell /usr/sbin/nologin edgar

COPY --from=build --chown=root:root /app/.venv /app/.venv

# EDGAR_RAG_HOST: inside a container the service has to listen on every
# interface to be reachable at all; compose.yaml publishes the port on the
# host's 127.0.0.1 only, since there is no authentication.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EDGAR_RAG_HOST=0.0.0.0 \
    EDGAR_RAG_PORT=8000 \
    EDGAR_RAG_INDEX_DIR=/data/index

WORKDIR /app
USER edgar:edgar
EXPOSE 8000

# Healthy only when the service answers and has an index loaded. Ollama being
# down does not make it unhealthy: /health reports that, and /ask says so.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json, sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0 if json.load(r)['status'] == 'ready' else 1)"]

CMD ["edgar-rag", "serve"]
