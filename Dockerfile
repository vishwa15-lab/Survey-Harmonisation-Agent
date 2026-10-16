# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.12 slim — small footprint, no unnecessary OS packages
FROM python:3.12-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Vishwa Nath"
LABEL description="Agentic survey harmonisation pipeline — OpenAI Agents SDK"

# ── System dependencies ───────────────────────────────────────────────────────
# gcc needed to compile some Python packages (e.g. numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container
WORKDIR /app

# ── Install dependencies first (layer caching) ────────────────────────────────
# Copy only pyproject.toml first — so pip install is cached
# unless dependencies actually change (not on every code change)
COPY pyproject.toml .

# Install the project dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# ── Copy source code ──────────────────────────────────────────────────────────
# Done AFTER pip install so code changes don't bust the pip cache
COPY agent/      ./agent/
COPY core/       ./core/
COPY evals/      ./evals/
COPY mcp_server/ ./mcp_server/
COPY tests/      ./tests/
COPY main.py     .

# ── Output & data directories ─────────────────────────────────────────────────
# runs/ is where harmonized.csv, mapping.json, issues.json are written
# data/ is where SQLite session DB is stored (SQLiteSession needs this to exist)
RUN mkdir -p runs data

# ── Environment variables ─────────────────────────────────────────────────────
# These are overridden at runtime via docker-compose.yml or --env-file
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Default command ───────────────────────────────────────────────────────────
# Run oracle eval (no API key needed) — safe default for CI/testing
CMD ["python", "main.py", "eval", "--oracle"]
