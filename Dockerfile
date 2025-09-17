# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser esi_agents ./esi_agents

ARG EXTRAS="dev"
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e .${EXTRAS:+[${EXTRAS}]} \
    && rm -rf /root/.cache/pip

COPY --chown=appuser:appuser . .

USER appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

CMD ["python", "-m", "esi_agents.cli.esi_batch", "--help"]
