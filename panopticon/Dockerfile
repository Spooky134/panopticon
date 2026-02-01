FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1 
ENV PYTHONPATH=/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY panopticon/pyproject.toml panopticon/poetry.lock ./

RUN pip install --no-cache-dir poetry==2.3.1

RUN poetry config virtualenvs.create false  \
    && poetry install --no-interaction --no-ansi --without dev --no-root

COPY panopticon/panopticon/ .