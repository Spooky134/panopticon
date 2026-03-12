FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libavfilter-dev \
    libsrtp2-dev libopus-dev libvpx-dev ffmpeg \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY vstream_service/pyproject.toml vstream_service/poetry.lock ./


RUN pip install --no-cache-dir poetry==2.3.1

RUN poetry config virtualenvs.create false  \
    && poetry install --no-interaction --no-ansi --without dev --no-root

COPY vstream_service .

COPY proto/ /app/proto/

RUN mkdir -p /app/generated && \
    python3 -m grpc_tools.protoc \
        -I/app/proto \
        --python_out=/app/generated \
        --grpc_python_out=/app/generated \
        /app/proto/ml_worker.proto
