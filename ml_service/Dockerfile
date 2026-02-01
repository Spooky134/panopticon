FROM nvcr.io/nvidia/tritonserver:25.12-py3

RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxv-dev \
    libx264-dev \
    build-essential cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ml_service/pyproject.toml ml_service/poetry.lock ./

RUN pip install --no-cache-dir poetry==2.3.1

RUN poetry config virtualenvs.create false  \
    && poetry install --no-interaction --no-ansi --only main --no-root

COPY proto/ /app/proto/

RUN mkdir -p /app/generated && \
    python3 -m grpc_tools.protoc \
        -I/app/proto \
        --python_out=/app/generated \
        --grpc_python_out=/app/generated \
        /app/proto/ml_worker.proto


WORKDIR /opt/tritonserver


EXPOSE 8000 8001 8002
