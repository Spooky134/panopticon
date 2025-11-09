FROM python:3.13-bookworm

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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY vstream_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY vstream_service/app/ .

COPY proto .

RUN python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    ml_worker.proto