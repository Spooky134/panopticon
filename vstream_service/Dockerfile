
## Устанавливаем curl для healthcheck
## RUN apt-get update && apt-get install -y curl && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
#RUN apt-get update && \
#    apt-get install -y \
#    curl \
#    libgl1 \
#    libglib2.0-0 \
#    libsm6 \
#    libxext6 \
#    libxrender-dev \
#    libavcodec-dev \
#    libavformat-dev \
#    libswscale-dev \
#    && rm -rf /var/lib/apt/lists/*


#EXPOSE 8000
## STUN
#EXPOSE 3478/udp
#EXPOSE 49152-65535/udp


FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/

WORKDIR /app

# RUN apt-get update && apt-get install -y curl && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
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

# Создаем protobuf файлы (если они еще не сгенерированы)
RUN python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    ml_worker.proto

#EXPOSE 8000
#EXPOSE 10000-10010/udp

CMD ["python", "main.py"]