FROM python:3.13.5-slim

WORKDIR /app


# Устанавливаем curl для healthcheck
# RUN apt-get update && apt-get install -y curl && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ .

ENV PYTHONPATH=/ \
    HOST=0.0.0.0

EXPOSE 8000
# STUN
EXPOSE 3478/udp
EXPOSE 49152-65535/udp 

CMD ["python", "main.py"]