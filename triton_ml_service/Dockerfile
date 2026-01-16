# Используем официальный образ Triton. Версия 24.12 — самая свежая на начало 2026.
FROM nvcr.io/nvidia/tritonserver:24.12-py3

# Устанавливаем системные зависимости, которые нужны OpenCV и твоей модели
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


COPY triton_ml_service/requirements.txt /tmp/requirements.txt


RUN pip3 install --no-cache-dir -r /tmp/requirements.txt


WORKDIR /opt/tritonserver


EXPOSE 8000 8001 8002


CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]