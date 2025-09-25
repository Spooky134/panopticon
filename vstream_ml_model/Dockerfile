FROM python:3.13.5-slim

WORKDIR /app


# Устанавливаем curl для healthcheck
RUN apt-get update && apt-get install -y curl && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ .

ENV PYTHONPATH=/

CMD ["python", "main.py"]