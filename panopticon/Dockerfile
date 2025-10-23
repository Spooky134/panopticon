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
#
#RUN pip install --upgrade pip

COPY panopticon/requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY panopticon/panopticon/ .

# EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]