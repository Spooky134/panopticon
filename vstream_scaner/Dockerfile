FROM python:3.13-bookworm

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода не требуется, так как том синхронизирует файлы
# COPY . .

EXPOSE 8000

# Команда запуска будет переопределена в docker-compose.yml
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]