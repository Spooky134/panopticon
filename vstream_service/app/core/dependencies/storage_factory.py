# dependencies/s3_client.py
from typing import AsyncGenerator
from fastapi import Depends
from config.config import settings
from storage.s3_storage import S3Storage
from config.s3_client import get_s3_client

async def get_s3_storage(s3_client=Depends(get_s3_client)) -> AsyncGenerator[S3Storage, None]:
    """Зависимость для S3Storage с готовым клиентом"""
    storage = S3Storage(
        s3_client=s3_client,
        bucket_name=settings.S3_BUCKET_NAME
    )
    await storage.ensure_bucket()  # Создаем бакет если нужно
    yield storage