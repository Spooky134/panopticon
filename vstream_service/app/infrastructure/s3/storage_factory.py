from typing import AsyncGenerator
from fastapi import Depends
from config.settings import settings
from infrastructure.s3.storage import S3Storage
from core.s3_client import get_s3_client


async def get_s3_storage(s3_client=Depends(get_s3_client)) -> AsyncGenerator[S3Storage, None]:
    storage = S3Storage(
        s3_client=s3_client,
        bucket_name=settings.S3_BUCKET_NAME
    )
    await storage.ensure_bucket()  # Создаем бакет если нужно
    yield storage