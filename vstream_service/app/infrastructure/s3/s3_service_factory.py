from typing import AsyncGenerator
from fastapi import Depends
from config.settings import settings
from infrastructure.s3.s3_service import S3Service
from core.s3_client import get_s3_client


async def get_s3_service(s3_client=Depends(get_s3_client)) -> AsyncGenerator[S3Service, None]:
    storage = S3Service(
        s3_client=s3_client,
        bucket_name=settings.S3_BUCKET_NAME
    )
    await storage.ensure_bucket()  # Создаем бакет если нужно
    yield storage