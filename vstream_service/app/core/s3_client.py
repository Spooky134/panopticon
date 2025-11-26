import aioboto3
from config.settings import settings
from typing import AsyncGenerator, Optional


s3_client_instance = None  # глобально хранить клиент

async def get_s3_client() -> AsyncGenerator:
    global s3_client_instance
    if s3_client_instance is None:
        session_args = {"region_name": settings.S3_REGION}
        client_args = {"service_name": "s3"}

        if settings.S3_ENDPOINT_URL:
            client_args["endpoint_url"] = settings.S3_ENDPOINT_URL
            client_args["aws_access_key_id"] = settings.S3_ACCESS_KEY or "minioadmin"
            client_args["aws_secret_access_key"] = settings.S3_SECRET_KEY or "minioadmin"
            client_args["use_ssl"] = False

        session = aioboto3.Session(**session_args)
        s3_client_instance = await session.client(**client_args).__aenter__()  # не закрываем, пока живо приложение

    yield s3_client_instance