from app.aws.s3_video_storage import S3VideoStorage
import aioboto3
from config.settings import settings


async def create_s3_video_storage() -> S3VideoStorage:
    session_args = {"region_name": settings.S3_REGION}
    client_args = {"service_name": "s3"}

    if settings.S3_URL:
        client_args["endpoint_url"] = settings.S3_URL
        client_args["aws_access_key_id"] = settings.S3_ACCESS_KEY or "minioadmin"
        client_args["aws_secret_access_key"] = settings.S3_SECRET_KEY or "minioadmin"
        client_args["use_ssl"] = False

    session = aioboto3.Session(**session_args)
    s3_client = await session.client(**client_args).__aenter__()


    video_storage = S3VideoStorage(
        s3_client=s3_client,
        bucket_name=settings.S3_BUCKET_NAME
    )

    await video_storage.ensure_bucket()
    return video_storage