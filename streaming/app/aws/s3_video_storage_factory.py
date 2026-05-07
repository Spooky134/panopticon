from app.aws.s3_video_storage import S3VideoStorage
import aioboto3
from app.config.settings import settings


async def create_s3_video_storage() -> S3VideoStorage:
    session_args = {"region_name": settings.s3.region}
    client_args = {"service_name": "s3"}

    if settings.s3.url:
        client_args["endpoint_url"] = settings.s3.url
        client_args["aws_access_key_id"] = settings.s3.access_key
        client_args["aws_secret_access_key"] = settings.s3.secret_key
        client_args["use_ssl"] = False

    session = aioboto3.Session(**session_args)
    s3_client = await session.client(**client_args).__aenter__()


    video_storage = S3VideoStorage(
        s3_client=s3_client,
        bucket_name=settings.s3.bucket_name
    )

    await video_storage.ensure_bucket()
    return video_storage