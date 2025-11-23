from botocore.exceptions import ClientError

from core.logger import get_logger


logger = get_logger(__name__)


class S3Storage:
    def __init__(self, s3_client, bucket_name: str):
        self.client = s3_client
        self.bucket_name = bucket_name

    async def ensure_bucket(self):
        try:
            await self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' found")
        except ClientError as e:
            logger.warning(f"Bucket not found, creating one: {self.bucket_name} ({e})")
            await self.client.create_bucket(Bucket=self.bucket_name)

    async def upload_file(self, file_path: str, object_name: str) -> str:
        logger.info(f"Loading {file_path} → {self.bucket_name}/{object_name}")
        try:
            await self.client.upload_file(file_path, self.bucket_name, object_name)
            logger.info(f"Successfully uploaded {object_name}")
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")

        return object_name

    async def upload_bytes(self, data: bytes, object_name: str, content_type="video/mp4"):
        logger.info(f"Loading an object {object_name} ({len(data)} byte)")
        try:
            await self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=data,
                ContentType=content_type,
            )
            logger.info(f"Object {object_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error when upload_bytes: {e}")
