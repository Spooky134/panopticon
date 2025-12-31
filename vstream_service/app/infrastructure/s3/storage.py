from botocore.exceptions import ClientError

from core.logger import get_logger


logger = get_logger(__name__)


class S3Storage:
    def __init__(self, s3_client, bucket_name: str, prefix="videos"):
        self.client = s3_client
        self.bucket_name = bucket_name
        self.prefix = prefix

    async def ensure_bucket(self):
        try:
            await self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' found")
        except ClientError as e:
            logger.warning(f"Bucket not found, creating one: {self.bucket_name} ({e})")
            await self.client.create_bucket(Bucket=self.bucket_name)

    async def upload_file(self, file_path: str, object_name: str) -> str:
        logger.info(f"Simple Loading {file_path} → {self.bucket_name}/{self.prefix}/{object_name}")

        s3_key = f"{self.prefix}/{object_name}"
        try:
            await self.client.upload_file(Bucket=self.bucket_name,
                                          Key=s3_key,
                                          Filename=file_path,
                                          )
            logger.info(f"Successfully uploaded {object_name}")
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")

        return s3_key

    async def upload_multipart(self, file_path: str, object_name: str):
        logger.info(f"Multipart loading {file_path} → {self.bucket_name}/{self.prefix}/{object_name}")
        s3_key = f"{self.prefix}/{object_name}"
        try:
            mp = await self.client.create_multipart_upload(Bucket=self.bucket_name, Key=s3_key)
            parts = []
            part_number = 1
            chunk_size = 5 * 1024 * 1024

            with open(file_path, "rb") as file:
                while chunk := file.read(chunk_size):
                    resp = await self.client.upload_part(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=mp["UploadId"],
                        Body=chunk,
                    )
                    parts.append({
                        "PartNumber": part_number,
                        "ETag": resp["ETag"]
                    })
                    part_number += 1

            await self.client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=mp["UploadId"],
                MultipartUpload={"Parts": parts}
            )

            logger.info(f"Object {object_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error when upload_bytes: {e}")

        return s3_key