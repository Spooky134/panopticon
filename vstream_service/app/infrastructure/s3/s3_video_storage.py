from uuid import UUID

from botocore.exceptions import ClientError

from core.logger import get_logger


logger = get_logger(__name__)


class S3VideoStorage:
    def __init__(self, s3_client, bucket_name: str):
        self._client = s3_client
        self._bucket_name = bucket_name

    @property
    def bucket_name(self):
        return self._bucket_name

    async def ensure_bucket(self):
        try:
            await self._client.head_bucket(Bucket=self._bucket_name)
            logger.info(f"bucket '{self._bucket_name}' found")
        except ClientError as e:
            logger.warning(f"bucket not found, creating one: {self._bucket_name} ({e})")
            await self._client.create_bucket(Bucket=self._bucket_name)

    async def upload_file(self, file_path: str, streaming_session_id: str, prefix:str="videos") -> str:
        object_name = streaming_session_id
        logger.info(f"session: {streaming_session_id} - simple Loading {file_path} → {self._bucket_name}/{prefix}/{object_name}")

        s3_key = f"{prefix}/{object_name}"
        try:
            await self._client.upload_file(Bucket=self._bucket_name,
                                          Key=s3_key,
                                          Filename=file_path,
                                          )
            logger.info(f"session: {streaming_session_id} - successfully uploaded {object_name}")
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - error uploading file {file_path}: {e}")

        return s3_key

    async def upload_multipart(self, file_path: str, streaming_session_id: UUID, prefix:str="videos") -> str:
        object_name = streaming_session_id
        logger.info(f"session: {streaming_session_id} - multipart loading {file_path} → {self._bucket_name}/{prefix}/{object_name}")
        s3_key = f"{prefix}/{object_name}"
        try:
            mp = await self._client.create_multipart_upload(Bucket=self._bucket_name, Key=s3_key)
            parts = []
            part_number = 1
            chunk_size = 5 * 1024 * 1024

            with open(file_path, "rb") as file:
                while chunk := file.read(chunk_size):
                    resp = await self._client.upload_part(
                        Bucket=self._bucket_name,
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

            await self._client.complete_multipart_upload(
                Bucket=self._bucket_name,
                Key=s3_key,
                UploadId=mp["UploadId"],
                MultipartUpload={"Parts": parts}
            )
            logger.info(f"session: {streaming_session_id} - the video has been successfully uploaded to S3")

        except Exception as e:
            logger.error(f"session: {streaming_session_id} - error to save video to s3: {e}")
        #TODO вернуть s3_key если ничего не сломалось
        return s3_key

    async def close(self):
        if self._client:
            logger.info(f"closing S3 storage client")
            await self._client.__aexit__(None, None, None)