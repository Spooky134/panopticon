import mimetypes
from pathlib import Path
from typing import Optional
from uuid import UUID
import aiofiles
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

    def _generate_s3_key(self, object_name: str, file_path: str, prefix: str) -> str:
        ext = Path(file_path).suffix
        return f"{prefix}/{object_name}{ext}"

    def guess_content_type(self, file_path: str) -> str:
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type

    async def ensure_bucket(self):
        try:
            await self._client.head_bucket(Bucket=self._bucket_name)
            logger.info(f"bucket '{self._bucket_name}' found")
        except ClientError as e:
            logger.warning(f"bucket not found, creating one: {self._bucket_name} ({e})")
            await self._client.create_bucket(Bucket=self._bucket_name)

    async def upload_multipart(self, streaming_session_id: UUID, file_path: str,
                               object_name: str, prefix:str="videos", mime_type: str="video/mp4") -> Optional[str]:
        s3_key = self._generate_s3_key(
            object_name=object_name,
            file_path=file_path,
            prefix=prefix,
        )
        logger.info(
            f"session: {streaming_session_id} - multipart loading {file_path}"
            f" → {self._bucket_name}/{s3_key}"
        )

        upload_id = None
        try:
            create_kwargs = {"Bucket": self._bucket_name, "Key": s3_key}
            if mime_type:
                create_kwargs["ContentType"] = mime_type
            else:
                create_kwargs["ContentType"] = self.guess_content_type(file_path)

            mp = await self._client.create_multipart_upload(**create_kwargs)
            upload_id = mp["UploadId"]

            parts = []
            part_number = 1
            chunk_size = 5 * 1024 * 1024

            async with aiofiles.open(file_path, "rb") as file:
                while chunk := await file.read(chunk_size):
                    resp = await self._client.upload_part(
                        Bucket=self._bucket_name,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
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
                UploadId=upload_id,
                MultipartUpload={"Parts": parts}
            )
            logger.info(f"session: {streaming_session_id} - upload success")
            return s3_key
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - error upload failed: {e}")
            await self.abort(
                streaming_session_id=streaming_session_id,
                upload_id=upload_id,
                s3_key=s3_key
            )
            return None

    async def abort(self, streaming_session_id: UUID, upload_id: Optional[str], s3_key: str):
        logger.warning(f"session: {streaming_session_id} - aborting multipart upload {upload_id}")
        if not upload_id:
            return

        try:
            await self._client.abort_multipart_upload(
                Bucket=self._bucket_name,
                Key=s3_key,
                UploadId=upload_id
            )
        except Exception as abort_err:
            logger.error(f"session: {streaming_session_id} - failed to abort upload: {abort_err}")

    async def close(self):
        if self._client:
            logger.info(f"closing s3 storage client")
            await self._client.close()