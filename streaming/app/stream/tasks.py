import os
from app.core.taskiq_broker import broker
from app.core.logging import get_logger
from taskiq import Context, TaskiqDepends

logger = get_logger(__name__)


@broker.task
async def uploading_video_to_s3(
        session_id: str,
        s3_key: str,
        file_path: str,
        context: Context = TaskiqDepends()
) -> None:
        try:
            storage = context.state.s3_storage
            await storage.upload_multipart(
                session_id=session_id,
                file_path=file_path,
                object_name=s3_key
            )

            logger.info(f"session: {session_id} - background video uploading success")
        except Exception as e:
            logger.error(f"session: {session_id} - background upload critical error: {e}")
        finally:
            # TODO нужно добавить умную очистку чтобы при ошибки не удалять файл а пробовать загрузить повторно
            # TODO также добавить очистку накопившихся файлов
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"session: {session_id} - temporary file removed")
            except Exception as e:
                logger.error(f"session: {session_id} - error removing temporary file: {e}")
