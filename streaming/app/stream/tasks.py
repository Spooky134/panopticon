import os
from uuid import UUID

import app.video.models
import app.session.models
from app.core.taskiq_broker import broker
from app.core.logging import get_logger
from app.core.unit_of_work import UnitOfWork
from app.session.repository import SessionRepository
from app.video.entities import StreamingVideoEntity, VideoMetaEntity
from taskiq import Context, TaskiqDepends

logger = get_logger(__name__)




@broker.task
async def uploading_video_to_s3(
        session_id: str,
        file_path: str,
        width: int,
        height:int,
        duration: float,
        codec: str,
        file_size: int,
        mime_type: str,
        fps: float,
        bit_rate: float,
        frame_count: int,
        context: Context = TaskiqDepends()
) -> None:
        try:
            async with UnitOfWork() as uow:
                repo = SessionRepository(uow.session)
                storage = context.state.s3_storage

                video_meta = VideoMetaEntity(
                    file_size=file_size,
                    duration=duration,
                    width=width,
                    height=height,
                    codec=codec,
                    frame_count=frame_count,
                    fps=fps,
                    bit_rate=bit_rate,
                    mime_type=mime_type
                )

                s3_key = await storage.upload_multipart(
                    streaming_session_id=session_id,
                    file_path=file_path,
                    object_name=str(session_id),
                    mime_type=mime_type
                )
                if s3_key is not None:
                    video_entity = StreamingVideoEntity(
                        s3_key=s3_key,
                        streaming_session_id=UUID(session_id),
                        meta=video_meta
                    )

                    await repo.attach_video(
                        UUID(session_id),
                        video_entity
                    )
                    await uow.commit()
                    logger.info(f"session: {session_id} - background video attach success")
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


# @broker.task
# async def parse_and_update_link_task(user_id: int, link_id: int) -> None:
#     logger.info(f"Parsing link {link_id} for user {user_id}")
#     link_parser = AsyncLinkInfoParser(headers=HEADERS)
#     async with UnitOfWork() as uow:
#         link_repo = LinkRepository(uow.session)
#         try:
#             link = await link_repo.get(user_id, link_id)
#             if not link:
#                 logger.info(f"Link {link_id} for user {user_id} not found.")
#                 return

#             parsed_link = await link_parser.fetch(link.url)

#             await link_repo.update(user_id, link_id, parsed_link)
#         except Exception as e:
#             logger.error(e)
