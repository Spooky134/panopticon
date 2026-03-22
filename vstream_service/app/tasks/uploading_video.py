import app.video.models
import app.session.models
from app.core import broker
from app.core.logger import get_logger
from app.core.unit_of_work import UnitOfWork
from vstream_service.app.session.repository import SessionRepository


logger = get_logger(__name__)


@broker.task
async def uploading_video_to_s3(self, session_id: str, file_path: str, video_meta: VideoMetaEntity):
        try:
            async with UnitOfWork as uow:
                repo = SessionRepository(uow.session)
                storage = uploading_video_to_s3.state.s3_storage


                s3_key = await storage.upload_multipart(
                    streaming_session_id=session_id,
                    file_path=file_path,
                    object_name=str(session_id),
                    mime_type=video_meta.mime_type
                )
                if s3_key is not None:
                    video_entity = StreamingVideoEntity(
                        s3_key=s3_key,
                        streaming_session_id=session_id,
                        meta=video_meta
                    )

                    await repo.attach_video(
                        session_id,
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
