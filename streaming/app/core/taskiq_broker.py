from taskiq import TaskiqEvents, TaskiqState
from taskiq_aio_pika import AioPikaBroker

from app.core import logging
from app.config.settings import settings
from app.aws.s3_video_storage_factory import create_s3_video_storage


logger = logging.get_logger(__name__)

broker = AioPikaBroker(
    url=str(settings.taskiq.url),
)

@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def on_worker_startup(state: TaskiqState) -> None:
    # Конфигурирование логирования для воркера
    state.s3_storage = await create_s3_video_storage()
    logger.info("Worker startup complete, got state: %s", state)

@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def on_worker_shutdown(state: TaskiqState) -> None:
    # Конфигурирование логирования для воркера
    await state.s3_storage.close()
    logger.info("Worker shutdown complete, got state: %s", state)