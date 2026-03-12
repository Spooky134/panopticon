from taskiq import TaskiqEvents, TaskiqState
from taskiq_aio_pika import AioPikaBroker

from app.core import logging
from app.config.settings import settings

logger = logging.get_logger(__name__)

broker = AioPikaBroker(
    url=str(settings.taskiq.url),
)

@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def on_worker_startup(state: TaskiqState) -> None:
    # Конфигурирование логирования для воркера

    logger.info("Worker startup complete, got state: %s", state)
