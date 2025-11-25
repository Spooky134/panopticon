import av
import asyncio
from datetime import datetime

from core.logger import get_logger


logger = get_logger(__name__)


#TODO контекстный менеджер или функция с конеткстом
class BaseFrameCollector:
    def __init__(self):
        self.output_file = None

    async def add_frame(self, frame):
        raise NotImplementedError

    async def finalize(self):
        raise NotImplementedError

    async def get_metadata(self):
        raise NotImplementedError