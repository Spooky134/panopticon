import os
import av
import asyncio
from datetime import datetime
from uuid import UUID

from utils.base_frame_collector import BaseFrameCollector

from core.logger import get_logger


logger = get_logger(__name__)


class FrameCollector(BaseFrameCollector):
    def __init__(self, session_id: UUID):
        super().__init__()
        self.session_id = session_id
        self.frames = []
        self.start_time = datetime.now()
        self.output_file = f"/tmp/{session_id}.mp4"
        self._lock = asyncio.Lock()
        self.metadata = None

    async def add_frame(self, frame):
        try:
            async with self._lock:
                self.frames.append(frame.to_ndarray(format="bgr24"))
        except Exception as e:
            logger.error(f"session: {self.session_id} - Error adding frame in session: {e}")

    async def finalize(self):
        logger.info(f"session: {self.session_id} - We are starting finalization of frames={len(self.frames)}")
        if not self.frames:
            logger.warning(f"session: {self.session_id} - There are no frames to save.")
            return None

        logger.info(f"session: {self.session_id} - Save {len(self.frames)} frames to {self.output_file}")
        try:
            container = av.open(self.output_file, mode="w")
            stream = container.add_stream("libx264", rate=30)
            stream.pix_fmt = "yuv420p"
            stream.width = self.frames[0].shape[1]
            stream.height = self.frames[0].shape[0]

            for frame_array in self.frames:
                frame = av.VideoFrame.from_ndarray(frame_array, format="bgr24")
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)

            container.close()

            await self._calculate_metadata()

            logger.info(f"session: {self.session_id} - The video is saved locally: {self.output_file}")
        except Exception as e:
            logger.error(f"session: {self.session_id} - Error while compiling video: {e}")

        return self.output_file

    #TODO прокачать метод
    async def _calculate_metadata(self):
        file_size = os.path.getsize(self.output_file)
        duration = len(self.frames) / 30.0
        mime_type = "video/mp4"

        self.metadata = {
            "duration": duration,
            "file_size": file_size,
            "mime_type": mime_type
        }