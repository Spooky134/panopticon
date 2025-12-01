import os
import av
import asyncio
from uuid import UUID
from utils.base_frame_collector import BaseFrameCollector

from core.logger import get_logger


logger = get_logger(__name__)


class FrameCollector(BaseFrameCollector):
    def __init__(self, session_id: UUID):
        super().__init__()
        self.session_id = session_id
        self.frames = []

        self.temp_dir = f"/tmp/collected_data/"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.file_name = f"{session_id}.mp4"
        self.output_file_path = os.path.join(self.temp_dir, self.file_name)

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

        logger.info(f"session: {self.session_id} - Save {len(self.frames)} frames to {self.output_file_path}")
        try:
            container = av.open(self.output_file_path, mode="w")
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

            logger.info(f"session: {self.session_id} - The video is saved locally: {self.output_file_path}")
        except Exception as e:
            logger.error(f"session: {self.session_id} - Error while compiling video: {e}")

        return self.output_file_path

    async def cleanup(self):
        logger.info(f"session: {self.session_id} - collector cleaning up")
        try:
            if os.path.exists(self.output_file_path):
                os.remove(self.output_file_path)
            logger.info(f"session: {self.session_id} - temporary file removed")
        except Exception as e:
            logger.error(f"session: {self.session_id} - error removing temporary file: {e}")

    async def get_output_file_path(self):
        return self.output_file_path

    #TODO прокачать метод
    async def get_metadata(self):
        file_size = os.path.getsize(self.output_file_path)
        duration = len(self.frames) / 30.0
        mime_type = "video/mp4"

        if not self.metadata:
            self.metadata = {
                "duration": duration,
                "file_size": file_size,
                "mime_type": mime_type
            }

        return self.metadata