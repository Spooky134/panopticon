import os
from fractions import Fraction

import av
import asyncio
from uuid import UUID
from utils.base_frame_collector import BaseFrameCollector

from core.logger import get_logger


logger = get_logger(__name__)


class FrameCollector(BaseFrameCollector):
    def __init__(self, session_id: UUID):
        super().__init__()
        self._session_id = session_id

        self._temp_dir = f"/tmp/collected_data/"
        os.makedirs(self._temp_dir, exist_ok=True)
        self._file_name = f"{self._session_id}.mp4"
        self._output_file_path = os.path.join(self._temp_dir, self._file_name)

        self._lock = asyncio.Lock()
        self._metadata = None

        self._container = av.open(self._output_file_path, mode="w")
        self._stream = None
        self._frame_index = None

        self.FPS = 30
        self.CODEC = "libx264"
        self.BIT_RATE = 5_000_000
        self.MIME_TYPE = "video/mp4"
        self.PIXEL_FORMAT = "yuv420p"
        self.OPTIONS = {
            "crf": "23",
            "preset": "fast",
            "tune": "zerolatency"
        }


    @property
    def file_name(self):
        return self._file_name

    @property
    def output_file_path(self):
        return self._output_file_path

    def file_exists(self):
        return os.path.exists(self._output_file_path)

    async def add_frame(self, frame: av.VideoFrame):
        async with self._lock:
            try:
                if self._stream is None:
                    self._stream = self._container.add_stream(self.CODEC, rate=self.FPS, options=self.OPTIONS)
                    self._stream.width = frame.width
                    self._stream.height = frame.height
                    self._stream.pix_fmt = self.PIXEL_FORMAT
                    self._stream.time_base = Fraction(1, self.FPS)
                    self._stream.bit_rate = self.BIT_RATE
                    self._frame_index = 0

                frame.pts = self._frame_index
                frame.time_base = Fraction(1, self.FPS)
                self._frame_index += 1

                for packet in self._stream.encode(frame):
                    self._container.mux(packet)

            except Exception as e:
                logger.error(f"session: {self._session_id} - error adding frame in session: {e}")

    async def finalize(self):
        logger.info(f"session: {self._session_id} - finalize video data to {self._output_file_path}")
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)

            logger.info(f"session: {self._session_id} - the video is saved locally: {self._output_file_path}")
        except Exception as e:
            logger.error(f"session: {self._session_id} - error while compiling video: {e}")
        finally:
            self._container.close()
        try:
            logger.info(f"session: {self._session_id} - getting metadata...")
            await self.get_metadata()
        except Exception as e:
            logger.error(f"session: {self._session_id} - error getting metadata: {e}")

        return self._output_file_path, self._metadata

    async def cleanup(self):
        logger.info(f"session: {self._session_id} - collector cleaning up")
        try:
            if os.path.exists(self._output_file_path):
                os.remove(self._output_file_path)
            logger.info(f"session: {self._session_id} - temporary file removed")
        except Exception as e:
            logger.error(f"session: {self._session_id} - error removing temporary file: {e}")

    async def get_metadata(self):
        if self._metadata:
            return self._metadata

        if not os.path.exists(self._output_file_path):
            logger.warning(f"session: {self._session_id} - no metadata file found, creating new one")
            return None

        try:
            container = av.open(self._output_file_path, mode="r")
        except Exception as e:
            logger.error(f"session: {self._session_id} - failed to open video for metadata: {e}")
            return None
        video_stream = next((s for s in container.streams if s.type == "video"), None)

        if not video_stream:
            logger.error(f"session: {self._session_id} - no video stream found in file")
            return None

        if video_stream.duration is not None and video_stream.time_base is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            if video_stream.average_rate:
                duration = float(video_stream.frames) / float(video_stream.average_rate)
            else:
                duration = None

        file_size = os.path.getsize(self.output_file_path)

        if video_stream.average_rate:
            avg_fps = float(video_stream.average_rate)
        else:
            avg_fps = None

        self._metadata = {
            "path": self._output_file_path,
            "file_size": file_size,
            "duration": duration,
            "width": video_stream.codec_context.width,
            "height": video_stream.codec_context.height,
            "codec": video_stream.codec_context.name,
            "frame_count": video_stream.frames if video_stream.frames else None,
            "fps": avg_fps,
            "bit_rate": video_stream.bit_rate if video_stream.bit_rate else None,
            "mime_type": self.MIME_TYPE
        }
        container.close()

        return self._metadata