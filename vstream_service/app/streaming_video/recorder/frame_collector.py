import os
from typing import Tuple
import av
from uuid import UUID
import threading
import queue
from fractions import Fraction

from app.core.logger import get_logger
from app.streaming_video.entities import VideoMetaEntity

logger = get_logger(__name__)


class FrameCollector:
    def __init__(self, streaming_session_id: UUID):
        self._STREAMING_SESSION_ID = streaming_session_id

        self._TEMP_DIR = f"/tmp/collected_data/"
        os.makedirs(self._TEMP_DIR, exist_ok=True)
        self._FILE_NAME = f"{self._STREAMING_SESSION_ID}.mp4"
        self._OUTPUT_FILE_PATH = os.path.join(self._TEMP_DIR, self._FILE_NAME)

        self.FPS = 30
        self.CODEC = "libx264"
        self.BIT_RATE = 2_000_000
        self.PIXEL_FORMAT = "yuv420p"
        self.OPTIONS = {
            "crf": "23",
            "preset": "fast",
            "tune": "zerolatency"
        }
        self.MIME_TYPE = "video/mp4"

        self._queue = queue.Queue(maxsize=30)

        self._running = True

        self._worker_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self._worker_thread.start()

        self._metadata = None

        self._container = None
        self._stream = None
        self._frame_index = 0

    async def add_frame(self, frame: av.VideoFrame):
        if not self._running:
            return
        try:
            # frame_copy = frame.reformat(format=frame.format)
            self._queue.put_nowait(frame)
        except queue.Full:
            logger.warning(f"session: {self._STREAMING_SESSION_ID} - drop recording frame (queue full)")
        except Exception as e:
            logger.error(f"session: {self._STREAMING_SESSION_ID} - error queuing frame: {e}")

    def _recording_worker(self):
        logger.info(f"session: {self._STREAMING_SESSION_ID} - recording thread started")

        try:
            self._container = av.open(self._OUTPUT_FILE_PATH, mode="w")
        except Exception as e:
            logger.error(f"session: {self._STREAMING_SESSION_ID} - failed to open file: {e}")
            return

        while self._running or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if self._stream is None:
                    self._stream = self._container.add_stream(self.CODEC, rate=self.FPS)
                    self._stream.width = frame.width
                    self._stream.height = frame.height
                    self._stream.pix_fmt = self.PIXEL_FORMAT
                    self._stream.bit_rate = self.BIT_RATE
                    self._stream.options = self.OPTIONS

                frame.pts = self._frame_index
                frame.time_base = Fraction(1, self.FPS)
                self._frame_index += 1

                for packet in self._stream.encode(frame):
                    self._container.mux(packet)

            except Exception as e:
                logger.error(f"session: {self._STREAMING_SESSION_ID} - encode error: {e}")
            finally:
                self._queue.task_done()

        if self._stream is not None:
            try:
                for packet in self._stream.encode():
                    self._container.mux(packet)
            except Exception as e:
                logger.error(f"session: {self._STREAMING_SESSION_ID} - final encode error: {e}")

        if self._container is not None:
            self._container.close()
            logger.info(f"session: {self._STREAMING_SESSION_ID} - container closed")

    async def finalize(self) -> Tuple[str, VideoMetaEntity]:
        logger.info(f"session: {self._STREAMING_SESSION_ID} - finalizing video data to {self._OUTPUT_FILE_PATH}")

        self._running = False

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            logger.info(f"session: {self._STREAMING_SESSION_ID} - recording thread stopped")

        try:
            logger.info(f"session: {self._STREAMING_SESSION_ID} - getting metadata...")
            self._metadata = await self.get_metadata()
        except Exception as e:
            logger.error(f"session: {self._STREAMING_SESSION_ID} - error getting metadata: {e}")
            self._metadata = None

        return self._OUTPUT_FILE_PATH, self._metadata

    async def get_metadata(self) -> VideoMetaEntity:
        if self._metadata:
            return self._metadata

        if not os.path.exists(self._OUTPUT_FILE_PATH):
            logger.warning(f"session: {self._STREAMING_SESSION_ID} - no metadata file found, creating new one")
            return None

        try:
            container = av.open(self._OUTPUT_FILE_PATH, mode="r")
        except Exception as e:
            logger.error(f"session: {self._STREAMING_SESSION_ID} - failed to open video for metadata: {e}")
            return None
        video_stream = next((s for s in container.streams if s.type == "video"), None)

        if not video_stream:
            logger.error(f"session: {self._STREAMING_SESSION_ID} - no video stream found in file")
            return None

        if video_stream.duration is not None and video_stream.time_base is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            if video_stream.average_rate:
                duration = float(video_stream.frames) / float(video_stream.average_rate)
            else:
                duration = None

        file_size = os.path.getsize(self._OUTPUT_FILE_PATH)

        if video_stream.average_rate:
            avg_fps = float(video_stream.average_rate)
        else:
            avg_fps = None

        self._metadata = VideoMetaEntity(file_size=file_size,
                                         duration=duration,
                                         width=video_stream.codec_context.width,
                                         height=video_stream.codec_context.height,
                                         codec=video_stream.codec_context.name,
                                         frame_count=video_stream.frames if video_stream.frames else None,
                                         fps=avg_fps,
                                         bit_rate=video_stream.bit_rate if video_stream.bit_rate else None,
                                         mime_type=self.MIME_TYPE)

        container.close()

        return self._metadata
