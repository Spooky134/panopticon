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

        self._temp_dir = f"/tmp/collected_data/"
        os.makedirs(self._temp_dir, exist_ok=True)
        self._file_name = f"{session_id}.mp4"
        self._output_file_path = os.path.join(self._temp_dir, self._file_name)

        self._lock = asyncio.Lock()
        self._metadata = None

    @property
    def file_name(self):
        return self._file_name

    @property
    def output_file_path(self):
        return self._output_file_path

    def file_exists(self):
        return os.path.exists(self._output_file_path)

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

        logger.info(f"session: {self.session_id} - Save {len(self.frames)} frames to {self._output_file_path}")
        try:
            container = av.open(self._output_file_path, mode="w")
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

            logger.info(f"session: {self.session_id} - the video is saved locally: {self._output_file_path}")
        except Exception as e:
            logger.error(f"session: {self.session_id} - error while compiling video: {e}")

        return self._output_file_path

    async def cleanup(self):
        logger.info(f"session: {self.session_id} - collector cleaning up")
        try:
            if os.path.exists(self._output_file_path):
                os.remove(self._output_file_path)
            logger.info(f"session: {self.session_id} - temporary file removed")
        except Exception as e:
            logger.error(f"session: {self.session_id} - error removing temporary file: {e}")


    async def get_metadata(self):
        if self._metadata:
            return self._metadata

        if not os.path.exists(self._output_file_path):
            logger.warning(f"session: {self.session_id} - no metadata file found, creating new one")
            return None

        try:
            container = av.open(self._output_file_path, mode="r")
        except Exception as e:
            logger.error(f"session: {self.session_id} - failed to open video for metadata: {e}")
            return None
        video_stream = next((s for s in container.streams if s.type == "video"), None)

        if not video_stream:
            logger.error(f"session: {self.session_id} - no video stream found in file")
            return None

        if video_stream.duration is not None and video_stream.time_base is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            if video_stream.average_rate:
                duration = float(video_stream.frames) / float(video_stream.average_rate)
            else:
                duration = None

        file_size = os.path.getsize(self.output_file_path)

        mime_type = "video/mp4"

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
            "mime_type": mime_type
        }
        container.close()

        return self._metadata