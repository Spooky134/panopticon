import asyncio

import av
from aiortc import VideoStreamTrack
from infrastructure.triton_proccessor.video_processor import VideoProcessor
from infrastructure.video.frame_collector import FrameCollector
from core.logger import get_logger


logger = get_logger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track: VideoStreamTrack, processor: VideoProcessor, collector:FrameCollector=None):
        super().__init__()
        self._track = track
        self._processor = processor
        self._collector = collector


    async def recv(self) -> av.VideoFrame:
        frame = await self._track.recv()
        if self._processor:
            frame = await self._processor.process_frame(frame)

        if self._collector:
            try:
                frame_copy = av.VideoFrame.from_ndarray(
                    frame.to_ndarray(format=frame.format.name),
                    format=frame.format.name
                )
                frame_copy.pts = frame.pts
                frame_copy.time_base = frame.time_base
                # frame_for_save = frame.reformat(format=frame.format)
                asyncio.create_task(self._collector.add_frame(frame_copy))
            except Exception as e:
                logger.error(f"session: {self._collector._streaming_session_id} - error adding frame to collector: {e}")

        return frame


