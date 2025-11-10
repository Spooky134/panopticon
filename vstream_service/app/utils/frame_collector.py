import av
import asyncio
import os
from datetime import datetime

from storage.s3_storage import S3Storage


#TODO контекстный менеджер или функция с конеткстом
class FrameCollector:
    def __init__(self, session_id: str, s3_storage: S3Storage, upload_prefix="videos/"):
        self.session_id = session_id
        self.s3_storage = s3_storage
        self.upload_prefix = upload_prefix
        self.frames = []
        self.start_time = datetime.now()
        self.output_file = f"/tmp/{session_id}.mp4"
        self._lock = asyncio.Lock()

    async def add_frame(self, frame):
        try:
            async with self._lock:
                self.frames.append(frame.to_ndarray(format="bgr24"))
        except Exception as e:
            print(f"[FrameCollector:{self.session_id}] Error adding frame: {e}")

    async def finalize(self):
        print(f"[FrameCollector:{self.session_id}] We are starting finalization of frames={len(self.frames)}")
        if not self.frames:
            print(f"[FrameCollector:{self.session_id}] There are no frames to save.")
            return

        print(f"[{self.session_id}] Save {len(self.frames)} frames to {self.output_file}")
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
            print(f"[FrameCollector:{self.session_id}] The video is saved locally: {self.output_file}")
        except Exception as e:
            print(f"[FrameCollector:{self.session_id}] Error while compiling video: {e}")
            return
        try:
            object_name = f"{self.upload_prefix}{self.session_id}.mp4"

            await self.s3_storage.ensure_bucket()
            print(f"[FrameCollector:{self.session_id}] Loading {self.output_file} → {object_name}")
            await self.s3_storage.upload_file(self.output_file, object_name)
            os.remove(self.output_file)
            print(f"[FrameCollector:{self.session_id}] The video has been successfully uploaded to S3: {object_name}")
        except Exception as e:
            print(f"[FrameCollector:{self.session_id}] Error loading in: {e}")
