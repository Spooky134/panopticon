# video/frame_collector.py
import av
import asyncio
import os
from datetime import datetime


class FrameCollector:
    def __init__(self, session_id: str, s3_storage, upload_prefix="videos/"):
        self.session_id = session_id
        self.s3_storage = s3_storage
        self.upload_prefix = upload_prefix
        self.frames = []
        self.start_time = datetime.utcnow()
        self.output_file = f"/tmp/{session_id}.mp4"
        self._lock = asyncio.Lock()

    async def add_frame(self, frame):
        try:
            async with self._lock:
                self.frames.append(frame.to_ndarray(format="bgr24"))
        except Exception as e:
            print(f"[FrameCollector:{self.session_id}] Ошибка при добавлении кадра: {e}")


    async def finalize(self):
        print(f"[FrameCollector:{self.session_id}] Начинаем финализацию, кадров={len(self.frames)}")
        """Сохраняет накопленные кадры в видео и загружает в S3"""
        if not self.frames:
            print(f"[FrameCollector:{self.session_id}] Нет кадров для сохранения.")
            return

        print(f"[{self.session_id}] Сохраняем {len(self.frames)} кадров в {self.output_file}")
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
            print(f"[FrameCollector:{self.session_id}] Видео сохранено локально: {self.output_file}")
        except Exception as e:
            print(f"[FrameCollector:{self.session_id}] Ошибка при сборке видео: {e}")
            return
        try:
            object_name = f"{self.upload_prefix}{self.session_id}.mp4"

            await self.s3_storage.ensure_bucket()
            print(f"[FrameCollector:{self.session_id}] Загружаем {self.output_file} → {object_name}")
            await self.s3_storage.upload_file(self.output_file, object_name)
            os.remove(self.output_file)
            print(f"[FrameCollector:{self.session_id}] Видео успешно загружено в S3: {object_name}")
        except Exception as e:
            print(f"[FrameCollector:{self.session_id}] Ошибка при загрузке в S3: {e}")
