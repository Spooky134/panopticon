from aiortc import VideoStreamTrack
import av
import cv2


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()  # Получаем кадр из исходного потока
        img = frame.to_ndarray(format="bgr24")  # Конвертируем в массив OpenCV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Применяем серый фильтр
        gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Преобразуем обратно в 3-канальное изображение
        new_frame = av.VideoFrame.from_ndarray(gray_colored, format="bgr24")  # Создаем новый кадр
        new_frame.pts = frame.pts  # Синхронизируем временные метки
        new_frame.time_base = frame.time_base  # Устанавливаем временную базу
        return new_frame  # Возвращаем обработанный кадр