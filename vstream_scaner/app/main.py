import asyncio
from fastapi import FastAPI, BackgroundTasks, Request
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import cv2
from fastapi.responses import HTMLResponse
import av
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates

app = FastAPI()
pcs = set()
templates = Jinja2Templates(directory="./")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

@app.post("/offer")
async def offer(sdp_data: dict, background_tasks: BackgroundTasks):
    offer = RTCSessionDescription(sdp_data["sdp"], sdp_data["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print(f"Получен трек: {track.kind}")
        if track.kind == "video":
            transformed_track = VideoTransformTrack(track)  # Создаем измененный поток
            pc.addTrack(transformed_track)  # Добавляем его в соединение

    @pc.on("iceconnectionstatechange")
    def on_ice_state_change():
        print(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "closed", "disconnected"]:
            background_tasks.add_task(close_peer_connection, pc)

    await pc.setRemoteDescription(offer)
    
    # Ждем сбора ICE-кандидатов
    await asyncio.sleep(1)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def close_peer_connection(pc: RTCPeerConnection):
    if pc in pcs:
        pcs.discard(pc)
    await pc.close()
    print("PeerConnection закрыт")
