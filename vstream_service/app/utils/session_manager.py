import asyncio
import os

from aiortc import RTCConfiguration

from api.schemas.sdp import SDPData
from utils.session import Session
from utils.frame_collector import FrameCollector
from webrtc.video_transform_track import VideoTransformTrack
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from storage.s3_storage import S3Storage
from db.repositories import TestingSessionRepository, TestingVideoRepository
from core.logger import get_logger
from datetime import datetime


logger = get_logger(__name__)

#TODO добавить сохранение видео в бд
#TODO сохранение видео в таблицу с сессиями
#TODO попробовать перенести сохранение в сервис api
#TODO обьедиенение сервисов сохранения в один????
class SessionManager:
    def __init__(self,
                 connection_manager: ConnectionManager,
                 processor_manager: ProcessorManager,
                 ice_servers,
                 s3_storage: S3Storage,
                 testing_session_repository: TestingSessionRepository = None,
                 testing_video_repository: TestingVideoRepository = None,
                 ):
        self.connection_manager = connection_manager
        self.processor_manager = processor_manager
        self.s3_storage = s3_storage
        self.ice_servers = ice_servers
        self.testing_session_repository = testing_session_repository
        self.testing_video_repository = testing_video_repository
        self.sessions: dict[str, Session] = {}

    def _register_event_handlers(self, session: Session):
        peer_connection = session.peer_connection
        session_id = session.session_id

        @peer_connection.on("track")
        async def on_track(track):
            logger.info(f"session: {session_id} - Track received: {track.kind}")
            if track.kind == "video":
                transformed = VideoTransformTrack(track, session.grpc_processor, session.collector)
                peer_connection.addTrack(transformed)

        @peer_connection.on("iceconnectionstatechange")
        async def on_ice_state_change():
            state = peer_connection.iceConnectionState
            logger.info(f"session: {session_id} - ICE state → {state}")

            if state in ["failed", "closed", "disconnected"]:
                logger.info(f"session: {session_id} - ICE state → {state}")
                await self._dispose_session(session_id)

    async def initiate_session(self, user_id:str, session_id: str, sdp_data: SDPData) -> dict:
        if session_id in self.sessions:
            await self._dispose_session(session_id)

        peer_connection = await self.connection_manager.create_connection(
            session_id=session_id,
            rtc_config=RTCConfiguration(iceServers=self.ice_servers)
        )
        grpc_processor = await self.processor_manager.create_processor(
            session_id=session_id
        )
        collector = FrameCollector(
            session_id=session_id)

        session = Session(
            session_id=session_id,
            user_id=user_id,
            peer_connection=peer_connection,
            video_processor=grpc_processor,
            collector=collector
        )

        self._register_event_handlers(session)

        answer = await session.start(sdp_data=sdp_data)

        self.sessions[session_id] = session
        logger.info(f"session: {session_id} - Created for user {user_id}")

        testing_session = await self.testing_session_repository.update(session_id=session_id,
                                                                       data={"status": "running",
                                                                             "started_at": self.sessions.get(session_id).started_at})
        return answer

    async def _dispose_session(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return

        logger.info(f"session: {session_id} - Cleaning up")

        try:
            await session.finalize()
        except Exception as e:
            logger.error(f"session: {session_id} - Finalize error: {e}")

        await self.save_session_result(session_id=session_id)

        await asyncio.gather(
            self.processor_manager.close_processor(session_id),
            self.connection_manager.close_connection(session_id),
            return_exceptions=True,
        )

        self.sessions.pop(session_id, None)
        logger.info(f"session: {session_id} - Cleaned up")

    async def save_session_result(self, session_id: str):
        session = self.sessions.get(session_id)
        upload_prefix = "videos/"
        object_name = f"{upload_prefix}{session_id}.mp4"

        # TODO прверка существования collector
        # TODO проверка существования файла на выходе
        output_file = session.collector.output_file
        # TODO удаление видео


        try:
            await self.s3_storage.ensure_bucket()
            logger.info(f"session: {session_id} - Loading {output_file} → {object_name}")
            await self.s3_storage.upload_file(output_file, object_name)
            os.remove(output_file)
            logger.info(f"session: {session_id} - The video has been successfully uploaded to S3: {object_name}")
        except Exception as e:
            logger.error(f"session: {session_id} - Error loading in: {e}")

        data = {
            "testing_session_id": session_id,
            "s3_key": object_name,
            "s3_bucket": self.s3_storage.bucket_name,
            "duration": session.collector.metadata.get("duration"),
            "file_size": session.collector.metadata.get("file_size"),
            "mime_type": session.collector.metadata.get("mime_type"),
            "created_at": datetime.now()
        }

        await self.testing_video_repository.create(data=data)

        await self.testing_session_repository.update(session_id, {
            "status": "finished",
            "ended_at": self.sessions.get(session_id).finished_at,
        })

    async def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    async def dispose_all_sessions(self):
        for session_id in self.sessions.keys():
            await self._dispose_session(session_id=session_id)