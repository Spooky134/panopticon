from fastapi import Depends

from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from config import settings
from utils.streaming_session_manager import StreamingSessionManager


def get_streaming_session_manager(connection_manager: ConnectionManager = Depends(ConnectionManager),
                                  processor_manager: ProcessorManager = Depends(ProcessorManager),
                                  ice_servers: list = Depends(lambda: settings.ice_servers)) -> StreamingSessionManager:
    return StreamingSessionManager(connection_manager=connection_manager,
                                   processor_manager=processor_manager,
                                   ice_servers=ice_servers)

