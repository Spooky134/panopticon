from fastapi import Depends

from utils.frame_collector import FrameCollector
from utils.frame_collector_factory import FrameCollectorFactory
from webrtc.connection_manager_factory import get_connection_manager
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager_factory import get_processor_manager
from grpc_client.processor_manager import ProcessorManager
from config import settings
from utils.streaming_session_manager import StreamingSessionManager


def get_streaming_session_manager(connection_manager: ConnectionManager = Depends(get_connection_manager),
                                  processor_manager: ProcessorManager = Depends(get_processor_manager),
                                  collector_factory: FrameCollectorFactory = Depends(FrameCollectorFactory),
                                  ice_servers: list = Depends(lambda: settings.ice_servers)
                                  ) -> StreamingSessionManager:
    if not hasattr(get_streaming_session_manager, 'instance'):
        get_streaming_session_manager.instance = StreamingSessionManager(connection_manager=connection_manager,
                                                                         processor_manager=processor_manager,
                                                                         collector_factory=collector_factory,
                                                                         ice_servers=ice_servers)
    return get_streaming_session_manager.instance

