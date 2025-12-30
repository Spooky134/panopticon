from fastapi import Depends

from utils.streaming_session_manager import StreamingSessionManager

from utils.frame_collector_factory import FrameCollectorFactory
from utils.get_frame_collector_factory import get_frame_collector_factory

from grpc_client.processor_factory import VideoProcessorFactory
from grpc_client.get_processor_factory import get_processor_factory

from webrtc.connection_factory import ConnectionFactory
from webrtc.get_connection_factory import get_connection_factory


def get_streaming_session_manager(connection_factory: ConnectionFactory = Depends(get_connection_factory),
                                  processor_factory: VideoProcessorFactory = Depends(get_processor_factory),
                                  collector_factory: FrameCollectorFactory = Depends(get_frame_collector_factory),
                                  ) -> StreamingSessionManager:
    if not hasattr(get_streaming_session_manager, 'instance'):
        get_streaming_session_manager.instance = StreamingSessionManager(connection_factory=connection_factory,
                                                                         processor_factory=processor_factory,
                                                                         collector_factory=collector_factory,
                                                                         max_sessions=1000)
    return get_streaming_session_manager.instance

