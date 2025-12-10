import uuid

from utils.frame_collector import FrameCollector

class FrameCollectorFactory:
    def create(self, session_id: uuid.UUID) -> FrameCollector:
        return FrameCollector(session_id)