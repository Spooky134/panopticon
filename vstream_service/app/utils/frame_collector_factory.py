import uuid

from utils.frame_collector import FrameCollector


#TODO настройки для коллектора
class FrameCollectorFactory:
    def __init__(self):
        pass

    def create(self, session_id: uuid.UUID) -> FrameCollector:
        return FrameCollector(session_id)