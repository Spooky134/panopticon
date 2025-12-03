from utils.frame_collector import FrameCollector

class FrameCollectorFactory:
    def create(self, session_id):
        return FrameCollector(session_id)