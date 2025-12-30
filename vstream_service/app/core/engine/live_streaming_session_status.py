from enum import Enum

class LiveStreamingSessionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    FINALIZING = "finalizing"