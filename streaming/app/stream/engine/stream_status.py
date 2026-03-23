from enum import Enum

class StreamStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    FINALIZING = "finalizing"