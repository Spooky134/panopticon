from enum import Enum

class ProcessorType(str, Enum):
    CALIBRATION = "calibration"
    MONITORING = "monitoring"