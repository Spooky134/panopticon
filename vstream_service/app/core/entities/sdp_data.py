from dataclasses import dataclass

@dataclass(frozen=True)
class SDPData:
    sdp: str
    type: str
