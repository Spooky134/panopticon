from dataclasses import dataclass


@dataclass(frozen=True)
class SDPEntity:
    sdp: str
    type: str
