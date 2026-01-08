from pydantic import BaseModel

class SDP(BaseModel):
    sdp: str
    type: str
