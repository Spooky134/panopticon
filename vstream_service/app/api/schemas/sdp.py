from pydantic import BaseModel

class SDPData(BaseModel):
    sdp: str
    type: str
