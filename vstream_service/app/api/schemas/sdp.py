from pydantic import BaseModel, ConfigDict

class SDP(BaseModel):
    sdp: str
    type: str

    model_config = ConfigDict(from_attributes=True)