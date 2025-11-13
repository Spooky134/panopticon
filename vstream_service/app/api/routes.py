from api.services.stream_service import StreamService
from api.services.stream_service_factory import get_stream_service
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials
from api.schemas.sdp import SDPData
from core.security.token import get_token

router = APIRouter(prefix="/stream", tags=["stream"])


@router.post("/offer")
async def offer(sdp_data: SDPData,
                token: HTTPAuthorizationCredentials = Depends(get_token),
                stream_service: StreamService = Depends(get_stream_service)):
    data = await stream_service.offer(token=token, sdp_data=sdp_data)

    return data