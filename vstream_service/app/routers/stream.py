from services.stream_service import StreamService
from core.dependencies.service_factory import stream_service_factory
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials
from schemas.sdp import SDPData
from core.dependencies.token import get_token

router = APIRouter(prefix="/stream", tags=["stream"])




@router.post("/offer")
async def offer(sdp_data: SDPData,
                background_tasks: BackgroundTasks,
                token: HTTPAuthorizationCredentials = Depends(get_token),
                stream_service: StreamService = Depends(stream_service_factory(StreamService))):
    data = await stream_service.offer(sdp_data=sdp_data, background_tasks=background_tasks, token=token)

    return data


# @router.post("/offer")
# async def offer(request: Request,
#                 sdp_data: dict,
#                 background_tasks: BackgroundTasks,
#                 stream_service: StreamService = Depends(stream_service_factory(StreamService))):
#     data  = await stream_service.offer(sdp_data=sdp_data, background_tasks=background_tasks, request=request)
#
#     return data