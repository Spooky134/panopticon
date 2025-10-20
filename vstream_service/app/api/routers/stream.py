from fastapi import APIRouter, BackgroundTasks, Depends
from services import StreamService
from dependencies import service_factory

router = APIRouter(prefix="/stream", tags=["stream"])


@router.post("/offer")
async def offer(sdp_data: dict,
                background_tasks: BackgroundTasks,
                stream_service: StreamService = Depends(service_factory(StreamService))):
    data  = await stream_service.offer(sdp_data=sdp_data, background_tasks=background_tasks)
    
    return data