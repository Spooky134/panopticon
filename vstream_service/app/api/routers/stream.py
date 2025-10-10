from fastapi import APIRouter, BackgroundTasks, Depends
from services import StreamService

router = APIRouter(prefix="/stream", tags=["stream"])

stream_service: StreamService = StreamService()

@router.post("/offer")
async def offer(sdp_data: dict, background_tasks: BackgroundTasks):
    data  = await stream_service.offer(sdp_data=sdp_data, background_tasks=background_tasks)
    
    return data