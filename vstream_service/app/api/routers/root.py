from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from config import settings


router = APIRouter(tags=["root"])

templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
                                                     # "host": settings.EXTERNAL_IP,
                                                     # "port": settings.TURN_PORT})