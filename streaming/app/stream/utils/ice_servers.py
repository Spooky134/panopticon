import time
from typing import List
from app.config.settings import settings
from uuid import UUID
from aiortc import RTCIceServer


def get_ice_servers(identifier: UUID, ttl: int=24*3600, origin: str = "client") -> List[RTCIceServer]:
    # STUN
    ice_servers = [server.build_ice() for server in settings.stun_servers]

    expiration_time = int(time.time()) + ttl
    # TURN
    ice_servers += [
        server.build_ice(identifier, expiration_time, origin) 
        for server in settings.turn_servers
    ]

    return ice_servers