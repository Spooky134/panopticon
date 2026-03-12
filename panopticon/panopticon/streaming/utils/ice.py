
from django.conf import settings
import time
from uuid import UUID


def get_ice_servers(identifier: UUID, ttl: int=24*3600, origin: str = "client") -> list[dict]:
    # STUN
    ice_servers = [server.build_ice() for server in settings.STUN_SERVERS]

    expiration_time = int(time.time()) + ttl
    # TURN
    ice_servers += [
        server.build_ice(identifier, expiration_time, origin) 
        for server in settings.TURN_SERVERS
    ]

    return ice_servers