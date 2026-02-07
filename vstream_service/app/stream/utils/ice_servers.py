from dataclasses import dataclass
import hmac
import hashlib
import base64
import time
from config.settings import settings
from uuid import UUID
from aiortc import RTCIceServer

@dataclass(frozen=True)
class Credentials:
    username: str
    credential: str

def get_turn_credentials(identifier: UUID, ttl: int, origin: str) -> Credentials:
    expiration_time = int(time.time()) + ttl

    username = f"{expiration_time}:{origin}:{identifier}"

    digester = hmac.new(
        settings.TURN_SHARED_SECRET.encode(),
        username.encode(),
        hashlib.sha1
    )
    credential = base64.b64encode(digester.digest()).decode()

    return Credentials(
        username=username,
        credential=credential
    )

def get_ice_servers(identifier: UUID, ttl: int=24*3600, origin: str = settings.VSTREAM_SERVICE_NAME) -> list[RTCIceServer]:
    cred = get_turn_credentials(identifier, ttl, origin)

    ice_servers = [RTCIceServer(urls=[url]) for url in settings.STUN_SERVERS]
    ice_servers += [
        RTCIceServer(urls=[url], username=cred.username, credential=cred.credential)
        for url in settings.TURN_SERVERS
    ]

    return ice_servers