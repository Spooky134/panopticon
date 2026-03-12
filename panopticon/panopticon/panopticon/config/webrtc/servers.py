from dataclasses import dataclass
import hmac
import hashlib
import base64
import time
from uuid import UUID


@dataclass(frozen=True)
class STUNServer:
    url: str

    def build_ice(self) -> dict:
        return {"urls": [self.url]}


@dataclass(frozen=True)
class Credentials:
    username: str
    credential: str


@dataclass(frozen=True)
class TURNServer:
    url: str
    shared_secret: str

    def _generate_credentials(self, identifier: UUID, expiration_time: int, origin: str) -> Credentials:
        # expiration_time = int(time.time()) + expiration_time

        username = f"{expiration_time}:{origin}:{identifier}"

        digester = hmac.new(
            self.shared_secret.encode(),
            username.encode(),
            hashlib.sha1
        )
        credential = base64.b64encode(digester.digest()).decode()

        return Credentials(username, credential)

    def build_ice(self, identifier: UUID, expiration_time: int, origin: str) -> dict:
        cred = self._generate_credentials(identifier, expiration_time, origin)

        return {
            'urls': [self.url],
            'username': cred.username,
            'credential': cred.credential
        }