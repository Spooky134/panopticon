from config import settings
from webrtc.connection_factory import ConnectionFactory


def get_connection_factory() -> ConnectionFactory:
    return ConnectionFactory(ice_servers=settings.ice_servers)