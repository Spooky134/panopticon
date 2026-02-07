from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
from typing import Callable
from uuid import UUID

class ConnectionFactory:
    def __init__(self, ice_servers_factory: Callable):
        # self._rtc_config = RTCConfiguration(
        #     iceServers=[RTCIceServer(**server) for server in ice_servers]
        # )
        # self._rtc_config = RTCConfiguration(iceServers=ice_servers)
        self._ice_servers_factory = ice_servers_factory


    def create(self, session_id: UUID) -> RTCPeerConnection:
        rtc_config = RTCConfiguration(iceServers=self._ice_servers_factory(identifier=session_id))
        return RTCPeerConnection(configuration=rtc_config)
