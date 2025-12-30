from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

class ConnectionFactory:
    def __init__(self, ice_servers: list):
        # self._rtc_config = RTCConfiguration(
        #     iceServers=[RTCIceServer(**server) for server in ice_servers]
        # )
        self._rtc_config = RTCConfiguration(iceServers=ice_servers)


    def create(self) -> RTCPeerConnection:
        return RTCPeerConnection(configuration=self._rtc_config)
