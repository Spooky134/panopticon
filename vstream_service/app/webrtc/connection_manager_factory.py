from webrtc.connection_manager import ConnectionManager


def get_connection_manager():
    return ConnectionManager(max_connections=1000)