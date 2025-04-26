import zmq
import queue
import threading
import time


class ArborServerCommsHandler:
    """Handles socket communication between manager and training process"""

    def __init__(self, host="localhost"):
        self.host = host
        self.context = zmq.Context()

        # Command socket (REQ/REP pattern)
        self.command_socket = self.context.socket(zmq.REP)
        self.command_port = self.command_socket.bind_to_random_port(f"tcp://{host}")

        # Status socket (PUB/SUB pattern)
        self.status_socket = self.context.socket(zmq.SUB)
        self.status_port = self.status_socket.bind_to_random_port(f"tcp://{host}")

        # Data socket (PUSH/PULL pattern)
        self.data_socket = self.context.socket(zmq.PUSH)
        self.data_port = self.data_socket.bind_to_random_port(f"tcp://{host}")

        self.broadcast_socket = self.context.socket(zmq.PUB)
        self.broadcast_port = self.broadcast_socket.bind_to_random_port(f"tcp://{host}")

    def receive_status(self):
        while True:
            status = self.status_socket.recv_json()
            yield status

    def send_command(self, command):
        self.command_socket.send_json(command)
        return self.command_socket.recv_json()  # Wait for acknowledgment

    def send_data(self, data):
        self.data_socket.send_json(data)

    def send_broadcast(self, message):
        self.broadcast_socket.send_json({"message": message})

    def close(self):
        self.command_socket.close()
        self.status_socket.close()
        self.data_socket.close()
        self.broadcast_socket.close()
        self.context.term()


class ArborScriptCommsHandler:
    def __init__(
        self,
        host,
        command_port,
        status_port,
        data_port,
        broadcast_port,
        is_main_process,
    ):
        self.context = zmq.Context()
        self.is_main_process = is_main_process

        # Command socket (main process only)
        if is_main_process:
            self.command_socket = self.context.socket(zmq.REQ)
            self.command_socket.connect(f"tcp://{host}:{command_port}")

            self.status_socket = self.context.socket(zmq.PUB)
            self.status_socket.connect(f"tcp://{host}:{status_port}")
        else:
            self.command_socket = None
            self.status_socket = None

        # Data socket (all processes)
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.connect(f"tcp://{host}:{data_port}")

        # Broadcast socket (all processes)
        self.broadcast_socket = self.context.socket(zmq.SUB)
        self.broadcast_socket.connect(f"tcp://{host}:{broadcast_port}")
        self.broadcast_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def send_status(self, status):
        if self.status_socket is not None:
            self.status_socket.send_json(status)

    def receive_command(self):
        if self.command_socket is not None:
            while True:
                command = self.command_socket.recv_json()
                # Send acknowledgment
                self.command_socket.send_json({"status": "received"})
                yield command

    def receive_data(self):
        # return self.data_queue.get()
        return self.data_socket.recv_json()

    def receive_broadcast(self):
        while True:
            broadcast = self.broadcast_socket.recv_json()
            yield broadcast

    def close(self):
        if self.command_socket is not None:
            self.command_socket.close()
        if self.status_socket is not None:
            self.status_socket.close()
        self.data_socket.close()
        self.broadcast_socket.close()
        self.context.term()
