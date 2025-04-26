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
    def __init__(self, host, command_port, status_port, data_port, broadcast_port):
        self.context = zmq.Context()

        # Command socket
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.connect(f"tcp://{host}:{command_port}")

        # Status socket
        self.status_socket = self.context.socket(zmq.PUB)
        self.status_socket.connect(f"tcp://{host}:{status_port}")

        # Data socket
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.connect(f"tcp://{host}:{data_port}")

        # Broadcast socket
        self.broadcast_socket = self.context.socket(zmq.SUB)
        self.broadcast_socket.connect(f"tcp://{host}:{broadcast_port}")

    #     self.data_queue = queue.Queue()
    #     self._start_data_receiver()

    # def _start_data_receiver(self):
    #     def receiver():
    #         while True:
    #             try:
    #                 data = self.data_socket.recv_json()
    #                 self.data_queue.put(data)
    #             except Exception as e:
    #                 print(f"Error receiving data: {e}")
    #                 break

    #     self.receiver_thread = threading.Thread(target=receiver, daemon=True)
    #     self.receiver_thread.start()

    def send_status(self, status):
        self.status_socket.send_json(status)

    def receive_command(self):
        while True:
            command = self.command_socket.recv_json()
            # Send acknowledgment
            self.command_socket.send_json({"status": "received"})
            yield command

    def receive_data(self):
        # return self.data_queue.get()
        return self.data_socket.recv_json()

    # def has_data(self):
    #     return not self.data_queue.empty()

    def receive_broadcast(self):
        while True:
            broadcast = self.broadcast_socket.recv_json()
            yield broadcast

    def close(self):
        self.command_socket.close()
        self.status_socket.close()
        self.data_socket.close()
        self.broadcast_socket.close()
        self.context.term()
