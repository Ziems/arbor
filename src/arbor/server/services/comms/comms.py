import socket
import json
import time
from typing import Optional

def get_free_port() -> int:
    """
    Return a free TCP port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]

class ArborServerCommsHandler:
    """Handles socket communication between manager and training process"""
    def __init__(self, host="localhost", command_port=None, status_port=None, data_port=None):
        self.host = host
        self.command_port = command_port or get_free_port()
        self.status_port = status_port or get_free_port()
        self.data_port = data_port or get_free_port()

        # Initialize sockets
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Allow reuse of address
        self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.status_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind sockets
        self.command_socket.bind((host, command_port))
        self.status_socket.bind((host, status_port))
        self.data_socket.bind((host, data_port))

        # Listen for connections
        self.command_socket.listen(1)
        self.status_socket.listen(1)
        self.data_socket.listen(1)

        self.connections = {}
        self.running = True

    def accept_connections_loop(self):
        """Continuously accept connections from training process"""
        print("Starting connection acceptance loop...")
        while self.running:
            try:
                if 'command' not in self.connections or self.connections['command'].fileno() == -1:
                    print("Waiting for command connection...")
                    self.connections['command'], _ = self.command_socket.accept()
                    print("Command connection established")

                if 'status' not in self.connections or self.connections['status'].fileno() == -1:
                    print("Waiting for status connection...")
                    self.connections['status'], _ = self.status_socket.accept()
                    print("Status connection established")

                if 'data' not in self.connections or self.connections['data'].fileno() == -1:
                    print("Waiting for data connection...")
                    self.connections['data'], _ = self.data_socket.accept()
                    print("Data connection established")

                # All connections established, sleep briefly before checking again
                time.sleep(1)

            except socket.error as e:
                print(f"Socket error: {e}")
                # Remove dead connections
                for key in list(self.connections.keys()):
                    try:
                        if self.connections[key].fileno() == -1:
                            del self.connections[key]
                    except socket.error:
                        del self.connections[key]
                time.sleep(1)  # Wait before retrying

    def send_command(self, command: dict):
        """Send command to training process"""
        if 'command' not in self.connections:
            print("No command connection available")
            return
        try:
            msg = json.dumps(command) + "\n"
            self.connections['command'].sendall(msg.encode())
        except socket.error as e:
            print(f"Error sending command: {e}")
            del self.connections['command']

    def send_data(self, data: dict):
        """Send training data to training process"""
        if 'data' not in self.connections:
            print("No data connection available")
            return
        try:
            msg = json.dumps(data) + "\n"
            self.connections['data'].sendall(msg.encode())
        except socket.error as e:
            print(f"Error sending data: {e}")
            del self.connections['data']

    def receive_status(self):
        """Receive status updates from training process"""
        buffer = ""
        while self.running:
            if 'status' not in self.connections:
                print("No status connection available")
                time.sleep(1)
                continue

            try:
                chunk = self.connections['status'].recv(4096).decode()
                if not chunk:
                    del self.connections['status']
                    continue

                buffer += chunk
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    try:
                        yield json.loads(message)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON received: {message}")
            except socket.error as e:
                print(f"Error receiving status: {e}")
                del self.connections['status']
                time.sleep(1)

    def close(self):
        """Close all connections"""
        self.running = False
        for conn in self.connections.values():
            try:
                conn.close()
            except socket.error:
                pass
        self.command_socket.close()
        self.status_socket.close()
        self.data_socket.close()

class ArborScriptCommsHandler:
    def __init__(self, host: str, command_port: int, status_port: int, data_port: int):
        # Connect to manager's sockets
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to ports
        self.command_socket.connect((host, command_port))
        self.status_socket.connect((host, status_port))
        self.data_socket.connect((host, data_port))

        self.running = True

    def receive_command(self):
        """Receive command from manager"""
        buffer = ""
        while self.running:
            chunk = self.command_socket.recv(4096).decode()
            if not chunk:
                break
            buffer += chunk
            while "\n" in buffer:
                message, buffer = buffer.split("\n", 1)
                try:
                    yield json.loads(message)
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")

    def receive_data(self) -> Optional[list]:
        """Receive training data from manager"""
        buffer = ""
        try:
            # Read data in chunks until we get a complete JSON object
            while True:
                chunk = self.data_socket.recv(4096).decode()
                if not chunk:  # Connection closed
                    return None

                buffer += chunk

                # Look for complete JSON objects by counting brackets
                bracket_count = 0
                in_string = False
                escape_next = False

                for i, char in enumerate(buffer):
                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1

                            # Found a complete JSON array
                            if bracket_count == 0:
                                try:
                                    complete_json = buffer[:i+1]
                                    data = json.loads(complete_json)
                                    buffer = buffer[i+1:]  # Keep remainder for next time
                                    return data
                                except json.JSONDecodeError:
                                    continue  # Not valid JSON yet, keep reading

                # If we get here, we need more data to complete the JSON object
                continue

        except Exception as e:
            print(f"Error receiving data: {e}")
            return None

    def send_status(self, status: dict):
        """Send status update to manager"""
        msg = json.dumps(status) + "\n"
        self.status_socket.sendall(msg.encode())

    def close(self):
        """Close all connections"""
        self.running = False
        self.command_socket.close()
        self.status_socket.close()
        self.data_socket.close()

