from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


def get_free_port() -> int:
    """
    Return a randomly selected free TCP port on localhost from a selection of 3-4 ports.
    """
    import random
    import socket

    ports = []
    for _ in range(random.randint(5, 10)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 0))
                ports.append(s.getsockname()[1])
        except Exception as e:
            logger.error(f"Error binding to port: {e}")
    return random.choice(ports)


def strip_prefix(model: str) -> str:
    prefixes = ["openai/", "huggingface/", "local:", "arbor:"]
    for prefix in prefixes:
        if model.startswith(prefix):
            model = model[len(prefix) :]
    return model
