from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional
import threading
import time

import zmq

from arbor.server.services.comms.async_batch_requester import BatchResult
from arbor.server.utils.helpers import get_free_port

LOGGER = logging.getLogger(__name__)


class TrainerControlServer:
    """Server-side helper that talks to the trainer endpoint.

    External batch producers can use this class to coordinate work with the
    trainer by polling for status, submitting batch results, and reporting
    inference lifecycle events. It is the counterpart to
    ``TrainerControlClient`` which runs inside the trainer process.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        context: Optional[zmq.Context] = None,
        recv_timeout: float | None = None,
        send_timeout: float | None = 2.0,
    ) -> None:
        """Initialize the control server client."""
        self.port = get_free_port()
        self.endpoint = f"tcp://{host}:{self.port}"
        self._ctx = context or zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._recv_timeout_ms = int(recv_timeout * 1000) if recv_timeout else None
        self._send_timeout_ms = int(send_timeout * 1000) if send_timeout else None
        # Serialize REQ send/recv to avoid EFSM with concurrent callers
        self._io_lock = threading.Lock()

    def __enter__(self) -> "TrainerControlServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Create and connect the underlying ZeroMQ REQ socket."""
        if self._socket is not None:
            return
        socket = self._ctx.socket(zmq.REQ)
        if self._recv_timeout_ms is not None:
            socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        if self._send_timeout_ms is not None:
            socket.setsockopt(zmq.SNDTIMEO, self._send_timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket

    def close(self) -> None:
        """Close the REQ socket."""
        if self._socket is None:
            return
        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close(0)
        finally:
            self._socket = None

    def _ensure_socket(self) -> zmq.Socket:
        if self._socket is None:
            raise RuntimeError("TrainerControlServer socket is not connected")
        return self._socket

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._io_lock:
            socket = self._ensure_socket()
            try:
                socket.send_json(payload)
                return socket.recv_json()
            except zmq.error.Again as exc:
                raise TimeoutError(
                    f"Timed out waiting for response to command {payload.get('cmd')}"
                ) from exc
            except Exception as exc:  # pragma: no cover - transport errors
                LOGGER.exception("TrainerControlServer request failed")
                LOGGER.error(f"Request failed: {payload}")
                import traceback

                LOGGER.error(
                    f"Full traceback: {''.join(traceback.format_tb(exc.__traceback__))}"
                )
                raise RuntimeError("Control request failed") from exc

    def get_status(self) -> Dict[str, Any]:
        """Fetch trainer status, including pending batch ids."""
        resp = self._request({"cmd": "status"})
        if not resp.get("ok", False):
            raise RuntimeError(
                f"Error getting status: {resp.get('error', 'Unknown error')}"
            )
        return resp

    def submit_batch(self, batch: BatchResult | Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(batch, BatchResult):
            payload = batch.model_dump()
        else:
            payload = dict(batch)
        resp = self._request({"cmd": "submit_batch", "batch": payload})
        if not resp.get("ok", False):
            raise RuntimeError(
                f"Error submitting batch: {resp.get('error', 'Unknown error')}"
            )
        return resp

    def request_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        resp = self._request({"cmd": "checkpoint", "checkpoint_name": checkpoint_name})
        if not resp.get("ok", False):
            raise RuntimeError(
                f"Error requesting checkpoint: {resp.get('error', 'Unknown error')}"
            )
        return resp

    def request_terminate(self) -> Dict[str, Any]:
        resp = self._request({"cmd": "terminate"})
        if not resp.get("ok", False):
            raise RuntimeError(
                f"Error requesting terminate: {resp.get('error', 'Unknown error')}"
            )
        return resp

    def ping(self) -> Dict[str, Any]:
        resp = self._request({"cmd": "noop"})
        if not resp.get("ok", False):
            raise RuntimeError(f"Error pinging: {resp.get('error', 'Unknown error')}")
        return resp

    def wait_for_clients(
        self,
        expected_count: int = 1,
        *,
        timeout: float | None = 30.0,
        interval: float = 1,
        log_every: int = 10,
    ) -> None:
        """Wait until the trainer control client responds to ping.

        This sends NOOP requests and counts successful responses until
        ``expected_count`` successes are observed. Because the trainer exposes
        a single REP socket, this does not distinguish unique clients; it
        merely ensures the endpoint is responsive repeatedly.
        """
        successes = 0
        attempts = 0
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            if successes >= expected_count:
                return
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for {expected_count} client response(s) on {self.endpoint}"
                )

            try:
                resp = self._request({"cmd": "noop"})
                if resp.get("ok", False):
                    successes += 1
                else:
                    successes = 0
            except Exception:
                successes = 0

            attempts += 1
            if attempts % log_every == 0 and successes < expected_count:
                LOGGER.info(
                    "Waiting for trainer control client (successes=%d/%d)",
                    successes,
                    expected_count,
                )

            time.sleep(interval)
