from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import zmq

from arbor.server.services.comms.async_batch_requester import BatchResult

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
        endpoint: str,
        *,
        context: Optional[zmq.Context] = None,
        recv_timeout: float | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._ctx = context or zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._recv_timeout_ms = int(recv_timeout * 1000) if recv_timeout else None

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
            raise RuntimeError("Control request failed") from exc

    def get_status(self) -> Dict[str, Any]:
        """Fetch trainer status, including pending batch ids."""
        return self._request({"cmd": "status"})

    def notify_inference_start(self, inference_id: str) -> Dict[str, Any]:
        return self._request({"cmd": "inference_start", "id": inference_id})

    def notify_inference_end(self, inference_id: str) -> Dict[str, Any]:
        return self._request({"cmd": "inference_end", "id": inference_id})

    def submit_batch(self, batch: BatchResult | Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(batch, BatchResult):
            payload = batch.model_dump()
        else:
            payload = dict(batch)
        return self._request({"cmd": "submit_batch", "batch": payload})

    def request_checkpoint(self) -> Dict[str, Any]:
        return self._request({"cmd": "checkpoint"})

    def request_stop(self) -> Dict[str, Any]:
        return self._request({"cmd": "stop"})

    def ping(self) -> Dict[str, Any]:
        return self._request({"cmd": "noop"})
