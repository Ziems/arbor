import threading
import zmq
import wandb

from typing import Any, Dict, Optional, TYPE_CHECKING
from arbor.server.services.comms.async_batch_requester import BatchResult

if TYPE_CHECKING:
    from arbor.server.services.scripts.arbor_grpo_trainer import ArborGRPOTrainer


class TrainerControlClient(threading.Thread):
    """Trainer-side ZeroMQ server that coordinates external batch producers.

    This thread runs inside the GRPO trainer process and exposes a REP socket
    that mirrors the interface provided by :class:`TrainerControlServer`. It
    receives requests such as status polling, inference lifecycle updates, batch
    submissions, checkpoint triggers, and terminate signals, then routes them to the
    underlying :class:`AsyncBatchRequester` and trainer control logic.
    """

    def __init__(self, trainer: "ArborGRPOTrainer", endpoint: str):
        super().__init__(daemon=True)
        self.trainer = trainer
        self.endpoint = endpoint
        self._stop_event = threading.Event()
        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._lock = threading.Lock()

    def run(self) -> None:  # pragma: no cover - network loop
        socket = self._ctx.socket(zmq.REP)
        self._socket = socket
        socket.bind(self.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            try:
                events = dict(poller.poll(timeout=100))
            except zmq.error.ZMQError:
                if self._stop_event.is_set():
                    break
                raise

            if socket in events and events[socket] == zmq.POLLIN:
                try:
                    message = socket.recv_json()
                    try:
                        response = self._handle(message)
                    except Exception as handler_exc:
                        response = {"ok": False, "error": str(handler_exc)}
                    socket.send_json(response)
                except Exception as exc:
                    # Best-effort reply to keep REP state consistent
                    try:
                        socket.send_json({"ok": False, "error": str(exc)})
                    except Exception:
                        pass

        try:
            socket.close(0)
        finally:
            self._socket = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            try:
                tmp = self._ctx.socket(zmq.REQ)
                tmp.connect(self.endpoint)
                tmp.send_json({"cmd": "noop"})
                tmp.recv_json()
                tmp.close(0)
            except Exception:
                pass

    def _handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        cmd = message.get("cmd")
        requester = self.trainer.async_requester

        if cmd == "status":
            return {
                "ok": True,
                "pending_ids": requester.get_pending_batch_ids(),
                "pending_count": requester.get_pending_count(),
                "completed_count": requester.get_completed_count(),
                "global_step": int(self.trainer.state.global_step),
                "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            }
        if cmd == "submit_batch":
            batch_payload = message.get("batch")
            if batch_payload is None:
                return {"ok": False, "error": "missing batch payload"}
            try:
                batch = BatchResult.model_validate(batch_payload)
                requester.submit_batch_result(batch)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True}
        if cmd == "checkpoint":
            checkpoint_name = message.get("checkpoint_name")
            try:
                checkpoint_dir = self.trainer.save_external_checkpoint(checkpoint_name)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "checkpoint_dir": checkpoint_dir}
        if cmd == "terminate":
            self.trainer.control.should_training_stop = True  # type: ignore[attr-defined]
            return {"ok": True}
        if cmd == "noop":
            return {"ok": True}
        return {"ok": False, "error": f"unknown command: {cmd}"}
