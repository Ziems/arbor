from arbor.server.core.config import Settings
from arbor.server.services.jobs.job import Job


class InferenceJob(Job):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.process: Optional[subprocess.Popen] = None
        self.launch_kwargs = {}
        self.last_activity = None
        self._shutting_down = False
        self.launched_model: Optional[str] = None
        self.inference_count = 0
        self._session = None
        self.port: Optional[int] = None
        self.group_port = None
        self.vllm_client = None
        self._is_updating = 0  # Counter for weight updates in progress

    def is_server_running(self) -> bool:
        """Check if vLLM server is running."""
        return self.process is not None and self.process.poll() is None
