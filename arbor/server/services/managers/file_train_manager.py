from arbor.server.api.models.schemas import FineTuneRequest
from arbor.server.core.config import Config
from arbor.server.services.jobs.file_train_job import FileTrainJob
from arbor.server.services.jobs.job import JobEvent
from arbor.server.services.managers.base_manager import BaseManager
from arbor.server.services.managers.file_manager import FileManager
from arbor.server.services.managers.job_manager import JobStatus


class FileTrainManager(BaseManager):
    def __init__(self, config: Config, gpu_manager=None):
        super().__init__(config)
        self.gpu_manager = gpu_manager

    def fine_tune(
        self, request: FineTuneRequest, job: FileTrainJob, file_manager: FileManager
    ):
        job.status = JobStatus.RUNNING
        job.add_event(
            JobEvent(level="info", message="Starting fine-tuning job", data={})
        )

        # Determine fine-tuning type from method or auto-detect from file format
        if request.method is not None:
            # cast it to sft if it's supervised
            if request.method["type"] == "supervised":
                request.method["type"] = "sft"

            fine_tune_type = request.method["type"]
            job.add_event(
                JobEvent(
                    level="info",
                    message=f"Using specified training method: {fine_tune_type}",
                    data={},
                )
            )
        else:
            # Auto-detect based on file format
            detected_format = file_manager.check_file_format(request.training_file)
            if detected_format == "unknown":
                raise ValueError(
                    "Could not determine training method. File format is unknown. "
                    "Please specify the method parameter with type 'sft' or 'dpo'."
                )
            fine_tune_type = detected_format
            job.add_event(
                JobEvent(
                    level="info",
                    message=f"Auto-detected training method: {fine_tune_type}",
                    data={},
                )
            )

        if fine_tune_type not in ["dpo", "sft"]:
            raise ValueError(
                f"Unsupported training method: {fine_tune_type}. Supported methods: 'sft', 'dpo'"
            )

        job.fine_tune(request, file_manager, fine_tune_type)

    def cleanup(self) -> None:
        """Clean up FileTrainManager resources"""
        if self._cleanup_called:
            return

        self.logger.info(
            "FileTrainManager cleanup - checking for active training jobs..."
        )

        # Note: FileTrainManager doesn't directly track jobs, but we should log
        # that cleanup was called. The actual job cleanup happens in JobManager
        # since that's where the job instances are stored.

        self.logger.info("FileTrainManager cleanup completed")
        self._cleanup_called = True
