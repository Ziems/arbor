"""
Clean abstraction for running long-running processes without the Popen ugliness.
"""

import subprocess
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


class ProcessRunner:
    """Clean abstraction for running and managing long-running processes."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.process: Optional[subprocess.Popen] = None
        self.log_thread: Optional[threading.Thread] = None
        self.stop_logging = threading.Event()

    def start(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> subprocess.Popen:
        """
        Start a long-running process with clean logging.

        Args:
            command: Command and arguments to run
            env: Environment variables
            cwd: Working directory
            log_callback: Function to call with each log line
        """
        logger.info(f"Starting process for {self.job_id}: {' '.join(command)}")

        self.process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
        )

        # Start log streaming in background thread
        if log_callback or True:  # Always log at minimum
            self._start_log_streaming(log_callback)

        return self.process

    def _start_log_streaming(self, log_callback: Optional[Callable[[str], None]]):
        """Start background thread to stream process logs."""

        def stream_logs():
            if not self.process:
                return

            for line in iter(self.process.stdout.readline, ""):
                if self.stop_logging.is_set():
                    break

                line = line.strip()
                if line:
                    # Call custom callback if provided, otherwise log to our logger
                    if log_callback:
                        try:
                            log_callback(line)
                        except Exception as e:
                            logger.error(f"Error in log callback: {e}")
                    else:
                        # Only log directly if no callback is provided
                        logger.info(f"[{self.job_id}] {line}")

        self.log_thread = threading.Thread(target=stream_logs, daemon=True)
        self.log_thread.start()

    def terminate(self, timeout: int = 10) -> bool:
        """
        Gracefully terminate the process.

        Args:
            timeout: Seconds to wait before force killing

        Returns:
            True if terminated successfully
        """
        if not self.process:
            return True

        logger.info(f"Terminating process for {self.job_id}")

        # Stop log streaming
        self.stop_logging.set()

        try:
            # Try graceful termination first
            self.process.terminate()

            try:
                self.process.wait(timeout=timeout)
                logger.info(f"Process for {self.job_id} terminated gracefully")
                return True
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                logger.warning(f"Force killing process for {self.job_id}")
                self.process.kill()
                self.process.wait(timeout=5)
                logger.info(f"Process for {self.job_id} force killed")
                return True

        except Exception as e:
            logger.error(f"Error terminating process for {self.job_id}: {e}")
            return False
        finally:
            if self.log_thread and self.log_thread.is_alive():
                self.log_thread.join(timeout=1)

    def is_running(self) -> bool:
        """Check if the process is still running."""
        if not self.process:
            return False
        return self.process.poll() is None

    def wait(self, timeout: Optional[int] = None) -> int:
        """Wait for process to complete and return exit code."""
        if not self.process:
            return 0
        return self.process.wait(timeout=timeout)

    @property
    def pid(self) -> Optional[int]:
        """Get the process ID."""
        return self.process.pid if self.process else None

    @property
    def returncode(self) -> Optional[int]:
        """Get the process return code."""
        return self.process.returncode if self.process else None


class AccelerateProcessRunner(ProcessRunner):
    """Specialized runner for accelerate-based training processes."""

    def start_training(
        self,
        script_path: str,
        num_processes: int,
        main_process_port: int,
        script_args: List[str],
        accelerate_config: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> subprocess.Popen:
        """
        Start an accelerate-based training process.

        Args:
            script_path: Path to the training script
            num_processes: Number of processes for accelerate
            main_process_port: Port for main process
            script_args: Arguments to pass to the script (everything after script path)
            accelerate_config: Optional accelerate config file path
            env: Environment variables
            log_callback: Function to call with each log line
        """
        command = [
            "python",
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            str(main_process_port),
        ]

        if accelerate_config:
            command.extend(["--config_file", accelerate_config])

        command.append(script_path)
        command.extend(script_args)

        return self.start(command, env=env, log_callback=log_callback)

    def start_training_from_full_command(
        self,
        full_command: List[str],
        env: Optional[Dict[str, str]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> subprocess.Popen:
        """
        Start training from a pre-built command list (simpler alternative).

        Args:
            full_command: Complete command list ready to execute
            env: Environment variables
            log_callback: Function to call with each log line
        """
        return self.start(full_command, env=env, log_callback=log_callback)


class InferenceProcessRunner(ProcessRunner):
    """Specialized runner for inference server processes."""

    def start_inference_server(
        self,
        command_str: str,
        env: Optional[Dict[str, str]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> subprocess.Popen:
        """
        Start an inference server process.

        Args:
            command_str: Full command string (will be split)
            env: Environment variables
            log_callback: Function to call with each log line
        """
        # Clean up the command string and split it
        clean_command = command_str.replace("\\\n", " ").replace("\\", " ")
        command = clean_command.split()

        return self.start(command, env=env, log_callback=log_callback)
