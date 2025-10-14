"""
Clean abstraction for running long-running processes without the Popen ugliness.
"""

import signal
import sys
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

        # Signal handling for cluster environments
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown in cluster environments."""
        try:

            def signal_handler(signum, _):
                logger.info(
                    f"Received signal {signum} for job {self.job_id}, initiating graceful shutdown..."
                )
                self.terminate(timeout=5)  # Quick termination on signal

            # Handle common termination signals
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # Handle SIGUSR1/SIGUSR2 for custom cluster signals
            try:
                signal.signal(signal.SIGUSR1, signal_handler)
                signal.signal(signal.SIGUSR2, signal_handler)
            except (OSError, ValueError):
                # These signals might not be available on all systems
                pass
        except ValueError:
            # signal.signal() can only be called from the main thread
            logger.debug(
                f"Cannot setup signal handlers for job {self.job_id} - not in main thread"
            )
            pass

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

        # On Linux, ensure child processes receive SIGTERM if the parent dies
        preexec_fn = None
        if sys.platform.startswith("linux"):

            def _set_pdeathsig():
                try:
                    import ctypes

                    libc = ctypes.CDLL("libc.so.6")
                    PR_SET_PDEATHSIG = 1
                    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                except Exception:
                    # Best-effort; fall back to normal spawn if unavailable
                    pass

            preexec_fn = _set_pdeathsig

        self.process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            preexec_fn=preexec_fn,
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
