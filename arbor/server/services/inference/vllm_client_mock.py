"""
Mock vLLM client for testing and development without GPU dependencies.

Provides the same interface as VLLMClient but with lightweight mock implementations.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MockVLLMClient:
    """Mock vLLM client that simulates the real client interface."""

    def __init__(
        self,
        model_name: str = None,
        port: int = 8000,
        host: str = "localhost",
        group_port: int = None,
        connection_timeout: int = 60,
        **kwargs,
    ):
        self.model_name = model_name or "mock-model"
        self.port = port
        self.host = host
        self.group_port = group_port
        self.connection_timeout = connection_timeout
        self.base_url = f"http://{host}:{port}"
        self.is_running = False
        self.start_time = None

        logger.info(
            f"Mock vLLM client initialized for model: {self.model_name} on port {port}"
        )

    def start_server(self, **kwargs) -> bool:
        """Mock server start - always succeeds."""
        logger.info(f"Mock vLLM server starting on {self.host}:{self.port}")
        self.is_running = True
        self.start_time = time.time()
        return True

    def stop_server(self) -> bool:
        """Mock server stop - always succeeds."""
        logger.info("Mock vLLM server stopping")
        self.is_running = False
        return True

    def is_server_running(self) -> bool:
        """Check if mock server is running."""
        return self.is_running

    def wait_for_server(self, timeout: int = 60) -> bool:
        """Mock wait for server - returns immediately if running."""
        if self.is_running:
            logger.info("Mock vLLM server is ready")
            return True
        return False

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Mock text generation."""
        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Extract the last user message
        user_message = "Hello"
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "Hello")
                break

        # Generate a mock response
        mock_response = f"Mock response to: {user_message[:50]}..."

        return {
            "id": f"mock-completion-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": mock_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(mock_response.split()),
                "total_tokens": len(user_message.split()) + len(mock_response.split()),
            },
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Mock chat completion endpoint."""
        if stream:
            return self._mock_stream_response(messages, max_tokens, temperature)
        else:
            return await self.generate(messages, max_tokens, temperature, **kwargs)

    async def chat(self, json_body: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Mock chat method that matches the real VLLMClient interface."""
        messages = json_body.get("messages", [])
        model = json_body.get("model", self.model_name)
        max_tokens = json_body.get("max_tokens", 150)
        temperature = json_body.get("temperature", 0.7)
        stream = json_body.get("stream", False)

        return await self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )

    def _mock_stream_response(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ):
        """Mock streaming response generator."""
        user_message = "Hello"
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "Hello")
                break

        mock_response = f"Mock streaming response to: {user_message[:30]}..."
        words = mock_response.split()

        def stream_generator():
            for i, word in enumerate(words):
                chunk = {
                    "id": f"mock-stream-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + " " if i < len(words) - 1 else word
                            },
                            "finish_reason": None if i < len(words) - 1 else "stop",
                        }
                    ],
                }
                yield f"data: {chunk}\n\n"
                time.sleep(0.05)  # Simulate streaming delay

            # End of stream
            yield "data: [DONE]\n\n"

        return stream_generator()

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_name": self.model_name,
            "model_path": f"/mock/models/{self.model_name}",
            "model_size": "7B",
            "dtype": "float16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
            "quantization": None,
            "served_model_name": self.model_name,
        }

    def get_server_stats(self) -> Dict[str, Any]:
        """Get mock server statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "total_requests": 42,  # Mock counter
            "successful_requests": 40,
            "failed_requests": 2,
            "average_response_time_ms": 150.5,
            "model": self.model_name,
            "host": self.host,
            "port": self.port,
        }

    def health_check(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "model": self.model_name,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "memory_usage": "1.2GB",
            "gpu_utilization": "0%",  # Mock - no real GPU usage
            "last_request": time.time() - 30,  # Mock - 30 seconds ago
        }

    def shutdown(self):
        """Shutdown the mock client."""
        logger.info("Mock vLLM client shutting down")
        self.stop_server()

    def __del__(self):
        """Cleanup on deletion."""
        if self.is_running:
            self.shutdown()


# For compatibility with the real VLLMClient interface
VLLMClient = MockVLLMClient
