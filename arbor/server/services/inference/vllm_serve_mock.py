# Mock version of vllm_serve.py for testing purposes
import asyncio
import signal
import time
from argparse import Namespace

import uvicorn
from fastapi import FastAPI, Request


# Mock GPU operations - no actual GPU dependencies
class MockWeightSyncWorkerExtension:
    """Mock version of WeightSyncWorkerExtension for testing"""

    def __init__(self):
        self.pynccl_comm = None
        self.client_rank = None
        self.device = "cpu"
        self.model_runner = MockModelRunner()

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """Mock communicator initialization"""
        print(
            f"Mock: Initializing communicator with host={host}, port={port}, world_size={world_size}"
        )
        self.pynccl_comm = MockPyNcclCommunicator()
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype_str: str, shape: tuple) -> None:
        """Mock weight update"""
        print(f"Mock: Updating parameter {name} with dtype={dtype_str}, shape={shape}")
        time.sleep(0.01)  # Simulate some processing time

    def close_communicator(self) -> None:
        """Mock communicator cleanup"""
        print("Mock: Closing communicator")
        self.pynccl_comm = None
        self.client_rank = None


class MockPyNcclCommunicator:
    """Mock NCCL communicator for testing"""

    def __init__(self):
        self.group = MockStatelessProcessGroup()

    def broadcast(self, tensor, src: int):
        """Mock broadcast operation"""
        print(f"Mock: Broadcasting tensor from src={src}")


class MockStatelessProcessGroup:
    """Mock process group for testing"""

    def barrier(self):
        """Mock barrier operation"""
        print("Mock: Process group barrier")


class MockModelRunner:
    """Mock model runner for testing"""

    def __init__(self):
        self.model = MockModel()


class MockModel:
    """Mock model for testing"""

    def load_weights(self, weights):
        """Mock weight loading"""
        print(f"Mock: Loading {len(weights)} weight tensors")


class MockAsyncLLMEngine:
    """Mock async LLM engine for testing"""

    def __init__(self, *args, **kwargs):
        self.world_size = kwargs.get("world_size", 1)

    async def collective_rpc(self, method: str, args: tuple = ()):
        """Mock collective RPC call"""
        print(f"Mock: Collective RPC call to {method} with args {args}")
        await asyncio.sleep(0.001)  # Simulate async operation

    async def get_vllm_config(self):
        """Mock vLLM config"""
        return {"mock": True}

    async def reset_prefix_cache(self):
        """Mock prefix cache reset"""
        print("Mock: Resetting prefix cache")
        await asyncio.sleep(0.001)


def create_mock_app(args: Namespace) -> FastAPI:
    """Create mock FastAPI app with same endpoints as real vllm_serve"""
    app = FastAPI(title="Mock vLLM Server")

    # Mock engine
    engine = MockAsyncLLMEngine(
        world_size=args.tensor_parallel_size * args.data_parallel_size
    )

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "ok", "mock": True}

    @app.get("/get_world_size")
    async def get_world_size():
        """Get world size (mocked)"""
        return {"world_size": args.tensor_parallel_size * args.data_parallel_size}

    @app.post("/init_communicator")
    async def init_communicator(request: Request):
        """Mock communicator initialization"""
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size")
        print(
            f"Mock: Init communicator request - host={host}, port={port}, world_size={world_size}"
        )
        await engine.collective_rpc("init_communicator", args=(host, port, world_size))
        return {"status": "ok", "mock": True}

    @app.post("/update_named_param")
    async def update_named_param(request: Request):
        """Mock weight update endpoint"""
        data = await request.json()
        name = data.get("name")
        dtype_str = data.get("dtype")
        shape = data.get("shape")
        print(
            f"Mock: Update param request - name={name}, dtype={dtype_str}, shape={shape}"
        )
        await engine.collective_rpc(
            "update_named_param", args=(name, dtype_str, tuple(shape))
        )
        return {"status": "ok", "mock": True}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache(request: Request):
        """Mock prefix cache reset"""
        await engine.reset_prefix_cache()
        return {"status": "ok", "mock": True}

    @app.post("/close_communicator")
    async def close_communicator(request: Request):
        """Mock communicator cleanup"""
        await engine.collective_rpc("close_communicator")
        return {"status": "ok", "mock": True}

    # Mock OpenAI-compatible endpoints for basic functionality
    @app.post("/v1/completions")
    async def completions(request: Request):
        """Mock completions endpoint"""
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 100)

        # Return a simple mock completion
        return {
            "id": "mock-completion-1",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "text": f" This is a mock completion for prompt: {prompt[:50]}...",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": max_tokens,
                "total_tokens": len(prompt.split()) + max_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Mock chat completions endpoint"""
        data = await request.json()
        messages = data.get("messages", [])

        # Return a simple mock chat completion
        return {
            "id": "mock-chat-completion-1",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response from the chat completion endpoint.",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(
                    len(msg.get("content", "").split()) for msg in messages
                ),
                "completion_tokens": 10,
                "total_tokens": sum(
                    len(msg.get("content", "").split()) for msg in messages
                )
                + 10,
            },
        }

    @app.get("/v1/models")
    async def list_models():
        """Mock models endpoint - required for server readiness check"""
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mock-organization",
                }
            ],
        }

    return app


async def run_server(args: Namespace):
    """Run mock vLLM server"""
    print(f"Mock vLLM Server starting on {args.host}:{args.port}")
    print("This is a mock server for testing - no actual GPU operations will occur")

    app = create_mock_app(args)

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    config = uvicorn.Config(
        app=app,
        host=args.host or "0.0.0.0",
        port=args.port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        print("Mock vLLM Server shutting down...")


def main():
    """Main entry point - parse args and run mock server"""
    import argparse

    parser = argparse.ArgumentParser(description="Mock vLLM OpenAI-compatible server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Mock tensor parallel size"
    )
    parser.add_argument(
        "--data-parallel-size", type=int, default=1, help="Mock data parallel size"
    )
    parser.add_argument(
        "--model", type=str, default="mock-model", help="Mock model name"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Mock GPU memory utilization",
    )

    # Accept any additional arguments that might be passed from the real script
    args, unknown = parser.parse_known_args()

    print(f"Mock vLLM Server Args: {args}")
    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        print("Mock server interrupted")


if __name__ == "__main__":
    main()
