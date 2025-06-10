## Mock arbor sending over data for testing
import threading
import time

import zmq

from arbor.server.services.comms.comms import ArborServerCommsHandler

batch_example = [
    [  # module with different completions (Geography Expert)
        {  # geogrphy module
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "completion": [{"role": "assistant", "content": "Paris"}],
            "advantage": 0.9,
        },
        {  # math module
            "messages": [{"role": "user", "content": "What is 2 * 2 + 2?"}],
            "completion": [{"role": "assistant", "content": "6"}],
            "advantage": 0.8,
        },
    ],
    [  # module with different completions (Math Expert)
        {  # geogrphy module
            "messages": [
                {"role": "user", "content": "What is the capital of Germany?"}
            ],
            "completion": [
                {"role": "assistant", "content": "Berlin is the capital of Germany"}
            ],
            "advantage": 0.1,
        },
        {  # math module
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "completion": [{"role": "assistant", "content": "3"}],
            "advantage": 0.2,
        },
    ],
]


def flatten_batch(batch):
    return [item for sublist in batch for item in sublist]


import pdb

pdb.set_trace()


def debug_data_generator(server_comms_handler):
    idx = 0
    batch_size = 2  # Number of examples to send in each batch
    while True:
        # Create a batch of examples
        start_idx = (idx * batch_size) % len(batch_example)
        end_idx = min(start_idx + batch_size, len(batch_example))
        batch = batch_example[start_idx:end_idx]

        # If we don't have enough examples for a full batch, wrap around
        if len(batch) < batch_size:
            remaining = batch_size - len(batch)
            batch.extend(batch_example[:remaining])

        print(f"Sending batch: {batch}")  # Debug print
        server_comms_handler.send_data(batch)
        idx += 1
        time.sleep(1)

        if idx >= 25:
            server_comms_handler.send_command({"command": "save_model"})


def status_listener(server_comms_handler):
    # Need to set subscription for PUB/SUB pattern
    server_comms_handler.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    for status in server_comms_handler.receive_status():
        print(f"Status: {status}")


if __name__ == "__main__":
    server_comms_handler = ArborServerCommsHandler(
        host="localhost",
    )

    # launch the trainer here
    import subprocess
    import sys

    # Get available ports from the server comms handler
    command_port = server_comms_handler.command_port
    status_port = server_comms_handler.status_port
    data_port = server_comms_handler.data_port
    broadcast_port = server_comms_handler.broadcast_port
    handshake_port = server_comms_handler.handshake_port

    # Construct the command
    cmd = [
        sys.executable,
        "arbor/server/services/scripts/mmgrpo_training.py",
        "--debug",
        "--command_port",
        str(command_port),
        "--status_port",
        str(status_port),
        "--data_port",
        str(data_port),
        "--broadcast_port",
        str(broadcast_port),
        "--handshake_port",
        str(handshake_port),
        "--vllm_group_port",
        str(0),  # TODO: This is unused rn
        "--vllm_port",
        str(0),  # TODO: This is unused rn
        "--model",
        "Qwen/Qwen3-0.6B",
        "--trl_train_kwargs",
        '{"output_dir": ".", "report_to": "none"}',
    ]

    # Launch the training process
    training_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Start threads to monitor stdout/stderr
    def monitor_output(pipe, prefix):
        for line in pipe:
            print(f"{prefix}: {line.strip()}")

    stdout_thread = threading.Thread(
        target=monitor_output, args=(training_process.stdout, "TRAINER"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=monitor_output,
        args=(training_process.stderr, "TRAINER-ERROR"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    handshake_thread = threading.Thread(
        target=server_comms_handler.wait_for_clients, args=(1,), daemon=True
    )
    handshake_thread.start()

    debug_thread = threading.Thread(
        target=debug_data_generator, args=(server_comms_handler,), daemon=True
    )
    debug_thread.start()

    status_listener_thread = threading.Thread(
        target=status_listener, args=(server_comms_handler,), daemon=True
    )
    status_listener_thread.start()

    try:
        print("Server started. Press Ctrl+C to exit.")
        # Keep the main thread alive
        while True:
            time.sleep(1)
            # Check if training process is still alive
            if training_process.poll() is not None:
                print("Training process has terminated!")
                break
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        # Clean up
        training_process.terminate()
        server_comms_handler.close()
        print("Server shutdown complete.")
