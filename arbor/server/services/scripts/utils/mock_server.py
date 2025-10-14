## Mock arbor sending over data for testing
import threading
import time

import zmq

from arbor.server.services.comms.comms import ArborServerCommsHandler

group_example = [  # Entire group of trajectories
    [  # Trajectory with different modules
        {  # geography module
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "completion": [{"role": "assistant", "content": "Paris"}],
            "advantage": 0.9,
        },
        {  # math module
            "messages": [{"role": "user", "content": "What is 2 * 2 + 2?"}],
            "completion": [{"role": "assistant", "content": "6"}],
            "advantage": 0.8,
        },
        {  # car module
            "messages": [
                {"role": "user", "content": "When did the first honda civic come out?"}
            ],
            "completion": [{"role": "assistant", "content": "1973"}],
            "advantage": 0.7,
        },
    ],
    [  # Trajectory with different modules
        {  # geography module
            "messages": [
                {"role": "user", "content": "What is the capital of Germany?"}
            ],
            "completion": {
                "role": "assistant",
                "content": "Berlin is the capital of Germany",
            },
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


def debug_data_generator(server_comms_handler):
    idx = 0
    while True:
        print("Sending group:")  # Debug print
        server_comms_handler.send_data(group_example)
        idx += 1
        time.sleep(1)

        if idx >= 100:
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

    # Get available ports from the server comms handler
    command_port = server_comms_handler.command_port
    status_port = server_comms_handler.status_port
    data_port = server_comms_handler.data_port
    broadcast_port = server_comms_handler.broadcast_port
    handshake_port = server_comms_handler.handshake_port

    # Print the command that would be used to connect to this mock server
    print("\nTo connect to this mock server, run the following command:")
    print(
        "CUDA_VISIBLE_DEVICES=2 python arbor/server/services/scripts/mmgrpo_training.py \\"
    )
    print("    --debug \\")
    print(f"    --command_port {command_port} \\")
    print(f"    --status_port {status_port} \\")
    print(f"    --data_port {data_port} \\")
    print(f"    --broadcast_port {broadcast_port} \\")
    print(f"    --handshake_port {handshake_port} \\")
    print("    --vllm_group_port 0 \\")
    print("    --vllm_port 0 \\")
    print("    --model Qwen/Qwen3-0.6B \\")
    print('    --trl_train_kwargs \'{"output_dir": ".", "report_to": "none"}\'')
    print(
        "\nThis mock server will simulate sending training data to the training process."
    )
    print("Press Ctrl+C to exit the mock server.\n")

    server_comms_handler.wait_for_clients(1)

    debug_thread = threading.Thread(
        target=debug_data_generator, args=(server_comms_handler,), daemon=True
    )
    debug_thread.start()

    status_listener_thread = threading.Thread(
        target=status_listener, args=(server_comms_handler,), daemon=True
    )
    status_listener_thread.start()

    try:
        print("Mock server started and waiting for training process to connect...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down mock server...")
    finally:
        server_comms_handler.close()
        print("Mock server shutdown complete.")
