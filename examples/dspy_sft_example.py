import json

import requests

ARBOR_URL = "http://127.0.0.1:7453"


def upload_file(file_path, url=f"{ARBOR_URL}/api/files"):
    """
    Upload a file to the specified URL endpoint.

    Args:
        file_path (str): Path to the file to be uploaded
        url (str): URL endpoint for file upload (default: http://127.0.0.1:7453/api/files)

    Returns:
        requests.Response: Response from the server
    """
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, files=files)
    return response


# Example usage:
response = upload_file("./tests/data/training_data_sft.jsonl")
print(response.status_code)  # Print the HTTP status code
response_body = json.loads(response.text)
uploaded_file = response_body["id"]
print(uploaded_file)


# "HuggingFaceTB/SmolLM2-135M-Instruct"
# "Qwen/Qwen2-0.5B-Instruct"
def start_fine_tune(
    training_file,
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    url=f"{ARBOR_URL}/api/fine-tune",
):
    """
    Start a fine-tuning job with the specified training file and model.

    Args:
        training_file (str): ID of the uploaded training file
        model_name (str): Name of the base model to fine-tune (default: Qwen/Qwen2-0.5B-Instruct)
        url (str): URL endpoint for fine-tuning (default: http://127.0.0.1:8000/api/fine-tune)

    Returns:
        requests.Response: Response from the server
    """
    headers = {"Content-Type": "application/json"}
    data = {"model_name": model_name, "training_file": training_file}
    response = requests.post(url, headers=headers, json=data)
    return response


# Example usage (continuing from previous upload):
# fine_tune_response = start_fine_tune(uploaded_file)
# print(fine_tune_response.status_code)
# print(fine_tune_response.text)
