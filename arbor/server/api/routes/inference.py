import json
import uuid

from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/completions")
async def run_inference(
    request: Request,
):
    inference_manager = request.app.state.inference_manager
    raw_json = await request.json()

    # Generate a random hex ID
    request_id = str(uuid.uuid4())
    # Create requests directory if it doesn't exist
    with open(f"{request.app.state.log_dir}/inference_requests.jsonl", "a") as f:
        f.write(json.dumps({"id": request_id, "request": raw_json}) + "\n")

    # if a server isnt running, launch one
    if not inference_manager.is_server_running():
        print("No model is running, launching model...")
        print(f"Raw request JSON: {raw_json}")
        inference_manager.launch(
            raw_json["model"], {"max_context_length": 32768}
        )  # Placeholder for now

    # forward the request to the inference server
    completion = await inference_manager.run_inference(raw_json)

    with open(f"{request.app.state.log_dir}/inference_responses.jsonl", "a") as f:
        f.write(json.dumps({"id": request_id, "response": completion}) + "\n")

    return completion


@router.post("/launch")
async def launch_inference(request: Request):
    inference_manager = request.app.state.inference_manager
    raw_json = await request.json()
    inference_manager.launch(raw_json["model"], raw_json["launch_kwargs"])
    return {"message": "Inference server launched"}


@router.post("/kill")
async def kill_inference(request: Request):
    inference_manager = request.app.state.inference_manager
    inference_manager.kill()
    return {"message": "Inference server killed"}
