import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/completions")
async def run_inference(
    request: Request,
):
    inference_manager = request.app.state.inference_manager
    raw_json = await request.json()

    prefixes = ["openai/", "huggingface/", "local:", "arbor:"]
    for prefix in prefixes:
        if raw_json["model"].startswith(prefix):
            raw_json["model"] = raw_json["model"][len(prefix) :]

    # if a server isnt running, launch one
    if (
        not inference_manager.is_server_running()
        and not inference_manager.is_server_restarting()
    ):
        print("No model is running, launching model...")
        inference_manager.launch(raw_json["model"])

    if inference_manager.is_server_restarting():
        print("Waiting for server to finish restarting...")
        while inference_manager.is_server_restarting():
            time.sleep(5)
        # Update the model in the request
        raw_json["model"] = inference_manager.current_model

    # forward the request to the inference server
    completion = await inference_manager.run_inference(raw_json)

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
