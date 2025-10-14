from fastapi import APIRouter, Request

from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.server.utils.helpers import strip_prefix
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/completions")
async def run_inference(
    request: Request,
):
    inference_manager: InferenceManager = request.app.state.inference_manager
    raw_json = await request.json()
    raw_json["model"] = strip_prefix(raw_json["model"])

    # forward the request to the inference server
    completion = await inference_manager.route_inference(raw_json)

    # # Write combined request+completion to a single JSONL for inspection
    # try:
    #     combined = {
    #         "request": raw_json,
    #         "completion": completion,
    #     }
    #     with open("completion_inspect.jsonl", "a") as f:
    #         f.write(json.dumps(combined) + "\n")
    # except Exception as exc:
    #     logger.warning(f"Failed to write completion_inspect.jsonl: {exc}")

    return completion


@router.post("/launch")
async def launch_inference(request: Request):
    inference_manager: InferenceManager = request.app.state.inference_manager
    raw_json = await request.json()
    inference_manager.launch(raw_json["model"], raw_json["launch_kwargs"])
    return {"message": "Inference server launched"}


@router.post("/kill")
async def kill_inference(request: Request):
    inference_manager: InferenceManager = request.app.state.inference_manager
    inference_manager.cleanup()
    return {"message": "Inference server terminated"}
