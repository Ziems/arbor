from fastapi import APIRouter, Request

from arbor.core.logging import get_logger
from arbor.server.api.schemas import (
    InferenceLaunchRequest,
    InferenceTerminateRequest,
)
from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.utils.helpers import strip_prefix

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
async def launch_inference(request: Request, launch_request: InferenceLaunchRequest):
    inference_manager: InferenceManager = request.app.state.inference_manager
    normalized_request = launch_request.model_copy(
        # stripe the prefix (such as "openai/arbor:")
        update={"model": strip_prefix(launch_request.model)}
    )

    inference_job = inference_manager.launch_from_request(normalized_request)

    return {
        "message": "Inference server launched",
        "job_id": inference_job.id,
        "model": inference_job.launched_model_name,
    }


@router.post("/kill")
async def kill_inference(
    request: Request, terminate_request: InferenceTerminateRequest
):
    inference_manager: InferenceManager = request.app.state.inference_manager

    job_id = terminate_request.job_id

    if job_id == "*":
        inference_manager.cleanup()
        return {"message": "All inference servers terminated"}
    else:
        summary = inference_manager.terminate_job(
            job_id,
        )
        return {"message": "Inference server terminated", **summary}
