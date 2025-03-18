import time
import uuid
from fastapi import APIRouter, Request
from datetime import datetime
import asyncio
from typing import Optional

from arbor.server.api.models.schemas import ChatCompletionModel, ChatCompletionRequest, ChatCompletionMessage, ChatCompletionChoice
from arbor.server.services.job_manager import JobStatus

router = APIRouter()


@router.post("/completions", response_model=ChatCompletionModel)
def run_inference(request: Request, chat_completion_request: ChatCompletionRequest):
    inference_manager = request.app.state.inference_manager
    inference_manager.last_activity = datetime.now()

    job_manager = request.app.state.job_manager

    active_job = job_manager.get_active_job()

    if active_job is not None:
        active_job.status = JobStatus.PENDING_PAUSE

        while active_job.status != JobStatus.PAUSED: # TODO: This should be improved incase the job does not pause...etc
            time.sleep(0.5)

    # if a server isnt running, launch one
    inference_manager.launch(chat_completion_request.model)

    # forward the request to the inference server
    completion = inference_manager.run_inference(chat_completion_request)


    # Resume Training if it was paused
    if active_job is not None and active_job.status == JobStatus.PAUSED:
        active_job.status = JobStatus.PENDING_RESUME

    return completion

