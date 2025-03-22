import time
from fastapi import APIRouter, Request
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
from arbor.server.api.models.schemas import ChatCompletionModel, ChatCompletionRequest, ChatCompletionMessage, ChatCompletionChoice
from arbor.server.services.job_manager import JobStatus



@asynccontextmanager
async def lifespan(app):
    # Start the inactivity checker
    inactivity_task = asyncio.create_task(check_inactivity(app))
    yield
    # Cleanup: cancel the inactivity checker
    inactivity_task.cancel()
    try:
        await inactivity_task
    except asyncio.CancelledError:
        pass

async def check_inactivity(app):
    inactivity_timeout = app.state.settings.INACTIVITY_TIMEOUT
    while True:
        await asyncio.sleep(5) # check every 5 seconds
        inference_manager = app.state.inference_manager
        if inference_manager.last_activity is not None:
            if datetime.now() - inference_manager.last_activity > timedelta(seconds=inactivity_timeout):
                print("Inactivity timeout reached, killing server")
                inference_manager.kill()


router = APIRouter(lifespan=lifespan)


@router.post("/completions", response_model=ChatCompletionModel)
def run_inference(request: Request, chat_completion_request: ChatCompletionRequest):
    inference_manager = request.app.state.inference_manager
    job_manager = request.app.state.job_manager

    active_job = job_manager.get_active_job()

    if active_job is not None:
        active_job.status = JobStatus.PENDING_PAUSE

        while active_job.status != JobStatus.PAUSED: # TODO: This should be improved incase the job does not pause...etc
            time.sleep(0.5)

    # if a server isnt running, launch one
    if not inference_manager.is_server_running():
        inference_manager.launch(chat_completion_request.model)

    # forward the request to the inference server
    completion = inference_manager.run_inference(chat_completion_request)


    # Resume Training if it was paused
    if active_job is not None and active_job.status == JobStatus.PAUSED:
        active_job.status = JobStatus.PENDING_RESUME

    return completion

