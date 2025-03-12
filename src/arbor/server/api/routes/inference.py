import time
import uuid
from fastapi import APIRouter, Request

from arbor.server.api.models.schemas import ChatCompletionModel, ChatCompletionRequest, ChatCompletionMessage, ChatCompletionChoice
from arbor.server.services.job_manager import JobStatus

router = APIRouter()

@router.post("/completions", response_model=ChatCompletionModel)
def upload_file(request: Request, chat_completion_request: ChatCompletionRequest):
    active_job = request.app.state.job_manager.get_active_job()
    if active_job is not None:
        active_job.status = JobStatus.PENDING_PAUSE

        while active_job.status != JobStatus.PAUSED: # TODO: This should be improved incase the job does not pause...etc
            time.sleep(0.5)

    # Run Inference Here
    completion = ChatCompletionModel(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=chat_completion_request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello, how are you?"
                ),
                finish_reason="stop",
                index=0
            )
        ]
    )

    # Resume Training if it was paused
    if active_job is not None and active_job.status == JobStatus.PAUSED:
        active_job.status = JobStatus.PENDING_RESUME

    return completion

