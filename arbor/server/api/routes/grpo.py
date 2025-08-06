from fastapi import APIRouter, BackgroundTasks, Request

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
    GRPOTerminateRequest,
)
from arbor.server.services.managers.grpo_manager import GRPOManager
from arbor.server.services.managers.inference_manager import InferenceManager

router = APIRouter()


@router.post("/initialize", response_model=GRPOStatus)
def initialize_grpo(request: Request, grpo_initialize_request: GRPOInitializeRequest):
    inference_manager: InferenceManager = request.app.state.inference_manager
    grpo_manager: GRPOManager = request.app.state.grpo_manager
    grpo_status: GRPOStatus = grpo_manager.initialize(
        grpo_initialize_request, inference_manager
    )
    return grpo_status


@router.post("/{job_id}/step", response_model=GRPOStatus)
def run_grpo_step(request: Request, job_id: str, grpo_request: GRPOStepRequest):
    grpo_manager: GRPOManager = request.app.state.grpo_manager
    # Override job_id from URL
    grpo_request.job_id = job_id
    grpo_status: GRPOStatus = grpo_manager.route_grpo_step(grpo_request)

    return grpo_status


@router.post("/{job_id}/checkpoint", response_model=GRPOStatus)
def checkpoint(
    request: Request, job_id: str, grpo_checkpoint_request: GRPOCheckpointRequest
):
    grpo_manager: GRPOManager = request.app.state.grpo_manager
    # Override job_id from URL
    grpo_checkpoint_request.job_id = job_id
    grpo_status: GRPOStatus = grpo_manager.route_grpo_checkpoint(
        grpo_checkpoint_request
    )
    return grpo_status


@router.post("/{job_id}/cancel", response_model=GRPOStatus)
def cancel_grpo(request: Request, job_id: str):
    from fastapi import HTTPException

    grpo_manager: GRPOManager = request.app.state.grpo_manager

    try:
        grpo_status: GRPOStatus = grpo_manager.cancel(job_id)
        return grpo_status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel GRPO job: {str(e)}"
        )


@router.post("/{job_id}/terminate", response_model=GRPOStatus)
def terminate_grpo(request: Request, job_id: str):
    from arbor.server.api.models.schemas import GRPOTerminateRequest

    grpo_manager: GRPOManager = request.app.state.grpo_manager

    # Create the request object with job_id from URL
    terminate_request = GRPOTerminateRequest(job_id=job_id)
    grpo_status: GRPOStatus = grpo_manager.terminate(terminate_request)
    return grpo_status
