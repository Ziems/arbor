from fastapi import APIRouter, Request

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
    GRPOTerminateRequest,
    GRPOBaseRequest,  # TODO: These should be handled with a subclass like GRPOTerminateRequest
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


@router.post("/status", response_model=GRPOStatus)
def get_grpo_status(request: Request, grpo_request: GRPOBaseRequest):
    grpo_manager: GRPOManager = request.app.state.grpo_manager
    grpo_status: GRPOStatus = grpo_manager.get_job_status(grpo_request.job_id)
    return grpo_status


@router.post("/step", response_model=GRPOStatus)
def run_grpo_step(request: Request, grpo_request: GRPOStepRequest):
    grpo_manager: GRPOManager = request.app.state.grpo_manager
    grpo_status: GRPOStatus = grpo_manager.route_grpo_step(grpo_request)
    return grpo_status


@router.post("/checkpoint", response_model=GRPOStatus)
def checkpoint(request: Request, grpo_checkpoint_request: GRPOCheckpointRequest):
    grpo_manager: GRPOManager = request.app.state.grpo_manager
    grpo_status: GRPOStatus = grpo_manager.route_grpo_checkpoint(
        grpo_checkpoint_request
    )
    return grpo_status


@router.post("/cancel", response_model=GRPOStatus)
def cancel_grpo(request: Request, grpo_request: GRPOBaseRequest):
    from fastapi import HTTPException

    grpo_manager: GRPOManager = request.app.state.grpo_manager

    try:
        grpo_status: GRPOStatus = grpo_manager.cancel(grpo_request.job_id)
        return grpo_status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel GRPO job: {str(e)}"
        )

@router.post("/terminate", response_model=GRPOStatus)
def terminate_grpo(request: Request, grpo_request: GRPOTerminateRequest):
    grpo_manager: GRPOManager = request.app.state.grpo_manager
    grpo_status: GRPOStatus = grpo_manager.terminate(grpo_request)
    return grpo_status
