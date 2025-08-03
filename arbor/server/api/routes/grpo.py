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


# Create a grpo job
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


@router.post("/terminate", response_model=GRPOStatus)
def terminate_grpo(request: GRPOTerminateRequest):
    # No body needed for this request at this moment
    grpo_manager: GRPOManager = request.app.state.grpo_manager

    grpo_status: GRPOStatus = grpo_manager.terminate(request)
    return grpo_status
