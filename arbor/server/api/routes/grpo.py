import os
import subprocess

from fastapi import APIRouter, BackgroundTasks, Request

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOCheckpointResponse,
    GRPOConfigRequest,
    GRPOConfigResponse,
    GRPORequest,
    GRPOStepResponse,
    GRPOTerminateResponse,
)

router = APIRouter()


@router.post("/initialize", response_model=GRPOConfigResponse)
def initialize_grpo(request: Request, grpo_config_request: GRPOConfigRequest):
    inference_manager = request.app.state.inference_manager
    grpo_manager = request.app.state.grpo_manager
    grpo_manager.initialize(grpo_config_request, inference_manager)
    return GRPOConfigResponse(status="success")


# Create a grpo job
@router.post("/step", response_model=GRPOStepResponse)
def run_grpo_step(request: Request, grpo_request: GRPORequest):
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager
    step_data = grpo_manager.grpo_step(grpo_request, inference_manager)

    return GRPOStepResponse(status="success", **step_data)


@router.post("/checkpoint", response_model=GRPOCheckpointResponse)
def checkpoint(request: Request, grpo_checkpoint_request: GRPOCheckpointRequest):
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager
    checkpoint_data = grpo_manager.checkpoint(
        grpo_checkpoint_request, inference_manager
    )
    return GRPOCheckpointResponse(status="success", **checkpoint_data)


@router.post("/terminate", response_model=GRPOTerminateResponse)
def terminate_grpo(request: Request):
    # No body needed for this request at this moment
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager

    terminate_data = grpo_manager.terminate(inference_manager)
    return GRPOTerminateResponse(status="success", **terminate_data)
