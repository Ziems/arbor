import os
import subprocess

from fastapi import APIRouter, BackgroundTasks, Request

from arbor.server.api.models.schemas import (
    GRPOConfigRequest,
    GRPOConfigResponse,
    GRPORequest,
    GRPOStepResponse,
    GRPOTerminateRequest,
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
def run_grpo_step(
    request: Request, grpo_request: GRPORequest, background_tasks: BackgroundTasks
):
    inference_manager = request.app.state.inference_manager
    grpo_manager = request.app.state.grpo_manager

    current_model = grpo_manager.grpo_step(grpo_request, inference_manager)

    return GRPOStepResponse(status="success", current_model=current_model)


@router.post("/update_model", response_model=GRPOStepResponse)
def update_model(request: Request):
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager
    current_model = grpo_manager.update_model(request, inference_manager)
    return GRPOStepResponse(status="success", current_model=current_model)


@router.post("/terminate", response_model=GRPOTerminateResponse)
def terminate_grpo(request: Request):
    # No body needed for this request at this moment
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager

    final_model = grpo_manager.terminate(inference_manager)
    return GRPOTerminateResponse(status="success", current_model=final_model)
