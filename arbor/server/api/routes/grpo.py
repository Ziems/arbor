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
    GRPOUpdateModelRequest,
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

    grpo_model_data = grpo_manager.grpo_step(grpo_request, inference_manager)

    return GRPOStepResponse(status="success", **grpo_model_data)


@router.post("/update_model", response_model=GRPOStepResponse)
def update_model(request: Request, grpo_update_model_request: GRPOUpdateModelRequest):
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager
    try:
        update_model_data = grpo_manager.update_model(
            grpo_update_model_request, inference_manager
        )
    except Exception as e:
        import pdb

        pdb.set_trace()
        return GRPOStepResponse(status="error", error=str(e))
    return GRPOStepResponse(status="success", **update_model_data)


@router.post("/terminate", response_model=GRPOTerminateResponse)
def terminate_grpo(request: Request):
    # No body needed for this request at this moment
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager

    final_model = grpo_manager.terminate(inference_manager)
    return GRPOTerminateResponse(status="success", current_model=final_model)
