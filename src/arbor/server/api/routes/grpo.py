from fastapi import APIRouter, Request, BackgroundTasks
import subprocess
import os

from arbor.server.api.models.schemas import GRPOStepResponse, GRPORequest, GRPOConfigRequest, GRPOConfigResponse, GRPOTerminateRequest, GRPOTerminateResponse

router = APIRouter()

@router.post("/initialize", response_model=GRPOConfigResponse)
def initialize_grpo(request: Request, grpo_config_request: GRPOConfigRequest):
    inference_manager = request.app.state.inference_manager
    grpo_manager = request.app.state.grpo_manager
    grpo_manager.initialize_config(grpo_config_request, inference_manager)
    return GRPOConfigResponse(status="success")

# Create a grpo job
@router.post("/step", response_model=GRPOStepResponse)
def run_grpo_step(request: Request, grpo_request: GRPORequest, background_tasks: BackgroundTasks):
    inference_manager = request.app.state.inference_manager
    grpo_manager = request.app.state.grpo_manager

    current_model = grpo_manager.grpo_step(grpo_request, inference_manager)


    # if inference_manager.is_server_running():
    #     inference_manager.kill()
    #     while inference_manager.is_server_running(): # TODO: This should be done cleaner
    #         time.sleep(1)

    return GRPOStepResponse(status="success", current_model=current_model)

@router.post("/terminate", response_model=GRPOTerminateResponse)
def terminate_grpo(request: Request, grpo_request: GRPOTerminateRequest):
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager

    grpo_manager.terminate(inference_manager)
    return GRPOTerminateResponse(status="success")

@router.post("/test")
def test_grpo(request: Request):
    # Get the directory where grpo_testing2.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "grpo_testing2.py")
    print(script_path)

    # Start the accelerate process
    process = subprocess.Popen(
        ["CUDA_VISIBLE_DEVICES=1,2", "python", "-m", "accelerate.commands.launch", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return {"status": "success", "message": "Started GRPO test process"}
