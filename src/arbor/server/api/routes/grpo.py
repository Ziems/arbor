from fastapi import APIRouter, Request, BackgroundTasks

from arbor.server.api.models.schemas import JobStatusModel, GRPORequest, GRPOConfigRequest, GRPOConfigResponse, GRPOTerminateRequest, GRPOTerminateResponse

router = APIRouter()

@router.post("/initialize", response_model=GRPOConfigResponse)
def initialize_grpo(request: Request, grpo_config_request: GRPOConfigRequest):
    inference_manager = request.app.state.inference_manager
    grpo_manager = request.app.state.grpo_manager
    grpo_manager.initialize_config(grpo_config_request, inference_manager)
    return GRPOConfigResponse(status="success")

# Create a grpo job
@router.post("/step", response_model=JobStatusModel)
def run_grpo_step(request: Request, grpo_request: GRPORequest, background_tasks: BackgroundTasks):
    job_manager = request.app.state.job_manager
    inference_manager = request.app.state.inference_manager
    grpo_manager = request.app.state.grpo_manager

    job = job_manager.create_job()

    grpo_manager.grpo_step(grpo_request, job, inference_manager)


    # if inference_manager.is_server_running():
    #     inference_manager.kill()
    #     while inference_manager.is_server_running(): # TODO: This should be done cleaner
    #         time.sleep(1)

    job = job_manager.create_job()
    return JobStatusModel(id=job.id, status=job.status.value)

@router.post("/terminate", response_model=GRPOTerminateResponse)
def terminate_grpo(request: Request, grpo_request: GRPOTerminateRequest):
    grpo_manager = request.app.state.grpo_manager
    inference_manager = request.app.state.inference_manager

    grpo_manager.terminate(inference_manager)
    import pdb; pdb.set_trace()
    return GRPOTerminateResponse(status="success")
