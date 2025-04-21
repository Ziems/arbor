from fastapi import APIRouter, Request, BackgroundTasks

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
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    # Define the reward function, which rewards completions that are close to 20 characters
    def reward_len(completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

    training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    return {"status": "success"}