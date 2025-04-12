from arbor.server.services.inference_manager import InferenceManager
from arbor.server.core.config import Settings

settings = Settings()
my_manager = InferenceManager(settings)


my_manager.launch("Qwen/Qwen2.5-1.5B-Instruct")

response = my_manager.run_inference("How are you today?")
print(response)

my_manager.kill()
print("successfully killed inference manager")
print("existing process:", my_manager.process)