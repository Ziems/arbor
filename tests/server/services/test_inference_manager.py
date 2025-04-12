from arbor.server.services.inference_manager import InferenceManager
from arbor.server.core.config import Settings

settings = Settings()
my_manager = InferenceManager(settings)


my_manager.launch("meta-llama/Meta-Llama-3-8B")