import arbor
print(arbor.__file__)


from arbor.server.services.file_manager import FileManager
from arbor.server.services.inference_manager import InferenceManager
# import arbor.server.services.inference_manager
from arbor.server.core.config import Settings

settings = Settings()
my_manager = InferenceManager(settings)


my_manager.launch("meta-llama/Meta-Llama-3-8B")