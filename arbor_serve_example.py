from arbor import serve

serve(storage_path="/your-path", inference_gpus="0, 1", training_gpus="2, 3")