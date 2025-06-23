import arbor

arbor.serve(
    storage_path="/home/noah.ziems/.arbor/storage",
    inference_gpus="2",
    training_gpus="3",
)

if arbor.is_running():
    print("Server is running")
else:
    print("Server is not running")

arbor.stop()

if arbor.is_running():
    print("Server is running")
else:
    print("Server is not running")
