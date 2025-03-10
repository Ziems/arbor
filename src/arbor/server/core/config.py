from pydantic import BaseModel, ConfigDict

class Settings(BaseModel):
    model_config = ConfigDict(
        env_file=".env",  # Add any config options you need
    )

    STORAGE_PATH: str = "./storage"

settings = Settings()