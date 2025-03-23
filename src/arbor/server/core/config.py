from pydantic import BaseModel, ConfigDict

class Settings(BaseModel):
    model_config = ConfigDict(
        env_file=".env",  # Add any config options you need
    )

    STORAGE_PATH: str = "./storage"
    INACTIVITY_TIMEOUT: int = 5 # 5 seconds

settings = Settings()