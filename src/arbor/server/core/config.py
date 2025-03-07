from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    UPLOADS_DIR: str = "uploads"
    MODEL_CACHE_DIR: str = "model_cache"

    class Config:
        env_file = ".env"

settings = Settings()