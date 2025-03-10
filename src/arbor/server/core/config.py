from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    STORAGE_PATH: str = "./storage"
    class Config:
        env_file = ".env"

settings = Settings()