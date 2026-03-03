from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GROQ_API_KEY: str
    UPLOAD_DIR: str = "src/uploads"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
