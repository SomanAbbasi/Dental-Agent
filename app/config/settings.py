from functools import lru_cache
from pydantic import Field, SecretStr, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",  # crash if unknown env vars exist
    )
    
    # OpenRouter
    openrouter_api_key: SecretStr = Field(..., description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL",
    )
    
    openrouter_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct:free",
        description="Model identifier on OpenRouter",
    )
    
    
     # App
    app_env: str = Field(default="development")
    log_level: str = Field(default="DEBUG")
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    
    
     # Clinic
    clinic_name: str = Field(default="BrightSmile Dental Clinic")
    clinic_phone: str = Field(default="+92-42-0000000")
    
    
    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v):
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v.lower()

    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    
    # @field_validator("openrouter_model")
    # @classmethod
    # def validate_model(cls, v: str) -> str:
    #     if "/" not in v:
    #         raise ValueError("Model must be in format 'provider/model-name'")
    #     return v
    @field_validator("openrouter_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v or len(v.strip()) < 3:
            raise ValueError("Model name cannot be empty")
        return v.strip()
    

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings loader.
    Called once on startup, reused everywhere.
    lru_cache means .env is read exactly once.
    """
    return Settings()