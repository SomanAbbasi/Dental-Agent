from functools import lru_cache
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.config.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
   
    settings = get_settings()
    logger.info(
        "initializing_llm",
        model=settings.openrouter_model,
        env=settings.app_env,
    )
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        temperature=0.0,
        max_retries=3,
        timeout=30,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://brightsmile-dental.com",
                "X-Title": "BrightSmile Dental Agent",
            }
        },
    )


@lru_cache(maxsize=1)
def get_classifier_llm() -> ChatOpenAI:
   
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        temperature=0.0,
        max_retries=2,
        timeout=15,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://brightsmile-dental.com",
                "X-Title": "BrightSmile Guardrail Classifier",
            }
        },
    )