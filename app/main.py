

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.logger import get_logger
from app.config.settings import get_settings
from app.config.llm import get_llm, get_classifier_llm
from app.rag.retriever import get_vectorstore
from app.agents.graph import build_graph
from app.api.chat import router as chat_router
from app.api.appointments import router as appointments_router


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    settings = get_settings()
    logger.info("startup_begin", clinic=settings.clinic_name)

    # Pre-load LLM client
    get_llm()
    get_classifier_llm()

    # Pre-load FAISS index
    get_vectorstore()

    # Pre-load graph
    build_graph()

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.clinic_name} — AI Receptionist",
        description="Multilingual dental clinic booking agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # Register routers
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    app.include_router(appointments_router, prefix="/api/v1", tags=["appointments"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "clinic": settings.clinic_name}

    return app


app = create_app()

