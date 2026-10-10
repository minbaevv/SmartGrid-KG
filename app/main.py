from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from app.config import get_settings
from app.db import init_db, check_db
from app.api.v1.endpoints import router
from app.api.v1.predict import router as predict_router

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        app.state.http_client = client
        try:
            yield
        finally:
            from app.db import engine
            await engine.dispose()

app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_credentials=False, allow_methods=["GET"],
                   allow_headers=["*"])
app.include_router(router)
app.include_router(predict_router)

@app.get("/healthcheck", tags=["Health"])
async def healthcheck():
    try:
        await check_db()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ok", "stage": 2, "model_artifact_present": settings.model_path.exists(), "mode": "synthetic_demo"}
