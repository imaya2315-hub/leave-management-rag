from fastapi import FastAPI

from app import models  # noqa: F401 - import ensures models register on Base.metadata
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import Base, engine
from app.middleware.auth import RequestLoggingMiddleware

# Create all database tables if they don't already exist.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend for managing employee leaves",
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware)


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "The Leave Management API is up and running."}


app.include_router(api_router, prefix=settings.API_V1_PREFIX)
