"""FastAPI application entrypoint.

Run locally:
    uvicorn src.retinocare.api.main:app --reload
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from src.retinocare.api.routes import router

app = FastAPI(
    title="RetinoCare AI",
    description="Diabetic retinopathy screening agent — not a diagnostic tool.",
    version="0.1.0",
)

app.include_router(router)