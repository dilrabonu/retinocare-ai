"""Pydantic request/response models for the FastAPI service."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    severity_label: str
    severity_index: int
    confidence: float
    agent_recommendation: str
    sources: list[str]
    disclaimer: str