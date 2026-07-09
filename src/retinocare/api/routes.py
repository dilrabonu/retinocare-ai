"""API routes: /predict accepts an image, returns classification + agent recommendation."""

import io
from functools import lru_cache

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image

from src.retinocare.agents.rag_agent import RAGAgent
from src.retinocare.api.schemas import PredictionResponse
from src.retinocare.data.transforms import get_eval_transforms

router = APIRouter()

SEVERITY_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CHECKPOINT_PATH = "models/resnet18.pt"
KNOWLEDGE_BASE_DIR = "knowledge_base/guidelines"


@lru_cache(maxsize=1)
def _load_model():
    """Loads the trained checkpoint once and caches it (lru_cache) so every
    request doesn't reload weights from disk."""
    from src.retinocare.models.resnet_transfer import build_resnet18

    model = build_resnet18(num_classes=len(SEVERITY_LABELS), freeze_backbone=False)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@lru_cache(maxsize=1)
def _load_agent() -> RAGAgent:
    return RAGAgent(KNOWLEDGE_BASE_DIR)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG image.")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    transform = get_eval_transforms(image_size=224)
    import numpy as np

    tensor = transform(image=np.array(image))["image"].unsqueeze(0)  # (1, 3, 224, 224)

    model = _load_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    severity_index = int(torch.argmax(probs).item())
    confidence = float(probs[severity_index].item())
    severity_label = SEVERITY_LABELS[severity_index]

    agent = _load_agent()
    agent_result = agent.respond(severity_label, confidence)

    return PredictionResponse(
        severity_label=severity_label,
        severity_index=severity_index,
        confidence=confidence,
        agent_recommendation=agent_result["text"],
        sources=agent_result["sources"],
        disclaimer="This is a screening support tool, not a medical diagnosis. "
        "Please consult a qualified ophthalmologist for evaluation and care decisions.",
    )