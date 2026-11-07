from functools import lru_cache

from fastapi import APIRouter

from app.models.request import GenerateRequest
from app.models.response import GenerateResponse
from app.services.generation_service import GenerationService

# Keep generation routes together.
router = APIRouter(tags=["generation"])


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    """Create the service once and reuse its OpenAI client."""
    return GenerationService()


@router.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    """Send the request to the service and return its result."""
    # The service does the generation work.
    return get_generation_service().generate(request)
