from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.controllers.generate_controller import router as generate_router
from app.providers.base_provider import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.services.generation_service import (
    InvalidStructuredOutputError,
    UnsupportedOutputSchemaIdError,
)

# Start the API app.
app = FastAPI(title="LLM Gateway", version="0.1.0")

# Put generation routes under /v1.
app.include_router(generate_router, prefix="/v1")


@app.exception_handler(UnsupportedOutputSchemaIdError)
async def handle_unsupported_output_schema_id(
    _request: Request,
    _error: UnsupportedOutputSchemaIdError,
) -> JSONResponse:
    """Return a clear response for an unavailable schema registry."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Output schema IDs are not supported yet."},
    )


@app.exception_handler(InvalidStructuredOutputError)
async def handle_invalid_structured_output(
    _request: Request,
    _error: InvalidStructuredOutputError,
) -> JSONResponse:
    """Return a safe response for invalid generated JSON."""
    return JSONResponse(
        status_code=502,
        content={"detail": "The LLM provider returned invalid structured output."},
    )


@app.exception_handler(ProviderError)
async def handle_provider_error(
    _request: Request,
    error: ProviderError,
) -> JSONResponse:
    """Return a safe response for provider errors."""
    if isinstance(error, ProviderConfigurationError):
        status_code = 503
        message = "The LLM provider is not configured."
    elif isinstance(error, ProviderTimeoutError):
        status_code = 504
        message = "The LLM provider request timed out."
    elif isinstance(error, ProviderRateLimitError):
        status_code = 503
        message = "The LLM provider is temporarily rate limited."
    else:
        status_code = 502
        message = "The LLM provider request failed."

    return JSONResponse(
        status_code=status_code,
        content={"detail": message},
    )


@app.get("/health")
def health_check():
    """Tell us the API is running."""
    return {"status": "ok"}
