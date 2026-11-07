from fastapi import FastAPI

from app.controllers.generate_controller import router as generate_router

# Start the API app.
app = FastAPI(title="LLM Gateway", version="0.1.0")

# Put generation routes under /v1.
app.include_router(generate_router, prefix="/v1")


@app.get("/health")
def health_check():
    """Tell us the API is running."""
    return {"status": "ok"}
