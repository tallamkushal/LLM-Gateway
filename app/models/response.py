from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    """Describe the data the API sends back."""

    # Show which model and prompt were used.
    model: str
    prompt: str
    # Show the answer and the provider that made it.
    response: str
    provider: str = Field(default="openai")
    # Show token counts reported by the provider, if available.
    usage: dict[str, int | None]
