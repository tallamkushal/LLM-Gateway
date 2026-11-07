from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Describe the data a client sends to generate text."""

    # Use this model if the client does not choose one.
    model: str = Field(default="gpt-4o-mini")
    # Require a prompt with at least one character.
    prompt: str = Field(..., min_length=1)
    # Allow between 1 and 4096 output tokens.
    max_tokens: int = Field(default=256, ge=1, le=4096)
    # Keep temperature between 0 and 2.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject blank prompts and remove spaces at the ends."""
        if not value.strip():
            raise ValueError("prompt cannot be empty")
        return value.strip()
