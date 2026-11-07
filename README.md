# LLM Gateway

A minimal MVC-like FastAPI backend scaffold for an LLM gateway.

## Architecture

- `app/controllers` handles HTTP requests
- `app/models` defines request/response schemas
- `app/services` contains business logic
- `app/providers` abstracts LLM backends

## Run locally with uv

```bash
uv sync
```

In PowerShell, copy the example settings and add your OpenAI API key:

```powershell
Copy-Item .env.example .env
# Edit .env and replace the example value with your key.
uv run uvicorn app.main:app --reload --port 8002
```

Keep `.env` private. Do not commit your API key. A request to `/v1/generate` sends its prompt to OpenAI and may incur API charges.
Open Swagger UI at `http://127.0.0.1:8002/docs`.

## Example request

```powershell
$body = @{
    model = "gpt-4o-mini"
    prompt = "Hello from the gateway"
    max_tokens = 128
    temperature = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8002/v1/generate" `
    -Method Post -ContentType "application/json" -Body $body
```

## Notes

The backend uses OpenAI by default. It does not store prompts or responses in a database.
