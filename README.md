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
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and replace the example value with your key.
uv run uvicorn app.main:app --reload --port 8003
```

Keep `.env` private. Do not commit your API key. A request to `/v1/generate`
sends its prompt to the selected provider and may incur API charges.
Open Swagger UI at `http://127.0.0.1:8003/docs`.

### Use Google Gemini

Set `LLM_PROVIDER=google` and put your Gemini API key in `GEMINI_API_KEY` in
`.env`. Keep the OpenAI key out of the file if you are not using OpenAI. Start
the app with the command above, open Swagger, and use a Gemini model available
to your API key (for example, `gemini-3.8-flash`) in the request's `model`
field. The gateway defaults to OpenAI when `LLM_PROVIDER` is not set.

Both providers support inline JSON output schemas. Google requests use JSON
mode with the supplied schema and the gateway validates the returned JSON.

## Example request

```powershell
$body = @{
    model = "gpt-4o-mini"
    prompt = "Hello from the gateway"
    max_tokens = 128
    temperature = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8003/v1/generate" `
    -Method Post -ContentType "application/json" -Body $body
```

## Notes

The backend uses OpenAI by default and supports Google Gemini as an alternative.
It does not store prompts or responses in a database.

Invalid request data returns FastAPI's validation response. Provider errors use
safe gateway messages: timeouts return `504`, provider configuration and rate
limit errors return `503`, and other provider failures return `502`. Provider
error details are not included in the client response.
Gateway-specific retry policies and timeout configuration remain future work.
Optional inline output schemas are checked as JSON Schema Draft 2020-12 during
request validation and are passed to the selected provider for structured
generation. The gateway parses the generated JSON and validates it against the
schema before returning it. Invalid structured output returns a safe `502`
response.
Output schema IDs are accepted by request validation but return `501 Not
Implemented` until a schema registry is available.

Run the unit tests with:

```powershell
uv run python -m unittest discover -s tests -v
```
