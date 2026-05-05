# responses-adapter

Rust adapter that lets Codex use any OpenAI-compatible Chat Completions API
when Codex is configured with `wire_api = "responses"`.

The adapter accepts:

```text
POST /v1/responses
```

and forwards a translated streaming request to:

```text
POST {UPSTREAM_BASE_URL}/chat/completions
```

It then streams Responses-style SSE events back to Codex.

## Run

Create `.env` in this directory:

```bash
UPSTREAM_API_KEY=sk-...
UPSTREAM_BASE_URL=https://api.openai.com/v1
ADAPTER_MODEL=gpt-4o
ADAPTER_TIMEOUT=120
ADAPTER_HOST=127.0.0.1
ADAPTER_PORT=8787
```

Start the adapter:

```bash
cargo run
```

## Codex Config

Add this to `~/.codex/config.toml`:

```toml
model = "gpt-4o"
model_provider = "responses-adapter"

[model_providers.responses-adapter]
name = "Responses Adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
env_key = "UPSTREAM_API_KEY"
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `UPSTREAM_API_KEY` | Yes | API key for the upstream provider |
| `UPSTREAM_BASE_URL` | Yes | Base URL of the upstream Chat Completions endpoint |
| `ADAPTER_MODEL` | No | Fixed model override (bypasses model map) |
| `ADAPTER_MODEL_MAP` | No | JSON map of incoming model names to upstream models |
| `ADAPTER_THINKING` | No | Set to `enabled` to send thinking/reasoning fields |
| `ADAPTER_TIMEOUT` | No | Upstream request timeout in seconds (default: 120) |
| `ADAPTER_HOST` | No | Listen host (default: 127.0.0.1) |
| `ADAPTER_PORT` | No | Listen port (default: 8787) |
| `ADAPTER_MODELS` | No | JSON array of model objects for the `/v1/models` endpoint |
| `ADAPTER_DEBUG_BODY` | No | Set to `1` to print converted request body to stderr |

## Translation Rules

- Responses `instructions` becomes a Chat `system` message.
- Responses `message` input becomes Chat `system`, `user`, or `assistant`
  messages.
- Responses `function_call_output` becomes a Chat `tool` message.
- Responses `function` tools become Chat `function` tools.
- Namespaced tools are flattened into Chat-safe function names and decoded back
  when streamed to Codex.
- Streamed `delta.content` becomes `response.output_text.delta`.
- Streamed tool calls are accumulated and emitted as Responses `function_call`
  items.
- Assistant tool-call history is only forwarded when every `tool_call_id` has a
  matching tool result, which avoids chat sequencing errors.

## Thinking / Reasoning

Codex has `model_reasoning_effort`; the adapter forwards that as
`reasoning_effort` in the upstream request.

Some providers (e.g. DeepSeek) support a `thinking.enabled` protocol mode. In
thinking mode with tool calls, the provider may require the assistant message's
`reasoning_content` to be passed back during the same tool-call sub-turn. This
adapter keeps an in-memory reasoning store keyed by tool call id and can replay
that field for the tool-result continuation request.

Enable it with:

```bash
ADAPTER_THINKING=enabled
```

Leaving `ADAPTER_THINKING` unset is the safest default. If set to `disabled`,
the adapter does not send the `thinking` field but remains compatible with
`model_reasoning_effort`.

## Model Mapping

Map incoming model names to upstream models via `ADAPTER_MODEL_MAP`:

```bash
ADAPTER_MODEL_MAP='{"gpt-4o":"provider-model-a","gpt-4o-mini":"provider-model-b"}'
```

**Resolution priority** (first match wins):
1. `ADAPTER_MODEL_MAP` lookup on the incoming request's `model`
2. `ADAPTER_MODEL` override (if set)
3. The incoming request's `model` field as-is

## Models Endpoint

Configure the `/v1/models` response with `ADAPTER_MODELS`:

```bash
ADAPTER_MODELS='[{"id":"model-a","name":"Model A"},{"id":"model-b","name":"Model B"}]'
```

If not set, the endpoint returns an empty list.

## Development

```bash
cargo fmt
cargo test
```
