# responses-adapter

[![CI](https://github.com/YOUR_USERNAME/responses-adapter/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/responses-adapter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight Rust adapter that translates the OpenAI **Responses API** into
standard **Chat Completions API** requests. This lets tools like Codex (which
speak the Responses wire protocol) work with any OpenAI-compatible provider.

## Architecture

```text
Codex / Client                    responses-adapter                Upstream Provider
    │                                    │                                │
    │  POST /v1/responses                │                                │
    │  (Responses API format)            │                                │
    ├──────────────────────────────────►│                                │
    │                                    │  POST /chat/completions        │
    │                                    │  (Chat Completions format)     │
    │                                    ├───────────────────────────────►│
    │                                    │                                │
    │                                    │  SSE stream (deltas)           │
    │                                    │◄───────────────────────────────┤
    │                                    │                                │
    │  SSE stream                        │  (translated to Responses      │
    │  (Responses events)                │   events: output_text.delta,   │
    │◄──────────────────────────────────┤   function_call, etc.)         │
    │                                    │                                │
```

## Quick Start

1. **Clone and configure:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/responses-adapter.git
   cd responses-adapter
   cp .env.example .env
   # Edit .env with your API key and upstream URL
   ```

2. **Run:**

   ```bash
   cargo run
   ```

3. **Configure Codex** (`~/.codex/config.toml`):

   ```toml
   model = "gpt-4o"
   model_provider = "responses-adapter"

   [model_providers.responses-adapter]
   name = "Responses Adapter"
   base_url = "http://127.0.0.1:8787/v1"
   wire_api = "responses"
   env_key = "UPSTREAM_API_KEY"
   ```

## Building from Source

**Prerequisites:** Rust 1.75+ (install via [rustup](https://rustup.rs/))

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format check
cargo fmt -- --check
```

## Docker

Build and run with Docker:

```bash
docker build -t responses-adapter .

docker run -p 8787:8787 \
  -e UPSTREAM_API_KEY=sk-your-key \
  -e UPSTREAM_BASE_URL=https://api.openai.com/v1 \
  responses-adapter
```

Or with a `.env` file:

```bash
docker run -p 8787:8787 --env-file .env responses-adapter
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
| `ADAPTER_DEBUG_BODY` | No | Set to `1` to log converted request body (debug) |

See [`.env.example`](.env.example) for a fully documented template.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/responses` | Translate Responses API request to Chat Completions |
| `GET` | `/v1/models` | List configured models |
| `GET` | `/health` | Health check |

## Translation Rules

- Responses `instructions` becomes a Chat `system` message.
- Responses `message` input becomes Chat `system`, `user`, or `assistant` messages.
- Responses `function_call_output` becomes a Chat `tool` message.
- Responses `function` tools become Chat `function` tools.
- Namespaced tools are flattened into Chat-safe function names and decoded back when streamed.
- Streamed `delta.content` becomes `response.output_text.delta`.
- Streamed tool calls are accumulated and emitted as Responses `function_call` items.
- Assistant tool-call history is only forwarded when every `tool_call_id` has a matching tool result.

## Thinking / Reasoning

Codex has `model_reasoning_effort`; the adapter forwards that as `reasoning_effort` in the upstream request.

Some providers (e.g. DeepSeek) support a `thinking.enabled` protocol mode. In thinking mode with tool calls, the provider may require the assistant message's `reasoning_content` to be passed back during the same tool-call sub-turn. This adapter keeps an in-memory reasoning store keyed by tool call id and can replay that field for the tool-result continuation request.

Enable with:

```bash
ADAPTER_THINKING=enabled
```

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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes and add tests
4. Ensure `cargo fmt`, `cargo clippy`, and `cargo test` all pass
5. Commit and push
6. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).
