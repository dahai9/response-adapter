# deepseek-responses-adapter

Rust adapter that lets Codex use DeepSeek when Codex is configured with
`wire_api = "responses"` but the upstream only exposes OpenAI-compatible Chat
Completions.

The adapter accepts:

```text
POST /v1/responses
```

and forwards a translated streaming request to:

```text
POST https://api.deepseek.com/chat/completions
```

It then streams Responses-style SSE events back to Codex.

## Run

Create `.env` in this directory:

```bash
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_TIMEOUT=120
ADAPTER_HOST=127.0.0.1
ADAPTER_PORT=8787
```

For flash-only testing, set:

```bash
DEEPSEEK_MODEL=deepseek-v4-flash
```

Start the adapter:

```bash
cargo run
```

## Codex Config

Add this to `~/.codex/config.toml`:

```toml
model = "deepseek-v4-pro"
model_provider = "deepseek-responses-adapter"

[model_providers.deepseek-responses-adapter]
name = "DeepSeek Responses Adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
env_key = "DEEPSEEK_API_KEY"

[profiles.deepseek]
model = "deepseek-v4-pro"
model_provider = "deepseek-responses-adapter"
model_reasoning_effort = "high"
```

Then run:

```bash
codex -p deepseek
```

## Translation Rules

- Responses `instructions` becomes a Chat `system` message.
- Responses `message` input becomes Chat `system`, `user`, or `assistant`
  messages.
- Responses `function_call_output` becomes a Chat `tool` message.
- Responses `function` tools become Chat `function` tools.
- Namespaced tools are flattened into Chat-safe function names and decoded back
  when streamed to Codex.
- DeepSeek streamed `delta.content` becomes `response.output_text.delta`.
- DeepSeek streamed tool calls are accumulated and emitted as Responses
  `function_call` items.
- Assistant tool-call history is only forwarded when every `tool_call_id` has a
  matching tool result, which avoids DeepSeek/OpenAI chat sequencing errors.

## Thinking Mode

Codex already has `model_reasoning_effort`; the adapter forwards that as
DeepSeek `reasoning_effort` where possible.

DeepSeek `thinking.enabled` is a separate protocol mode. In thinking mode with
tool calls, DeepSeek requires the assistant message's `reasoning_content` to be
passed back during the same tool-call sub-turn. This Rust adapter keeps an
in-memory reasoning store keyed by tool call id and can replay that field for
the tool-result continuation request.

Enable it only when you need DeepSeek thinking semantics:

```bash
DEEPSEEK_THINKING=enabled
```

For normal Codex usage, leaving `DEEPSEEK_THINKING` unset is the safest default.
If it is set to `disabled`, the adapter treats that as "do not send the
DeepSeek `thinking` field" so it remains compatible with Codex
`model_reasoning_effort`.

## Development

```bash
cargo fmt
cargo test
```

## Model Mapping

The adapter supports mapping incoming model names to upstream DeepSeek models
via `DEEPSEEK_MODEL_MAP`. This lets Codex use its built-in model names (e.g.
`gpt-5.4`, `gpt-5.5`) while the adapter transparently routes to the correct
DeepSeek model.

```bash
DEEPSEEK_MODEL_MAP='{"gpt-5.4":"deepseek-v4-flash","gpt-5.5":"deepseek-v4-pro"}'
```

**Resolution priority** (first match wins):
1. `DEEPSEEK_MODEL_MAP` lookup on the incoming request's `model`
2. `DEEPSEEK_MODEL` override (if set)
3. The incoming request's `model` field as-is
4. `deepseek-v4-pro` (default)

If `DEEPSEEK_MODEL` is set, it acts as a fixed override that bypasses the map.
For per-model routing, set `DEEPSEEK_MODEL_MAP` and leave `DEEPSEEK_MODEL`
unset.

### Example: Codex with model-aware routing

```toml
[model_providers.deepseek-responses-adapter]
name = "DeepSeek Responses Adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
```

Then set in `.env`:

```bash
DEEPSEEK_MODEL_MAP={"gpt-5.4":"deepseek-v4-flash","gpt-5.5":"deepseek-v4-pro"}
```

Codex requests with `model = "gpt-5.4"` will hit `deepseek-v4-flash`, and
requests with `model = "gpt-5.5"` will hit `deepseek-v4-pro`.
