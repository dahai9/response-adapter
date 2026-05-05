# deepseek-responses-adapter

Local adapter that lets Codex talk to DeepSeek when Codex requires
`wire_api = "responses"` but the upstream provider exposes Chat Completions.

Codex sends:

```text
POST /v1/responses
```

This adapter calls:

```text
POST https://api.deepseek.com/chat/completions
```

Then it emits Responses-style SSE events back to Codex. The upstream DeepSeek
request uses `stream: true` plus `stream_options.include_usage`.

## Run

```bash
cd /home/dahai003/repo/codex_learn/deepseek-responses-adapter
export DEEPSEEK_API_KEY="..."
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project python deepseek_responses_adapter.py --host 127.0.0.1 --port 8787
```

If your system has Python installed, `python3 deepseek_responses_adapter.py ...`
works too.

Optional environment variables:

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_THINKING="disabled"  # keep disabled for Codex compatibility
export DEEPSEEK_TIMEOUT="120"
```

## Codex Config

Add a provider to `~/.codex/config.toml`:

```toml
model = "deepseek-v4-pro"
model_provider = "deepseek-responses-adapter"

[model_providers.deepseek-responses-adapter]
name = "DeepSeek Responses Adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
requires_openai_auth = false
```

If you prefer profile-scoped usage:

```toml
[profiles.deepseek]
model = "deepseek-v4-pro"
model_provider = "deepseek-responses-adapter"
```

Run Codex with:

```bash
codex -p deepseek
```

## What It Supports

- Responses `message` input to Chat `system` / `user` / `assistant` messages.
- Responses `function_call_output` to Chat `tool` messages.
- Responses `function` tools to Chat `function` tools.
- Responses `namespace` tools flattened into valid Chat function names, then
  expanded back into Responses `function_call` items for Codex.
- DeepSeek streamed `delta.content` to Responses `response.output_text.delta`,
  followed by a final Responses `message` output item.
- DeepSeek streamed tool call chunks to Responses `function_call` output items.

The adapter intentionally does not implement WebSocket transport, image input,
or web search.

Keep DeepSeek thinking mode disabled for Codex use. DeepSeek requires
`reasoning_content` from thinking-mode assistant messages to be passed back on
later requests, but Codex's Responses history does not preserve that field for
this adapter.

## Verify

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project python -m unittest -v
```
