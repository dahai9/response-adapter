#!/usr/bin/env python3
"""Local Responses API facade backed by DeepSeek Chat Completions.

Codex now speaks the OpenAI Responses API shape. DeepSeek's public endpoint is
OpenAI-compatible Chat Completions. This adapter accepts a small Responses API
subset from Codex, calls DeepSeek /chat/completions, then emits Responses-style
SSE events back to Codex.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"

JSON = dict[str, Any]


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


def now() -> int:
    return int(time.time())


def response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        if content is not None:
            return as_text(content)
    return json.dumps(value, ensure_ascii=False)


def content_items_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return as_text(content)

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ in {"input_text", "output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif typ == "input_image":
            image_url = item.get("image_url")
            if isinstance(image_url, str):
                parts.append(f"[image: {image_url}]")
    return "\n".join(parts)


def function_output_to_text(output: Any) -> str:
    return as_text(output)


class ToolNameMapper:
    def __init__(self) -> None:
        self._forward: dict[tuple[str | None, str], str] = {}
        self._reverse: dict[str, tuple[str | None, str]] = {}

    @staticmethod
    def _sanitize(name: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
        return name or "tool"

    def _unique_name(self, desired: str) -> str:
        base = self._sanitize(desired)[:64]
        if base not in self._reverse:
            return base
        for index in range(1, 1000):
            suffix = f"_{index}"
            candidate = f"{base[:64 - len(suffix)]}{suffix}"
            if candidate not in self._reverse:
                return candidate
        raise ValueError(f"too many duplicate tool names for {desired!r}")

    def add(self, name: str, namespace: str | None = None) -> str:
        key = (namespace, name)
        if key in self._forward:
            return self._forward[key]
        desired = f"{namespace}__{name}" if namespace else name
        encoded = self._unique_name(desired)
        self._forward[key] = encoded
        self._reverse[encoded] = (namespace, name)
        return encoded

    def decode(self, encoded: str) -> tuple[str | None, str]:
        return self._reverse.get(encoded, (None, encoded))


def normalize_schema(schema: Any) -> JSON:
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


def convert_responses_tools(tools: Any, mapper: ToolNameMapper) -> list[JSON]:
    if not isinstance(tools, list):
        return []

    converted: list[JSON] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        typ = tool.get("type")
        if typ == "function":
            name = tool.get("name")
            if not isinstance(name, str):
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": mapper.add(name),
                        "description": tool.get("description", ""),
                        "parameters": normalize_schema(tool.get("parameters")),
                    },
                }
            )
        elif typ == "namespace":
            namespace = tool.get("name")
            children = tool.get("tools")
            if not isinstance(namespace, str) or not isinstance(children, list):
                continue
            for child in children:
                if not isinstance(child, dict) or child.get("type") != "function":
                    continue
                child_name = child.get("name")
                if not isinstance(child_name, str):
                    continue
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": mapper.add(child_name, namespace=namespace),
                            "description": child.get("description", ""),
                            "parameters": normalize_schema(child.get("parameters")),
                        },
                    }
                )
    return converted


def sanitize_chat_messages(messages: list[JSON]) -> list[JSON]:
    tool_outputs: dict[str, JSON] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            tool_outputs[tool_call_id] = message

    sanitized: list[JSON] = []
    consumed_tool_outputs: set[str] = set()

    for message in messages:
        role = message.get("role")
        if role == "tool":
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            call_ids = [
                tool_call.get("id")
                for tool_call in tool_calls
                if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str)
            ]
            if not call_ids or any(call_id not in tool_outputs for call_id in call_ids):
                continue
            sanitized.append(message)
            for call_id in call_ids:
                sanitized.append(tool_outputs[call_id])
                consumed_tool_outputs.add(call_id)
            continue

        sanitized.append(message)

    return sanitized


def convert_responses_input(input_items: Any, mapper: ToolNameMapper) -> list[JSON]:
    if not isinstance(input_items, list):
        return []

    messages: list[JSON] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ == "message":
            role = item.get("role")
            if role not in {"system", "user", "assistant"}:
                continue
            text = content_items_to_text(item.get("content"))
            if text:
                messages.append({"role": role, "content": text})
        elif typ == "function_call":
            call_id = item.get("call_id") or item_id("call")
            name = item.get("name")
            namespace = item.get("namespace")
            if not isinstance(namespace, str):
                namespace = None
            arguments = item.get("arguments", "{}")
            if isinstance(name, str):
                encoded_name = mapper.add(name, namespace=namespace)
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": str(call_id),
                                "type": "function",
                                "function": {
                                    "name": encoded_name,
                                    "arguments": arguments
                                    if isinstance(arguments, str)
                                    else json.dumps(arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
        elif typ in {"function_call_output", "custom_tool_call_output"}:
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": function_output_to_text(item.get("output")),
                    }
                )
        elif typ == "custom_tool_call":
            call_id = item.get("call_id") or item_id("call")
            name = item.get("name")
            if isinstance(name, str):
                encoded_name = mapper.add(name)
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": str(call_id),
                                "type": "function",
                                "function": {
                                    "name": encoded_name,
                                    "arguments": item.get("input", "{}"),
                                },
                            }
                        ],
                    }
                )
    return messages


def reasoning_effort_for_deepseek(request: JSON) -> str | None:
    reasoning = request.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort")
    if effort == "xhigh":
        return "max"
    if effort in {"low", "medium", "high"}:
        return "high"
    return None


def build_deepseek_body(request: JSON, mapper: ToolNameMapper) -> JSON:
    messages: list[JSON] = []
    instructions = request.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    tools = convert_responses_tools(request.get("tools"), mapper)
    messages.extend(convert_responses_input(request.get("input"), mapper))
    messages = sanitize_chat_messages(messages)
    if not messages:
        messages.append({"role": "user", "content": ""})

    model = env("DEEPSEEK_MODEL") or request.get("model") or DEFAULT_MODEL
    body: JSON = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    effort = reasoning_effort_for_deepseek(request)
    if effort:
        body["reasoning_effort"] = effort

    thinking = env("DEEPSEEK_THINKING")
    if thinking in {"enabled", "disabled"}:
        body["thinking"] = {"type": thinking}

    return body


def codex_usage_from_chat_usage(usage: Any) -> JSON | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    cached = int(usage.get("prompt_cache_hit_tokens") or 0)
    reasoning_tokens = 0
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens,
    }


def response_created(resp_id: str, model: str | None = None) -> JSON:
    response: JSON = {"id": resp_id}
    if model:
        response["headers"] = {"openai-model": model}
    return {"type": "response.created", "response": response}


def response_completed(resp_id: str, usage: JSON | None, end_turn: bool | None) -> JSON:
    response: JSON = {"id": resp_id, "output": []}
    if usage is not None:
        response["usage"] = usage
    if end_turn is not None:
        response["end_turn"] = end_turn
    return {"type": "response.completed", "response": response}


def response_failed(resp_id: str, message: str, code: str = "adapter_error") -> JSON:
    return {
        "type": "response.failed",
        "response": {
            "id": resp_id,
            "error": {
                "type": "server_error",
                "code": code,
                "message": message,
            },
        },
    }


def assistant_message_item(text: str) -> JSON:
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "id": item_id("msg"),
            "content": [{"type": "output_text", "text": text}],
        },
    }


def function_call_item(tool_call: JSON, mapper: ToolNameMapper) -> JSON | None:
    if tool_call.get("type") != "function":
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    encoded_name = function.get("name")
    if not isinstance(encoded_name, str):
        return None
    namespace, name = mapper.decode(encoded_name)
    arguments = function.get("arguments", "{}")
    item: JSON = {
        "type": "function_call",
        "call_id": str(tool_call.get("id") or item_id("call")),
        "name": name,
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
    }
    if namespace:
        item["namespace"] = namespace
    return {"type": "response.output_item.done", "item": item}


def chat_completion_to_response_events(chat: JSON, mapper: ToolNameMapper) -> list[JSON]:
    resp_id = str(chat.get("id") or response_id())
    model = chat.get("model")
    events = [response_created(resp_id, model if isinstance(model, str) else None)]

    choices = chat.get("choices")
    message: JSON = {}
    finish_reason = None
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            maybe_message = first.get("message")
            if isinstance(maybe_message, dict):
                message = maybe_message
            finish_reason = first.get("finish_reason")

    content = message.get("content")
    if isinstance(content, str) and content:
        events.append(assistant_message_item(content))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                event = function_call_item(tool_call, mapper)
                if event is not None:
                    events.append(event)

    usage = codex_usage_from_chat_usage(chat.get("usage"))
    end_turn = False if finish_reason == "tool_calls" else True
    events.append(response_completed(resp_id, usage, end_turn))
    return events


def model_info(slug: str, display_name: str, priority: int) -> JSON:
    return {
        "slug": slug,
        "display_name": display_name,
        "description": "DeepSeek model exposed through deepseek-responses-adapter.",
        "default_reasoning_level": "high",
        "supported_reasoning_levels": [
            {"effort": "high", "description": "DeepSeek high reasoning effort"},
            {"effort": "xhigh", "description": "Mapped to DeepSeek max reasoning effort"},
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "additional_speed_tiers": [],
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": "",
        "supports_reasoning_summaries": False,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": "function",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": False,
        "supports_image_detail_original": False,
        "context_window": 64000,
        "max_context_window": 64000,
        "auto_compact_token_limit": None,
        "effective_context_window_percent": 90,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": False,
    }


def models_response() -> JSON:
    return {
        "models": [
            model_info("deepseek-v4-pro", "DeepSeek V4 Pro", 0),
            model_info("deepseek-v4-flash", "DeepSeek V4 Flash", 1),
            model_info("deepseek-chat", "DeepSeek Chat", 2),
            model_info("deepseek-reasoner", "DeepSeek Reasoner", 3),
        ]
    }


def call_deepseek(body: JSON, timeout: float) -> JSON:
    api_key = env("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    base_url = env("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL) or DEFAULT_DEEPSEEK_BASE_URL
    url = f"{base_url.rstrip('/')}/chat/completions"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {raw}") from exc
    return json.loads(raw)


def open_deepseek_stream(body: JSON, timeout: float) -> Any:
    api_key = env("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    base_url = env("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL) or DEFAULT_DEEPSEEK_BASE_URL
    url = f"{base_url.rstrip('/')}/chat/completions"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {raw}") from exc


def iter_deepseek_stream_chunks(response: Any) -> Any:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        if not payload:
            continue
        yield json.loads(payload)


class StreamingAccumulator:
    def __init__(self, mapper: ToolNameMapper) -> None:
        self.mapper = mapper
        self.resp_id: str | None = None
        self.model: str | None = None
        self.content_parts: list[str] = []
        self.tool_calls: dict[int, JSON] = {}
        self.finish_reason: str | None = None
        self.usage: JSON | None = None

    def ingest(self, chunk: JSON) -> list[JSON]:
        if self.resp_id is None:
            self.resp_id = str(chunk.get("id") or response_id())
        if self.model is None and isinstance(chunk.get("model"), str):
            self.model = chunk["model"]

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = codex_usage_from_chat_usage(usage)

        events: list[JSON] = []
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return events

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self.finish_reason = finish_reason
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            content = delta.get("content")
            if isinstance(content, str) and content:
                self.content_parts.append(content)
                events.append({"type": "response.output_text.delta", "delta": content})

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        self._ingest_tool_call_delta(tool_call)

        return events

    def _ingest_tool_call_delta(self, tool_call: JSON) -> None:
        index = tool_call.get("index")
        if not isinstance(index, int):
            index = len(self.tool_calls)
        current = self.tool_calls.setdefault(
            index,
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if isinstance(tool_call.get("id"), str):
            current["id"] = tool_call["id"]
        if isinstance(tool_call.get("type"), str):
            current["type"] = tool_call["type"]
        function = tool_call.get("function")
        if isinstance(function, dict):
            current_function = current.setdefault("function", {"name": "", "arguments": ""})
            name = function.get("name")
            if isinstance(name, str):
                current_function["name"] = current_function.get("name", "") + name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                current_function["arguments"] = current_function.get("arguments", "") + arguments

    def final_events(self) -> list[JSON]:
        resp_id = self.resp_id or response_id()
        events: list[JSON] = []
        content = "".join(self.content_parts)
        if content:
            events.append(assistant_message_item(content))

        for index in sorted(self.tool_calls):
            tool_call = self.tool_calls[index]
            event = function_call_item(tool_call, self.mapper)
            if event is not None:
                events.append(event)

        end_turn = False if self.finish_reason == "tool_calls" else True
        events.append(response_completed(resp_id, self.usage, end_turn))
        return events


def sse_frame(event: JSON) -> bytes:
    kind = str(event["type"])
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"event: {kind}\ndata: {data}\n\n".encode("utf-8")


class AdapterHandler(BaseHTTPRequestHandler):
    server_version = "deepseek-responses-adapter/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if env("ADAPTER_QUIET", "0") == "1":
            return
        super().log_message(fmt, *args)

    def _send_json(self, status: int, body: JSON) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse_events(self, events: list[JSON]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(sse_frame(event))
            self.wfile.flush()

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _write_sse(self, event: JSON) -> None:
        self.wfile.write(sse_frame(event))
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz"}:
            self._send_json(200, {"ok": True})
        elif self.path in {"/models", "/v1/models"}:
            self._send_json(200, models_response())
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        if self.path not in {"/responses", "/v1/responses"}:
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return

        resp_id = response_id()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body_bytes = self.rfile.read(length)
            request_body = json.loads(body_bytes.decode("utf-8"))
            if not isinstance(request_body, dict):
                raise ValueError("request body must be a JSON object")

            mapper = ToolNameMapper()
            deepseek_body = build_deepseek_body(request_body, mapper)
            upstream = open_deepseek_stream(
                deepseek_body,
                timeout=float(env("DEEPSEEK_TIMEOUT", "120") or "120"),
            )
            self._start_sse()
            accumulator = StreamingAccumulator(mapper)
            created_sent = False
            try:
                for chunk in iter_deepseek_stream_chunks(upstream):
                    if not created_sent:
                        events = accumulator.ingest(chunk)
                        self._write_sse(
                            response_created(
                                accumulator.resp_id or resp_id,
                                accumulator.model,
                            )
                        )
                        created_sent = True
                        for event in events:
                            self._write_sse(event)
                    else:
                        for event in accumulator.ingest(chunk):
                            self._write_sse(event)
                if not created_sent:
                    self._write_sse(response_created(resp_id))
                for event in accumulator.final_events():
                    self._write_sse(event)
            finally:
                upstream.close()
        except Exception as exc:
            if env("ADAPTER_DEBUG", "0") == "1":
                traceback.print_exc(file=sys.stderr)
            self._send_sse_events([response_created(resp_id), response_failed(resp_id, str(exc))])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=env("ADAPTER_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(env("ADAPTER_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    server = ThreadingHTTPServer((args.host, args.port), AdapterHandler)
    print(f"deepseek-responses-adapter listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
