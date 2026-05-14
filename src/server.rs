use std::collections::HashSet;
use std::convert::Infallible;
use std::error::Error as StdError;
use std::sync::Arc;

use async_stream::stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::header::{ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::adapter::{
    build_chat_body, response_created, response_failed, response_id, ReasoningStore,
    StreamingAccumulator,
};
use crate::config::Config;

#[derive(Clone)]
pub struct AppState {
    config: Config,
    client: reqwest::Client,
    reasoning_store: Arc<Mutex<ReasoningStore>>,
    trace_state: Arc<Mutex<TraceState>>,
}

pub fn router(config: Config) -> anyhow::Result<Router> {
    let client = reqwest::Client::builder()
        // Do not set a total request timeout for SSE: long model generations can
        // legitimately stream for several minutes. A total timeout aborts the
        // body with `error decoding response body <- operation timed out`.
        .connect_timeout(config.connect_timeout)
        .http1_only()
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .build()?;
    let state = AppState {
        config,
        client,
        reasoning_store: Arc::new(Mutex::new(ReasoningStore::default())),
        trace_state: Arc::new(Mutex::new(TraceState::default())),
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/responses", post(responses))
        .route("/v1/responses", post(responses))
        .with_state(state))
}

pub async fn run(config: Config) -> anyhow::Result<()> {
    let listen = config.listen;
    let app = router(config)?;
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!("responses-adapter listening on http://{listen}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({"ok": true}))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": state.config.models
    }))
}

async fn responses(State(state): State<AppState>, Json(request): Json<Value>) -> Response {
    let resp_id = response_id();
    let req_model = request.get("model").and_then(Value::as_str).unwrap_or("-");
    let tools_count = request
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    let input_count = request
        .get("input")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    tracing::info!(
        resp_id = %resp_id,
        model = %req_model,
        tools = tools_count,
        inputs = input_count,
        "incoming request"
    );

    let converted = {
        let store = state.reasoning_store.lock().await;
        match build_chat_body(&request, &state.config, &store) {
            Ok(converted) => converted,
            Err(error) => {
                tracing::error!(resp_id = %resp_id, error = %error, "failed to build chat body");
                let events = vec![
                    response_created(&resp_id, None),
                    response_failed(&resp_id, &error.to_string()),
                ];
                return sse_from_events(events).into_response();
            }
        }
    };
    if debug_body_enabled() {
        emit_debug_json(&resp_id, "converted.body", &converted.body);
    }
    trace_request_summary(&state.trace_state, &resp_id, &converted.body).await;

    let upstream_url = format!(
        "{}/chat/completions",
        state.config.base_url.trim_end_matches('/')
    );
    tracing::debug!(resp_id = %resp_id, url = %upstream_url, upstream_model = %converted.body["model"], "opening upstream stream");

    let upstream = match open_upstream_stream(&state, &converted.body).await {
        Ok(upstream) => upstream,
        Err(error) => {
            tracing::error!(resp_id = %resp_id, error = %error, "upstream request failed");
            let events = vec![
                response_created(&resp_id, None),
                response_failed(&resp_id, &error.to_string()),
            ];
            return sse_from_events(events).into_response();
        }
    };

    let store = state.reasoning_store.clone();
    let trace_state = state.trace_state.clone();
    let stream = stream! {
        let mut accumulator = StreamingAccumulator::new(converted.mapper);
        let mut created_sent = false;
        let mut upstream = upstream;
        let mut decoder = SseDecoder::default();

        while let Some(next) = upstream.next().await {
            match next {
                Ok(chunk) => {
                    match decoder.push(&chunk) {
                        Ok(chunks) => {
                            for json_chunk in chunks {
                                trace_upstream_chunk(&resp_id, &json_chunk);
                                let events = accumulator.ingest(&json_chunk);
                                if !created_sent {
                                    created_sent = true;
                                    // Do not synthesize OpenAI-Model from the mapped upstream model.
                                    // Codex treats that header as a server-side reroute signal.
                                    let ev = response_created(accumulator.resp_id().unwrap_or(&resp_id), None);
                                    trace_response_event(&trace_state, &resp_id, &ev).await;
                                    yield event(ev);
                                }
                                for ev in events {
                                    trace_response_event(&trace_state, &resp_id, &ev).await;
                                    yield event(ev);
                                }
                            }
                        }
                        Err(error) => {
                            let error_detail = anyhow_error_detail(&error);
                            tracing::error!(resp_id = %resp_id, error = %error, error_detail = %error_detail, "SSE decode error");
                            if accumulator.has_stream_progress() {
                                let completed_resp_id = accumulator
                                    .resp_id()
                                    .map(ToOwned::to_owned)
                                    .unwrap_or_else(|| resp_id.clone());
                                tracing::warn!(
                                    resp_id = %completed_resp_id,
                                    error = %error,
                                    error_detail = %error_detail,
                                    "finalizing partial response after SSE decode error"
                                );
                                if !created_sent {
                                    let ev = response_created(&completed_resp_id, None);
                                    trace_response_event(&trace_state, &resp_id, &ev).await;
                                    yield event(ev);
                                }
                                let mut store = store.lock().await;
                                for ev in accumulator.final_events_after_interruption(&mut store) {
                                    trace_response_event(&trace_state, &resp_id, &ev).await;
                                    yield event(ev);
                                }
                                return;
                            }
                            let ev = response_failed(accumulator.resp_id().unwrap_or(&resp_id), &error.to_string());
                            trace_response_event(&trace_state, &resp_id, &ev).await;
                            yield event(ev);
                            return;
                        }
                    }
                }
                Err(error) => {
                    let error_detail = reqwest_error_detail(&error);
                    tracing::error!(resp_id = %resp_id, error = %error, error_detail = %error_detail, "upstream stream error");
                    if accumulator.has_stream_progress() {
                        let completed_resp_id = accumulator
                            .resp_id()
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| resp_id.clone());
                        tracing::warn!(
                            resp_id = %completed_resp_id,
                            error = %error,
                            error_detail = %error_detail,
                            "finalizing partial response after upstream stream error"
                        );
                        if !created_sent {
                            let ev = response_created(&completed_resp_id, None);
                            trace_response_event(&trace_state, &resp_id, &ev).await;
                            yield event(ev);
                        }
                        let mut store = store.lock().await;
                        for ev in accumulator.final_events_after_interruption(&mut store) {
                            trace_response_event(&trace_state, &resp_id, &ev).await;
                            yield event(ev);
                        }
                        return;
                    }
                    let ev = response_failed(accumulator.resp_id().unwrap_or(&resp_id), &error.to_string());
                    trace_response_event(&trace_state, &resp_id, &ev).await;
                    yield event(ev);
                    return;
                }
            }
        }

        if !created_sent {
            let ev = response_created(&resp_id, None);
            trace_response_event(&trace_state, &resp_id, &ev).await;
            yield event(ev);
        }
        let mut store = store.lock().await;
        for ev in accumulator.final_events(&mut store) {
            trace_response_event(&trace_state, &resp_id, &ev).await;
            yield event(ev);
        }
        tracing::info!(resp_id = %resp_id, "stream completed");
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn open_upstream_stream(
    state: &AppState,
    body: &Value,
) -> Result<impl Stream<Item = reqwest::Result<Bytes>>, AdapterHttpError> {
    let url = format!(
        "{}/chat/completions",
        state.config.base_url.trim_end_matches('/')
    );
    let response = state
        .client
        .post(url)
        .header(AUTHORIZATION, format!("Bearer {}", state.config.api_key))
        .header(ACCEPT, "text/event-stream")
        .header(ACCEPT_ENCODING, "identity")
        .header(CONTENT_TYPE, "application/json")
        .json(body)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(AdapterHttpError::Upstream(status, text));
    }

    Ok(response.bytes_stream())
}

fn sse_from_events(events: Vec<Value>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Sse::new(futures_util::stream::iter(events.into_iter().map(event)))
}

fn event(value: Value) -> Result<Event, Infallible> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message")
        .to_string();
    Ok(Event::default()
        .event(kind)
        .json_data(value)
        .expect("serializable SSE event"))
}

fn reqwest_error_detail(error: &reqwest::Error) -> String {
    let mut tags = Vec::new();
    if error.is_timeout() {
        tags.push("timeout");
    }
    if error.is_connect() {
        tags.push("connect");
    }
    if error.is_body() {
        tags.push("body");
    }
    if error.is_decode() {
        tags.push("decode");
    }
    if let Some(status) = error.status() {
        tags.push(match status.as_u16() {
            400..=499 => "http_4xx",
            500..=599 => "http_5xx",
            _ => "http_status",
        });
    }

    let mut detail = if tags.is_empty() {
        "kind=unknown".to_string()
    } else {
        format!("kind={}", tags.join("|"))
    };
    if let Some(status) = error.status() {
        detail.push_str(&format!(" status={status}"));
    }
    if let Some(url) = error.url() {
        detail.push_str(&format!(" url={url}"));
    }
    let chain = error_chain(error);
    if !chain.is_empty() {
        detail.push_str(" chain=");
        detail.push_str(&chain);
    }
    detail
}

fn anyhow_error_detail(error: &anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" <- ")
}

fn error_chain(error: &dyn StdError) -> String {
    let mut parts = vec![error.to_string()];
    let mut source = error.source();
    while let Some(error) = source {
        parts.push(error.to_string());
        source = error.source();
    }
    parts.join(" <- ")
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn debug_body_enabled() -> bool {
    env_flag("ADAPTER_DEBUG_BODY")
}

fn trace_enabled() -> bool {
    env_flag("ADAPTER_DEBUG_TRACE")
}

fn stream_trace_enabled() -> bool {
    env_flag("ADAPTER_DEBUG_STREAM")
}

fn think_trace_enabled() -> bool {
    env_flag("ADAPTER_DEBUG_THINK")
}

fn trace_max_chars() -> Option<usize> {
    match std::env::var("ADAPTER_DEBUG_TRACE_MAX_CHARS") {
        Ok(value) if value == "0" || value.eq_ignore_ascii_case("none") => None,
        Ok(value) => value.parse::<usize>().ok(),
        Err(_) => Some(4000),
    }
}

async fn trace_request_summary(trace_state: &Arc<Mutex<TraceState>>, resp_id: &str, body: &Value) {
    if !trace_enabled() {
        return;
    }

    let sections = request_trace_sections(body);
    let new_sections = {
        let mut trace_state = trace_state.lock().await;
        trace_state.unseen_sections(sections)
    };
    for (label, body) in new_sections {
        emit_debug_section(resp_id, &label, &body);
    }
}

fn trace_upstream_chunk(resp_id: &str, chunk: &Value) {
    if stream_trace_enabled() {
        for (label, body) in upstream_trace_sections(chunk) {
            emit_debug_section(resp_id, &label, &body);
        }
    }
}

async fn trace_response_event(trace_state: &Arc<Mutex<TraceState>>, resp_id: &str, value: &Value) {
    if !trace_enabled() {
        return;
    }

    let sections = response_event_trace_sections(value);
    let new_sections = {
        let mut trace_state = trace_state.lock().await;
        trace_state.unseen_sections(sections)
    };
    for (label, body) in new_sections {
        emit_debug_section(resp_id, &label, &body);
    }
}

fn emit_debug_json(resp_id: &str, label: &str, value: &Value) {
    let rendered = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    eprintln!("[responses-adapter][{resp_id}][{label}] {rendered}");
    tracing::debug!(resp_id = %resp_id, label = label, body = %rendered, "adapter debug");
}

fn emit_debug_section(resp_id: &str, label: &str, body: &str) {
    if body.trim().is_empty() {
        return;
    }
    let body = render_trace_body(body);
    let rendered = format!("=======================\n{label}:\n{body}");
    eprintln!("{rendered}");
    tracing::debug!(resp_id = %resp_id, label = label, body = %body, "adapter trace");
}

fn render_trace_body(body: &str) -> String {
    let Some(max_chars) = trace_max_chars() else {
        return body.to_string();
    };
    let total = body.chars().count();
    if total <= max_chars {
        return body.to_string();
    }
    let mut rendered = body.chars().take(max_chars).collect::<String>();
    rendered.push_str(&format!(
        "\n... [trace truncated: {total} chars total; set ADAPTER_DEBUG_TRACE_MAX_CHARS=0 to show all]"
    ));
    rendered
}

#[derive(Default)]
struct TraceState {
    emitted_sections: HashSet<String>,
}

impl TraceState {
    fn unseen_sections(&mut self, current: Vec<(String, String)>) -> Vec<(String, String)> {
        current
            .into_iter()
            .filter(|(label, body)| self.emitted_sections.insert(trace_fingerprint(label, body)))
            .collect()
    }
}

fn trace_fingerprint(label: &str, body: &str) -> String {
    if label == "agent_toolcall" {
        let tool = body
            .lines()
            .find(|line| line.starts_with("[tool_call|"))
            .unwrap_or_default();
        let call_id = line_value(body, "call_id:").or_else(|| line_value(body, "id:"));
        if let Some(call_id) = call_id {
            return format!("{label}\n{tool}\ncall_id:{call_id}");
        }
    }
    format!("{label}\n{body}")
}

fn line_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    body.lines()
        .find_map(|line| line.strip_prefix(key).map(str::trim))
        .filter(|value| !value.is_empty() && *value != "-")
}

fn request_trace_sections(body: &Value) -> Vec<(String, String)> {
    let mut sections = vec![(
        "request".to_string(),
        format!(
            "model: {}\nstream: {}\ntool_choice: {}",
            display_json(body.get("model")),
            display_json(body.get("stream")),
            display_json(body.get("tool_choice")),
        ),
    )];

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let rendered = tools
            .iter()
            .enumerate()
            .map(render_tool_definition)
            .collect::<Vec<_>>()
            .join("\n\n");
        if !rendered.is_empty() {
            sections.push(("tools".to_string(), rendered));
        }
    }

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for (index, message) in messages.iter().enumerate() {
            sections.extend(message_trace_sections(index, message));
        }
    }

    sections
}

fn message_trace_sections(_index: usize, message: &Value) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let role = message.get("role").and_then(Value::as_str).unwrap_or("-");

    let reasoning = message.get("reasoning_content").and_then(Value::as_str);
    if let Some(content) = render_content(message.get("content")) {
        let label = match role {
            "assistant" => "agent",
            "tool" => "reply_toolcall",
            "user" => "user",
            "system" | "developer" => "system_prompt",
            other => other,
        };
        sections.push((
            label.to_string(),
            render_message_body(reasoning, &content, message.get("tool_call_id")),
        ));
    } else if let Some(reasoning) = reasoning.filter(|text| !text.is_empty()) {
        sections.push((
            "agent".to_string(),
            render_message_body(Some(reasoning), "", None),
        ));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for (call_index, call) in tool_calls.iter().enumerate() {
            sections.push((
                "agent_toolcall".to_string(),
                render_tool_call(call_index, call),
            ));
        }
    }

    sections
}

fn upstream_trace_sections(chunk: &Value) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
        return sections;
    };

    for choice in choices {
        let delta = choice.get("delta").unwrap_or(&Value::Null);

        let reasoning = reasoning_delta(delta);
        let content = delta
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if reasoning.is_some() || !content.is_empty() {
            sections.push((
                "agent".to_string(),
                render_message_body(reasoning, content, None),
            ));
        }

        if let Some(finish_reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
            sections.push((
                "event".to_string(),
                format!("finish_reason: {finish_reason}"),
            ));
        }
    }

    sections
}

fn response_event_trace_sections(value: &Value) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let event_type = value.get("type").and_then(Value::as_str).unwrap_or("-");
    let response_id = value
        .get("response")
        .and_then(|response| response.get("id"))
        .or_else(|| value.get("response_id"))
        .and_then(Value::as_str)
        .unwrap_or("-");
    let prefix = format!("[responses event type={event_type} response_id={response_id}]");

    if let Some(item) = value.get("item") {
        match item.get("type").and_then(Value::as_str).unwrap_or("-") {
            "function_call" | "custom_tool_call" | "tool_search_call" => {
                sections.push((
                    "agent_toolcall".to_string(),
                    render_response_tool_call(item),
                ));
            }
            _ => {}
        }
    }

    if let Some(error) = value.get("error").filter(|error| !error.is_null()) {
        sections.push(("event".to_string(), format!("{prefix}\nerror: {error}")));
    } else if event_type == "response.failed" {
        sections.push(("event".to_string(), prefix));
    }

    sections
}

fn render_tool_definition((index, tool): (usize, &Value)) -> String {
    let function = tool.get("function").unwrap_or(tool);
    let name = function.get("name").and_then(Value::as_str).unwrap_or("-");
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!(
        "[tool #{index}]\nname: {name}\ndescription:\n{}\nparameters: {}",
        text_preview(description),
        display_json(function.get("parameters")),
    )
}

fn render_tool_call(index: usize, call: &Value) -> String {
    let function = call.get("function").unwrap_or(&Value::Null);
    let name = function.get("name").and_then(Value::as_str).unwrap_or("-");
    let arguments = function
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!(
        "[tool_call|{name}]\nindex: {}\nid: {}\narguments:\n{}",
        call.get("index")
            .and_then(Value::as_u64)
            .unwrap_or(index as u64),
        call.get("id").and_then(Value::as_str).unwrap_or("-"),
        arguments,
    )
}

fn render_response_tool_call(item: &Value) -> String {
    let name = item.get("name").and_then(Value::as_str).unwrap_or("-");
    let input = item
        .get("input")
        .or_else(|| item.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!(
        "[tool_call|{name}]\ntype: {}\nid: {}\ncall_id: {}\ninput:\n{}",
        item.get("type").and_then(Value::as_str).unwrap_or("-"),
        item.get("id").and_then(Value::as_str).unwrap_or("-"),
        item.get("call_id").and_then(Value::as_str).unwrap_or("-"),
        input,
    )
}

fn render_message_body(
    reasoning: Option<&str>,
    content: &str,
    tool_call_id: Option<&Value>,
) -> String {
    let mut body = String::new();
    if let Some(tool_call_id) = tool_call_id.and_then(Value::as_str) {
        body.push_str("[reply_toolcall|");
        body.push_str(tool_call_id);
        body.push_str("]\n");
    }
    if let Some(reasoning) = reasoning.filter(|text| !text.is_empty()) {
        body.push_str("<think>");
        if think_trace_enabled() {
            body.push_str(reasoning);
        } else {
            body.push_str(&format!(
                "{} chars hidden; set ADAPTER_DEBUG_THINK=1 to show",
                reasoning.chars().count()
            ));
        }
        body.push_str("</think>");
    }
    if !content.is_empty() {
        body.push('\n');
        body.push_str(content);
    }
    body
}

fn render_content(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) if !text.is_empty() => Some(text.to_string()),
        Value::Array(parts) => {
            let rendered = parts
                .iter()
                .filter_map(render_content_part)
                .collect::<Vec<_>>()
                .join("\n");
            (!rendered.trim().is_empty()).then_some(rendered)
        }
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

fn render_content_part(part: &Value) -> Option<String> {
    part.get("text")
        .and_then(Value::as_str)
        .or_else(|| part.get("content").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn reasoning_delta(delta: &Value) -> Option<&str> {
    delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .or_else(|| delta.get("thinking"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn display_json(value: Option<&Value>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn text_preview(text: &str) -> String {
    const MAX_CHARS: usize = 240;
    let mut preview = text.chars().take(MAX_CHARS).collect::<String>();
    if text.chars().count() > MAX_CHARS {
        preview.push_str("...");
    }
    preview
}

#[derive(Default)]
struct SseDecoder {
    buffer: String,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) -> anyhow::Result<Vec<Value>> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut values = Vec::new();

        while let Some((index, boundary_len)) = sse_boundary(&self.buffer) {
            let block = self.buffer[..index].to_string();
            self.buffer.drain(..index + boundary_len);
            if let Some(value) = parse_sse_block(&block)? {
                values.push(value);
            }
        }

        Ok(values)
    }
}

fn sse_boundary(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|index| (index, 2));
    let crlf = buffer.find("\r\n\r\n").map(|index| (index, 4));
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(boundary), None) | (None, Some(boundary)) => Some(boundary),
        (None, None) => None,
    }
}

fn parse_sse_block(block: &str) -> anyhow::Result<Option<Value>> {
    let mut data = String::new();
    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        if !data.is_empty() {
            data.push('\n');
        }
        data.push_str(payload.trim());
    }
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&data)?))
}

#[derive(Debug, thiserror::Error)]
enum AdapterHttpError {
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error("Upstream HTTP {0}: {1}")]
    Upstream(StatusCode, String),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        request_trace_sections, response_event_trace_sections, upstream_trace_sections, SseDecoder,
        TraceState,
    };

    #[test]
    fn parses_openai_style_sse_chunk() {
        let mut decoder = SseDecoder::default();
        let got = decoder
            .push(
                br#"data: {"id":"1","choices":[{"delta":{"content":"hi"}}]}

data: [DONE]

"#,
            )
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["id"], "1");
    }

    #[test]
    fn parses_crlf_sse_boundaries() {
        let mut decoder = SseDecoder::default();
        let got = decoder
            .push(b"data: {\"id\":\"1\",\"choices\":[]}\r\n\r\ndata: [DONE]\r\n\r\n")
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["id"], "1");
    }

    #[test]
    fn buffers_split_sse_events() {
        let mut decoder = SseDecoder::default();
        assert!(decoder
            .push(br#"data: {"id":"1","choices":["#)
            .unwrap()
            .is_empty());
        let got = decoder.push(br#"]}"#).unwrap();
        assert!(got.is_empty());
        let got = decoder.push(b"\n\n").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["id"], "1");
    }

    #[test]
    fn request_trace_shows_messages_and_tools() {
        let body = json!({
            "model": "deepseek-chat",
            "stream": true,
            "messages": [
                {"role": "system", "content": "tool guidance"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": "{\"input\":\"*** Begin Patch\"}"
                        }
                    }]
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Use apply_patch",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": "auto"
        });

        let got = request_trace_sections(&body);

        assert!(got
            .iter()
            .any(|(label, body)| label == "request" && body.contains("model: \"deepseek-chat\"")));
        assert!(got
            .iter()
            .any(|(label, body)| label == "system_prompt" && body.contains("tool guidance")));
        assert!(got
            .iter()
            .any(|(label, body)| label == "tools" && body.contains("name: apply_patch")));
        assert!(got
            .iter()
            .any(|(label, body)| label == "agent_toolcall"
                && body.contains("[tool_call|apply_patch]")));
    }

    #[test]
    fn trace_state_appends_only_unseen_sections() {
        let first = request_trace_sections(&json!({
            "model": "deepseek-chat",
            "stream": true,
            "messages": [
                {"role": "system", "content": "tool guidance"},
                {"role": "user", "content": "你好"}
            ]
        }));
        let second = request_trace_sections(&json!({
            "model": "deepseek-chat",
            "stream": true,
            "messages": [
                {"role": "system", "content": "tool guidance"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "reasoning_content": "the user hello to me", "content": "你好"}
            ]
        }));

        let mut state = TraceState::default();
        let first_unseen = state.unseen_sections(first);
        let second_unseen = state.unseen_sections(second);

        assert_eq!(first_unseen.len(), 3);
        assert_eq!(second_unseen.len(), 1);
        assert_eq!(second_unseen[0].0, "agent");
        assert!(second_unseen[0].1.contains("<think>"));
        assert!(second_unseen[0]
            .1
            .contains("chars hidden; set ADAPTER_DEBUG_THINK=1 to show</think>"));
        assert!(!second_unseen[0].1.contains("the user hello to me"));
    }

    #[test]
    fn trace_state_deduplicates_response_tool_call_when_request_history_replays_it() {
        let response_event = response_event_trace_sections(&json!({
            "type": "response.output_item.done",
            "response_id": "resp_1",
            "item": {
                "id": "fc_1",
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "call_1",
                "input": "*** Begin Patch\n*** End Patch\n",
                "status": "completed"
            }
        }));
        let request_history = request_trace_sections(&json!({
            "model": "deepseek-chat",
            "stream": true,
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": "{\"input\":\"*** Begin Patch\\n*** End Patch\\n\"}"
                    }
                }]
            }]
        }));

        let mut state = TraceState::default();
        let response_unseen = state.unseen_sections(response_event);
        let history_unseen = state.unseen_sections(request_history);

        assert!(response_unseen
            .iter()
            .any(|(label, body)| label == "agent_toolcall"
                && body.contains("[tool_call|apply_patch]")));
        assert!(!history_unseen
            .iter()
            .any(|(label, _)| label == "agent_toolcall"));
    }

    #[test]
    fn upstream_trace_shows_think_only_when_stream_trace_is_enabled() {
        let chunk = json!({
            "id": "chatcmpl_1",
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": "need edit",
                    "content": "thinking",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": "{\"input\":\"*** Begin Patch\"}"
                        }
                    }]
                },
                "finish_reason": null
            }]
        });

        let got = upstream_trace_sections(&chunk);

        assert!(got.iter().any(|(label, body)| label == "agent"
            && body.contains("<think>9 chars hidden; set ADAPTER_DEBUG_THINK=1 to show</think>")
            && body.contains("thinking")));
        assert!(!got.iter().any(|(label, _)| label == "agent_toolcall"));
    }

    #[test]
    fn response_event_trace_shows_custom_tool_input() {
        let event = json!({
            "type": "response.output_item.done",
            "response_id": "resp_1",
            "item": {
                "id": "call_1",
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "call_1",
                "input": "*** Begin Patch\n*** End Patch\n",
                "status": "completed"
            }
        });

        let got = response_event_trace_sections(&event);

        assert!(got.iter().any(|(label, body)| label == "agent_toolcall"
            && body.contains("type: custom_tool_call")
            && body.contains("[tool_call|apply_patch]")
            && body.contains("*** Begin Patch\n*** End Patch\n")));
    }
}
