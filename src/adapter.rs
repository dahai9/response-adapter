use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

use crate::config::{Config, ThinkingMode};

pub type JsonMap = Map<String, Value>;

#[derive(Debug, Default)]
pub struct ReasoningStore {
    by_call_id: HashMap<String, String>,
    by_text_hash: HashMap<u64, String>,
}

impl ReasoningStore {
    pub fn reasoning_for_call(&self, call_id: &str) -> Option<&str> {
        self.by_call_id.get(call_id).map(String::as_str)
    }

    pub fn reasoning_for_text(&self, text: &str) -> Option<&str> {
        self.by_text_hash.get(&text_hash(text)).map(String::as_str)
    }

    pub fn remember_tool_reasoning<I>(&mut self, call_ids: I, reasoning: &str)
    where
        I: IntoIterator<Item = String>,
    {
        if reasoning.is_empty() {
            return;
        }
        for call_id in call_ids {
            self.by_call_id.insert(call_id, reasoning.to_string());
        }
    }

    pub fn remember_text_reasoning(&mut self, text: &str, reasoning: &str) {
        if text.is_empty() || reasoning.is_empty() {
            return;
        }
        self.by_text_hash
            .insert(text_hash(text), reasoning.to_string());
    }
}

#[derive(Debug, Default, Clone)]
pub struct ToolNameMapper {
    forward: HashMap<String, String>,
    reverse: HashMap<String, (Option<String>, String)>,
}

impl ToolNameMapper {
    pub fn add(&mut self, name: &str, namespace: Option<&str>) -> String {
        let key = match namespace {
            Some(namespace) if !namespace.is_empty() => format!("{namespace}/{name}"),
            _ => name.to_string(),
        };
        if let Some(encoded) = self.forward.get(&key) {
            return encoded.clone();
        }
        let base = encode_tool_name(namespace, name);
        let mut encoded = base.clone();
        let mut suffix = 2usize;
        while self.reverse.contains_key(&encoded) {
            encoded = format!("{base}_{suffix}");
            suffix += 1;
        }
        self.forward.insert(key, encoded.clone());
        self.reverse.insert(
            encoded.clone(),
            (
                namespace.map(ToOwned::to_owned).filter(|s| !s.is_empty()),
                name.to_string(),
            ),
        );
        encoded
    }

    pub fn decode(&self, encoded: &str) -> (Option<String>, String) {
        self.reverse
            .get(encoded)
            .cloned()
            .unwrap_or_else(|| (None, encoded.to_string()))
    }
}

#[derive(Debug)]
pub struct ConvertedRequest {
    pub body: Value,
    pub mapper: ToolNameMapper,
}

pub fn build_deepseek_body(
    request: &Value,
    config: &Config,
    reasoning_store: &ReasoningStore,
) -> ConvertedRequest {
    let mut mapper = ToolNameMapper::default();
    let mut messages = Vec::new();
    if let Some(instructions) = request.get("instructions").and_then(Value::as_str) {
        if !instructions.trim().is_empty() {
            messages.push(json!({"role": "system", "content": instructions}));
        }
    }

    let tools = convert_tools(request.get("tools"), &mut mapper);
    let effort = reasoning_effort_for_deepseek(request);
    let attach_reasoning = config.thinking == Some(ThinkingMode::Enabled) || effort.is_some();
    messages.extend(convert_input(
        request.get("input"),
        &mut mapper,
        reasoning_store,
        attach_reasoning,
    ));
    let messages = sanitize_chat_messages(messages);
    let messages = if messages.is_empty() {
        vec![json!({"role": "user", "content": ""})]
    } else {
        messages
    };

    let model = config
        .model_override
        .clone()
        .or_else(|| {
            request
                .get("model")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| Config::default_model().to_string());

    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true}
    });

    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = Value::String("auto".into());
    }

    if let Some(effort) = effort {
        body["reasoning_effort"] = Value::String(effort.into());
    }

    if config.thinking == Some(ThinkingMode::Enabled) {
        body["thinking"] = json!({"type": ThinkingMode::Enabled.as_str()});
    }

    ConvertedRequest { body, mapper }
}

pub fn convert_tools(tools: Option<&Value>, mapper: &mut ToolNameMapper) -> Vec<Value> {
    let Some(Value::Array(tools)) = tools else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tool in tools {
        let Some(tool_type) = tool.get("type").and_then(Value::as_str) else {
            continue;
        };
        if tool_type != "function" && tool.get("namespace").is_none() {
            continue;
        }
        let function = tool.get("function").unwrap_or(tool);
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            continue;
        };
        let namespace = tool.get("namespace").and_then(Value::as_str);
        let encoded_name = mapper.add(name, namespace);
        let description = function
            .get("description")
            .cloned()
            .unwrap_or_else(|| Value::String(String::new()));
        let parameters = function
            .get("parameters")
            .cloned()
            .or_else(|| tool.get("parameters").cloned())
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        out.push(json!({
            "type": "function",
            "function": {
                "name": encoded_name,
                "description": description,
                "parameters": parameters
            }
        }));
    }
    out
}

pub fn convert_input(
    input: Option<&Value>,
    mapper: &mut ToolNameMapper,
    reasoning_store: &ReasoningStore,
    attach_reasoning: bool,
) -> Vec<Value> {
    let items = match input {
        Some(Value::Array(items)) => items.as_slice(),
        Some(other) => return vec![json!({"role": "user", "content": content_to_text(other)})],
        None => return Vec::new(),
    };

    let mut out = Vec::new();
    for item in items {
        let Some(kind) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                if matches!(role, "system" | "user" | "assistant") {
                    let content = content_to_text(item.get("content").unwrap_or(&Value::Null));
                    let mut message = json!({"role": role, "content": content});
                    if attach_reasoning && role == "assistant" {
                        if let Some(reasoning) = reasoning_store.reasoning_for_text(&content) {
                            message["reasoning_content"] = Value::String(reasoning.to_string());
                        }
                    }
                    out.push(message);
                }
            }
            "function_call" | "custom_tool_call" => {
                let Some(name) = item.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let namespace = item.get("namespace").and_then(Value::as_str);
                let encoded_name = mapper.add(name, namespace);
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| item_id("call"));
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .or_else(|| {
                        item.get("input")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                    .unwrap_or_else(|| "{}".into());
                let mut message = json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": encoded_name, "arguments": arguments}
                    }]
                });
                let reasoning = reasoning_store
                    .reasoning_for_call(message["tool_calls"][0]["id"].as_str().unwrap_or_default())
                    .unwrap_or_default();
                message["reasoning_content"] = Value::String(reasoning.to_string());
                out.push(message);
            }
            "function_call_output" | "custom_tool_call_output" => {
                let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
                    continue;
                };
                let content = item
                    .get("output")
                    .map(content_to_text)
                    .or_else(|| item.get("result").map(content_to_text))
                    .unwrap_or_default();
                out.push(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
            }
            _ => {}
        }
    }
    out
}

pub fn sanitize_chat_messages(messages: Vec<Value>) -> Vec<Value> {
    let mut tool_outputs: HashMap<String, Vec<Value>> = HashMap::new();
    for message in &messages {
        if message.get("role").and_then(Value::as_str) == Some("tool") {
            if let Some(call_id) = message.get("tool_call_id").and_then(Value::as_str) {
                tool_outputs
                    .entry(call_id.to_string())
                    .or_default()
                    .push(message.clone());
            }
        }
    }

    let mut consumed = HashSet::new();
    let mut out = Vec::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str);
        if role == Some("tool") {
            continue;
        }
        let call_ids = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|call| {
                        call.get("id")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if call_ids.is_empty() {
            out.push(message);
            continue;
        }
        if call_ids
            .iter()
            .any(|call_id| !tool_outputs.contains_key(call_id))
        {
            continue;
        }
        out.push(message);
        for call_id in call_ids {
            if consumed.insert(call_id.clone()) {
                if let Some(outputs) = tool_outputs.get(&call_id) {
                    out.extend(outputs.iter().cloned());
                }
            }
        }
    }
    out
}

pub fn response_created(resp_id: &str, model: Option<&str>) -> Value {
    let mut response = json!({"id": resp_id});
    if let Some(model) = model {
        response["headers"] = json!({"openai-model": model});
    }
    json!({"type": "response.created", "response": response})
}

pub fn response_completed(resp_id: &str, usage: Option<Value>, end_turn: bool) -> Value {
    let mut response = json!({"id": resp_id, "output": [], "end_turn": end_turn});
    if let Some(usage) = usage {
        response["usage"] = usage;
    }
    json!({"type": "response.completed", "response": response})
}

pub fn response_failed(resp_id: &str, message: &str) -> Value {
    json!({
        "type": "response.failed",
        "response": {
            "id": resp_id,
            "error": {"type": "server_error", "code": "adapter_error", "message": message}
        }
    })
}

pub fn assistant_message_item(text: &str) -> Value {
    assistant_message_item_with_id(&item_id("msg"), text)
}

fn assistant_message_item_with_id(id: &str, text: &str) -> Value {
    json!({
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "id": id,
            "content": [{"type": "output_text", "text": text}]
        }
    })
}

fn assistant_message_item_added(id: &str) -> Value {
    json!({
        "type": "response.output_item.added",
        "item": {
            "type": "message",
            "role": "assistant",
            "id": id,
            "content": [{"type": "output_text", "text": ""}]
        }
    })
}

pub fn function_call_item(tool_call: &Value, mapper: &ToolNameMapper) -> Option<Value> {
    if tool_call.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let function = tool_call.get("function")?;
    let encoded_name = function.get("name")?.as_str()?;
    let (namespace, name) = mapper.decode(encoded_name);
    let arguments = function
        .get("arguments")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "{}".into());
    let mut item = json!({
        "type": "function_call",
        "call_id": tool_call.get("id").and_then(Value::as_str).map(ToOwned::to_owned).unwrap_or_else(|| item_id("call")),
        "name": name,
        "arguments": arguments
    });
    if let Some(namespace) = namespace {
        item["namespace"] = Value::String(namespace);
    }
    Some(json!({"type": "response.output_item.done", "item": item}))
}

#[derive(Debug, Default)]
pub struct StreamingAccumulator {
    mapper: ToolNameMapper,
    resp_id: Option<String>,
    model: Option<String>,
    content: String,
    reasoning_content: String,
    message_item_id: Option<String>,
    tool_calls: HashMap<usize, Value>,
    finish_reason: Option<String>,
    usage: Option<Value>,
}

impl StreamingAccumulator {
    pub fn new(mapper: ToolNameMapper) -> Self {
        Self {
            mapper,
            ..Default::default()
        }
    }

    pub fn resp_id(&self) -> Option<&str> {
        self.resp_id.as_deref()
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn ingest(&mut self, chunk: &Value) -> Vec<Value> {
        if self.resp_id.is_none() {
            self.resp_id = chunk
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .or_else(|| Some(response_id()));
        }
        if self.model.is_none() {
            self.model = chunk
                .get("model")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
        }
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(codex_usage_from_chat_usage(usage));
        }

        let mut events = Vec::new();
        let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
            return events;
        };
        for choice in choices {
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(reason.to_string());
            }
            let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
                continue;
            };
            if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
                self.reasoning_content.push_str(reasoning);
            }
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    if self.content.is_empty() {
                        let id = self
                            .message_item_id
                            .get_or_insert_with(|| item_id("msg"))
                            .clone();
                        events.push(assistant_message_item_added(&id));
                    }
                    self.content.push_str(content);
                    events.push(json!({"type": "response.output_text.delta", "delta": content}));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    self.ingest_tool_call_delta(call);
                }
            }
        }
        events
    }

    pub fn final_events(&mut self, store: &mut ReasoningStore) -> Vec<Value> {
        let resp_id = self.resp_id.clone().unwrap_or_else(response_id);
        let mut events = Vec::new();
        if !self.content.is_empty() {
            store.remember_text_reasoning(&self.content, &self.reasoning_content);
            let id = self
                .message_item_id
                .get_or_insert_with(|| item_id("msg"))
                .clone();
            events.push(assistant_message_item_with_id(&id, &self.content));
        }

        let mut call_ids = Vec::new();
        let mut indexes = self.tool_calls.keys().copied().collect::<Vec<_>>();
        indexes.sort_unstable();
        for index in indexes {
            if let Some(tool_call) = self.tool_calls.get(&index) {
                if let Some(call_id) = tool_call.get("id").and_then(Value::as_str) {
                    call_ids.push(call_id.to_string());
                }
                if let Some(event) = function_call_item(tool_call, &self.mapper) {
                    events.push(event);
                }
            }
        }
        store.remember_tool_reasoning(call_ids, &self.reasoning_content);

        let end_turn = self.finish_reason.as_deref() != Some("tool_calls");
        events.push(response_completed(&resp_id, self.usage.clone(), end_turn));
        events
    }

    fn ingest_tool_call_delta(&mut self, tool_call: &Value) {
        let index = tool_call
            .get("index")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or_else(|| self.tool_calls.len());
        let current = self.tool_calls.entry(index).or_insert_with(
            || json!({"id": null, "type": "function", "function": {"name": "", "arguments": ""}}),
        );
        if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
            current["id"] = Value::String(id.to_string());
        }
        if let Some(kind) = tool_call.get("type").and_then(Value::as_str) {
            current["type"] = Value::String(kind.to_string());
        }
        if let Some(function) = tool_call.get("function").and_then(Value::as_object) {
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                append_string(&mut current["function"]["name"], name);
            }
            if let Some(args) = function.get("arguments").and_then(Value::as_str) {
                append_string(&mut current["function"]["arguments"], args);
            }
        }
    }
}

pub fn codex_usage_from_chat_usage(usage: &Value) -> Value {
    let input = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(input + output);
    let cached = usage
        .get("prompt_cache_hit_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = usage
        .get("completion_tokens_details")
        .and_then(|v| v.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": total
    })
}

pub fn models_response() -> Value {
    json!({
        "object": "list",
        "data": [
            model_info("deepseek-v4-pro", "DeepSeek V4 Pro", 0),
            model_info("deepseek-v4-flash", "DeepSeek V4 Flash", 1),
            model_info("deepseek-chat", "DeepSeek Chat", 2),
            model_info("deepseek-reasoner", "DeepSeek Reasoner", 3)
        ]
    })
}

fn model_info(slug: &str, display_name: &str, priority: u64) -> Value {
    json!({
        "id": slug,
        "object": "model",
        "created": 0,
        "owned_by": "deepseek",
        "model": slug,
        "displayName": display_name,
        "description": "DeepSeek model exposed through deepseek-responses-adapter.",
        "defaultReasoningEffort": "high",
        "supportedReasoningEfforts": [
            {"reasoningEffort": "low", "description": "Mapped to DeepSeek high reasoning effort"},
            {"reasoningEffort": "medium", "description": "Mapped to DeepSeek high reasoning effort"},
            {"reasoningEffort": "high", "description": "DeepSeek high reasoning effort"},
            {"reasoningEffort": "xhigh", "description": "Mapped to DeepSeek max reasoning effort"}
        ],
        "inputModalities": ["text"],
        "supportsPersonality": false,
        "additionalSpeedTiers": [],
        "isDefault": priority == 0,
        "hidden": false,
        "priority": priority
    })
}

fn reasoning_effort_for_deepseek(request: &Value) -> Option<&'static str> {
    match request
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str)
    {
        Some("xhigh") => Some("max"),
        Some("low" | "medium" | "high") => Some("high"),
        _ => None,
    }
}

fn content_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(content_part_to_text)
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn content_part_to_text(value: &Value) -> Option<String> {
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    if let Some(text) = value.get("input_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    value.as_str().map(ToOwned::to_owned)
}

fn encode_tool_name(namespace: Option<&str>, name: &str) -> String {
    let raw = match namespace {
        Some(namespace) if !namespace.is_empty() => format!("{namespace}__{name}"),
        _ => name.to_string(),
    };
    let mut encoded = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if encoded.is_empty() {
        encoded = "tool".into();
    }
    if encoded
        .chars()
        .next()
        .map(|ch| ch.is_ascii_digit())
        .unwrap_or(true)
    {
        encoded.insert_str(0, "tool_");
    }
    encoded
}

fn append_string(value: &mut Value, suffix: &str) {
    let current = value.as_str().unwrap_or_default().to_string();
    *value = Value::String(format!("{current}{suffix}"));
}

pub fn response_id() -> String {
    item_id("resp")
}

fn item_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{prefix}_{nanos}")
}

fn text_hash(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}
