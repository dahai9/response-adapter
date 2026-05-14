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
    kinds: HashMap<String, ToolKind>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ToolKind {
    Function,
    Custom,
}

impl ToolNameMapper {
    pub fn add(&mut self, name: &str, namespace: Option<&str>) -> String {
        self.add_with_kind(name, namespace, ToolKind::Function)
    }

    fn add_custom(&mut self, name: &str) -> String {
        self.add_with_kind(name, None, ToolKind::Custom)
    }

    fn add_with_kind(&mut self, name: &str, namespace: Option<&str>, kind: ToolKind) -> String {
        let key = match namespace {
            Some(namespace) if !namespace.is_empty() => format!("{namespace}/{name}"),
            _ => name.to_string(),
        };
        if let Some(encoded) = self.forward.get(&key) {
            self.kinds.entry(encoded.clone()).or_insert(kind);
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
        self.kinds.insert(encoded.clone(), kind);
        encoded
    }

    pub fn decode(&self, encoded: &str) -> (Option<String>, String) {
        self.reverse
            .get(encoded)
            .cloned()
            .unwrap_or_else(|| (None, encoded.to_string()))
    }

    fn kind(&self, encoded: &str) -> ToolKind {
        self.kinds
            .get(encoded)
            .copied()
            .unwrap_or(ToolKind::Function)
    }

    fn has_custom_tools(&self) -> bool {
        self.kinds
            .values()
            .any(|kind| matches!(kind, ToolKind::Custom))
    }
}

#[derive(Debug)]
pub struct ConvertedRequest {
    pub body: Value,
    pub mapper: ToolNameMapper,
}

pub fn build_chat_body(
    request: &Value,
    config: &Config,
    reasoning_store: &ReasoningStore,
) -> anyhow::Result<ConvertedRequest> {
    let mut mapper = ToolNameMapper::default();
    let mut messages = Vec::new();
    if let Some(instructions) = request.get("instructions").and_then(Value::as_str) {
        if !instructions.trim().is_empty() {
            messages.push(json!({"role": "system", "content": instructions}));
        }
    }

    let tools = convert_tools(request.get("tools"), &mut mapper);
    let effort = resolve_reasoning_effort(request);
    let attach_reasoning = config.thinking == Some(ThinkingMode::Enabled) || effort.is_some();
    if mapper.has_custom_tools() {
        messages.push(json!({"role": "system", "content": custom_tool_bridge_instructions()}));
    }
    if !tools.is_empty() {
        messages.push(json!({"role": "system", "content": tool_turn_continuity_instructions()}));
    }
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

    let model = config.resolve_model(request.get("model").and_then(Value::as_str))?;

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
        body["reasoning_effort"] = Value::String(effort.to_string());
    }

    if config.thinking == Some(ThinkingMode::Enabled) {
        body["thinking"] = json!({"type": ThinkingMode::Enabled.as_str()});
    }

    Ok(ConvertedRequest { body, mapper })
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
        if tool_type == "tool_search" {
            let encoded_name = mapper.add("tool_search", None);
            let description = tool
                .get("description")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            let parameters = tool
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            out.push(json!({
                "type": "function",
                "function": {
                    "name": encoded_name,
                    "description": description,
                    "parameters": parameters
                }
            }));
            continue;
        }
        if tool_type == "custom" {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            let encoded_name = mapper.add_custom(name);
            let description = custom_tool_description(tool);
            out.push(json!({
                "type": "function",
                "function": {
                    "name": encoded_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Raw freeform input for this tool."
                            }
                        },
                        "required": ["input"],
                        "additionalProperties": false
                    }
                }
            }));
            continue;
        }
        // Handle namespace-type tools by flattening into individual function tools.
        // The Responses API "namespace" type groups tools under a server/namespace name,
        // but Chat Completions only supports flat function tools.
        if tool_type == "namespace" {
            let namespace_name = tool.get("name").and_then(Value::as_str).unwrap_or("");
            if let Some(Value::Array(inner_tools)) = tool.get("tools") {
                for inner_tool in inner_tools {
                    if inner_tool.get("type").and_then(Value::as_str) != Some("function") {
                        continue;
                    }
                    let Some(name) = inner_tool.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let encoded_name = mapper.add(name, Some(namespace_name));
                    let description = inner_tool
                        .get("description")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new()));
                    let parameters = inner_tool
                        .get("parameters")
                        .cloned()
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
            }
            continue;
        }

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
    let call_names = input_call_names(items);

    let mut out = Vec::new();
    for item in items {
        let Some(kind) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let role = match role {
                    "developer" => "system",
                    "system" | "user" | "assistant" => role,
                    _ => continue,
                };
                let content = content_to_text(item.get("content").unwrap_or(&Value::Null));
                let mut message = json!({"role": role, "content": content});
                if attach_reasoning && role == "assistant" {
                    if let Some(reasoning) = reasoning_store.reasoning_for_text(&content) {
                        message["reasoning_content"] = Value::String(reasoning.to_string());
                    }
                }
                out.push(message);
            }
            "tool_search_call" => {
                let encoded_name = mapper.add("tool_search", None);
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| item_id("call"));
                let arguments = item
                    .get("arguments")
                    .map(arguments_to_string)
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
            "custom_tool_call" => {
                let Some(name) = item.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let encoded_name = mapper.add_custom(name);
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| item_id("call"));
                let input = item
                    .get("input")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| "{}".into());
                let arguments = json!({"input": input}).to_string();
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
            "function_call" => {
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
            "tool_search_output" => {
                let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
                    continue;
                };
                let content = json!({
                    "status": item
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or("completed"),
                    "execution": item
                        .get("execution")
                        .and_then(Value::as_str)
                        .unwrap_or("client"),
                    "tools": item
                        .get("tools")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new()))
                })
                .to_string();
                out.push(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
            }
            "function_call_output" | "custom_tool_call_output" => {
                let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
                    continue;
                };
                let mut content = item
                    .get("output")
                    .map(content_to_text)
                    .or_else(|| item.get("result").map(content_to_text))
                    .unwrap_or_default();
                if is_apply_patch_output(item, call_id, &call_names)
                    && apply_patch_output_needs_recovery_hint(&content)
                {
                    content.push_str(apply_patch_recovery_guidance());
                }
                out.push(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
            }
            _ => {}
        }
    }
    out
}

fn input_call_names(items: &[Value]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for item in items {
        let Some(kind) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        if !matches!(
            kind,
            "custom_tool_call" | "function_call" | "tool_search_call"
        ) {
            continue;
        }
        let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
            continue;
        };
        let name = if kind == "tool_search_call" {
            Some("tool_search")
        } else {
            item.get("name").and_then(Value::as_str)
        };
        if let Some(name) = name {
            names.insert(call_id.to_string(), name.to_string());
        }
    }
    names
}

fn is_apply_patch_output(
    item: &Value,
    call_id: &str,
    call_names: &HashMap<String, String>,
) -> bool {
    item.get("type").and_then(Value::as_str) == Some("custom_tool_call_output")
        && (item.get("name").and_then(Value::as_str) == Some("apply_patch")
            || call_names
                .get(call_id)
                .is_some_and(|name| name == "apply_patch"))
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
    assistant_message_item_with_id(&item_id("msg"), text, None)
}

fn assistant_message_item_with_id(id: &str, text: &str, phase: Option<&str>) -> Value {
    let mut item = json!({
        "type": "message",
        "role": "assistant",
        "id": id,
        "content": [{"type": "output_text", "text": text}]
    });
    if let Some(phase) = phase {
        item["phase"] = Value::String(phase.to_string());
    }
    json!({
        "type": "response.output_item.done",
        "item": item
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
    let call_id = tool_call
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| item_id("call"));
    if namespace.is_none() && mapper.kind(encoded_name) == ToolKind::Custom {
        let input = custom_input_from_arguments(&arguments);
        return Some(json!({
            "type": "response.output_item.done",
            "item": {
                "type": "custom_tool_call",
                "call_id": call_id,
                "status": "completed",
                "name": name,
                "input": input
            }
        }));
    }
    if namespace.is_none() && name == "tool_search" {
        let arguments = serde_json::from_str::<Value>(&arguments)
            .unwrap_or_else(|_| json!({"query": arguments}));
        return Some(json!({
            "type": "response.output_item.done",
            "item": {
                "type": "tool_search_call",
                "call_id": call_id,
                "status": "completed",
                "execution": "client",
                "arguments": arguments
            }
        }));
    }
    let mut item = json!({
        "type": "function_call",
        "call_id": call_id,
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

    pub fn has_output_items(&self) -> bool {
        !self.content.is_empty() || !self.tool_calls.is_empty()
    }

    pub fn has_stream_progress(&self) -> bool {
        self.resp_id.is_some()
            || self.model.is_some()
            || !self.reasoning_content.is_empty()
            || self.has_output_items()
            || self.finish_reason.is_some()
            || self.usage.is_some()
    }

    pub fn has_finish_reason(&self) -> bool {
        self.finish_reason.is_some()
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
            self.usage = Some(responses_usage_from_chat_usage(usage));
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
        self.final_events_with_end_turn_override(store, None)
    }

    pub fn final_events_after_interruption(&mut self, store: &mut ReasoningStore) -> Vec<Value> {
        let end_turn_override = (!self.has_finish_reason()).then_some(false);
        self.final_events_with_end_turn_override(store, end_turn_override)
    }

    fn final_events_with_end_turn_override(
        &mut self,
        store: &mut ReasoningStore,
        end_turn_override: Option<bool>,
    ) -> Vec<Value> {
        let resp_id = self.resp_id.clone().unwrap_or_else(response_id);
        let mut events = Vec::new();
        if !self.content.is_empty() {
            let id = self
                .message_item_id
                .get_or_insert_with(|| item_id("msg"))
                .clone();
            let phase = (!self.tool_calls.is_empty() || self.content_needs_follow_up())
                .then_some("commentary");
            store.remember_text_reasoning(&self.content, &self.reasoning_content);
            events.push(assistant_message_item_with_id(&id, &self.content, phase));
        }

        let mut call_ids = Vec::new();
        let mut emitted_tool_call = false;
        let mut indexes = self.tool_calls.keys().copied().collect::<Vec<_>>();
        indexes.sort_unstable();
        for index in indexes {
            if let Some(tool_call) = self.tool_calls.get(&index) {
                if let Some(call_id) = tool_call.get("id").and_then(Value::as_str) {
                    call_ids.push(call_id.to_string());
                }
                if let Some(event) = function_call_item(tool_call, &self.mapper) {
                    events.push(event);
                    emitted_tool_call = true;
                }
            }
        }
        store.remember_tool_reasoning(call_ids, &self.reasoning_content);

        // Some OpenAI-compatible providers emit finish_reason="tool_calls" even
        // when the streamed delta contains only text or an incomplete tool call.
        // Codex treats end_turn=false as "continue sampling", so only request a
        // follow-up when we actually produced a callable Responses tool item.
        let needs_follow_up = emitted_tool_call || self.content_needs_follow_up();
        let end_turn = end_turn_override.unwrap_or(!needs_follow_up);
        events.push(response_completed(&resp_id, self.usage.clone(), end_turn));
        events
    }

    fn content_needs_follow_up(&self) -> bool {
        if self.content.trim().is_empty() || !self.tool_calls.is_empty() {
            return false;
        }
        if matches!(self.finish_reason.as_deref(), Some("tool_calls")) {
            return false;
        }
        assistant_text_is_work_preamble(&self.content)
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

pub fn responses_usage_from_chat_usage(usage: &Value) -> Value {
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

fn resolve_reasoning_effort(request: &Value) -> Option<&'static str> {
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

fn arguments_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn custom_tool_description(tool: &Value) -> String {
    if tool.get("name").and_then(Value::as_str) == Some("apply_patch") {
        return apply_patch_tool_description(tool);
    }

    let mut description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if !description.is_empty() {
        description.push('\n');
    }
    description.push_str(
        "This Responses custom/freeform tool is exposed through Chat Completions as a function. Put the exact raw tool input in the `input` string argument.",
    );
    if let Some(format) = tool.get("format").filter(|format| !format.is_null()) {
        description.push_str("\nOriginal freeform format: ");
        description.push_str(&format.to_string());
    }
    description
}

fn custom_tool_bridge_instructions() -> &'static str {
    "Some Responses freeform/custom tools are exposed to this Chat Completions backend as normal function tools with a single string argument named `input`. Put the exact raw freeform tool input in `input`; do not wrap the freeform payload in another JSON object inside that string."
}

fn tool_turn_continuity_instructions() -> &'static str {
    "When you decide to inspect, edit, verify, or otherwise continue working with tools, do not end your turn immediately after a status update or intent announcement. Avoid stopping after phrases like 'I will...', 'Let me...', 'Writing ... now', '开始...', or '继续...'. In the same turn, either make the next tool call immediately or continue directly until you can produce a concrete result."
}

fn apply_patch_tool_description(tool: &Value) -> String {
    let mut description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("Use the `apply_patch` tool to edit files.")
        .to_string();
    if !description.ends_with('.') {
        description.push('.');
    }
    description.push_str(
        "\nThis Responses freeform/custom tool is exposed through Chat Completions as a function. The function has exactly one argument named `input`; put the complete raw apply_patch patch text in `input`.",
    );
    description.push_str(apply_patch_usage_guidance());
    if let Some(format) = tool.get("format").filter(|format| !format.is_null()) {
        description.push_str("\nOriginal Responses freeform tool format:\n");
        description.push_str(&pretty_json(format));
    }
    description
}

fn apply_patch_usage_guidance() -> &'static str {
    "\nApply patch usage rules:\n- Always send one complete patch payload. It must start with `*** Begin Patch` and the final line must be exactly `*** End Patch`.\n- Use `*** Update File: path` for existing files, `*** Add File: path` only for new files, and `*** Delete File: path` only when removing a file.\n- For update hunks, include enough unchanged context lines around the edit so the patch can be matched. If matching fails, read the current file and retry with fresher context.\n- Prefer small focused patches. For large rewrites, split the work into multiple smaller `*** Update File:` patches instead of one huge patch.\n- Do not use shell heredocs, Python scripts, or ad hoc redirection as a fallback for file edits; retry with apply_patch.\n- Example update:\n*** Begin Patch\n*** Update File: path/to/file.py\n@@\n old line\n+new line\n*** End Patch"
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn apply_patch_output_needs_recovery_hint(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    lower.contains("already exists")
        || lower.contains("file exists")
        || lower.contains("no such file")
        || lower.contains("failed to find")
        || lower.contains("expected")
        || lower.contains("apply_patch verification failed")
        || lower.contains("invalid patch")
}

fn apply_patch_recovery_guidance() -> &'static str {
    "\n\nAdapter guidance: the apply_patch attempt failed. Do not abandon apply_patch or switch to shell heredocs. Retry with apply_patch. If the target file already exists, read the current file and retry with `*** Update File:`. Use `*** Add File:` only for paths that do not exist. If the error says `invalid patch` or `The last line of the patch must be '*** End Patch'`, the patch payload was incomplete or malformed: split the edit into a smaller patch and ensure the final line is exactly `*** End Patch`."
}

fn custom_input_from_arguments(arguments: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(arguments) else {
        return arguments.to_string();
    };
    value
        .get("input")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| arguments.to_string())
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

fn assistant_text_is_work_preamble(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    let mut segments = vec![trimmed.to_string()];
    if let Some(last_line) = trimmed.lines().rev().find(|line| !line.trim().is_empty()) {
        let last_line = last_line.trim();
        if !last_line.is_empty() && last_line != trimmed {
            segments.push(last_line.to_string());
        }
    }
    if let Some(last_sentence) = last_sentence(trimmed) {
        if !last_sentence.is_empty() && !segments.iter().any(|segment| segment == &last_sentence) {
            segments.push(last_sentence);
        }
    }

    segments
        .iter()
        .any(|segment| segment_is_work_preamble(segment))
}

fn segment_is_work_preamble(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let starts_like_preamble = [
        "now i'll",
        "now i will",
        "i'll",
        "i will",
        "let me",
        "i'm going to",
        "i am going to",
        "writing",
        "rewriting",
        "creating",
        "updating",
        "editing",
        "implementing",
        "fixing",
        "testing",
        "optimizing",
        "improving",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
        || [
            "开始",
            "现在",
            "接下来",
            "继续",
            "我会",
            "我将",
            "先",
            "正在",
            "下面",
        ]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix));
    if !starts_like_preamble {
        return false;
    }

    [
        "rewrite",
        "rewriting",
        "write",
        "writing",
        "create",
        "creating",
        "update",
        "updating",
        "edit",
        "editing",
        "implement",
        "implementing",
        "fix",
        "fixing",
        "verify",
        "test",
        "testing",
        "optimize",
        "optimizing",
        "improve",
        "improving",
        "重写",
        "写",
        "创建",
        "生成",
        "更新",
        "修改",
        "编辑",
        "实现",
        "修复",
        "验证",
        "测试",
        "优化",
        "改进",
    ]
    .iter()
    .any(|needle| lower.contains(needle) || trimmed.contains(needle))
}

fn last_sentence(text: &str) -> Option<String> {
    let mut parts = text
        .split([
            '\n', '.', '!', '?', '。', '！', '？', ',', '，', ';', '；', ':', '：',
        ])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    parts.pop().map(ToOwned::to_owned)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_tools_exposes_tool_search_as_chat_function() {
        let tools = json!([
            {
                "type": "tool_search",
                "execution": "client",
                "description": "Search deferred tools",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        ]);
        let mut mapper = ToolNameMapper::default();

        let got = convert_tools(Some(&tools), &mut mapper);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["type"], "function");
        assert_eq!(got[0]["function"]["name"], "tool_search");
        assert_eq!(got[0]["function"]["description"], "Search deferred tools");
        assert_eq!(got[0]["function"]["parameters"]["required"][0], "query");
    }

    #[test]
    fn function_call_item_restores_tool_search_call() {
        let mut mapper = ToolNameMapper::default();
        mapper.add("tool_search", None);
        let tool_call = json!({
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "tool_search",
                "arguments": "{\"query\":\"mcp files\",\"limit\":3}"
            }
        });

        let got = function_call_item(&tool_call, &mapper).expect("tool_search call");

        assert_eq!(got["type"], "response.output_item.done");
        assert_eq!(got["item"]["type"], "tool_search_call");
        assert_eq!(got["item"]["call_id"], "call_1");
        assert_eq!(got["item"]["execution"], "client");
        assert_eq!(got["item"]["arguments"]["query"], "mcp files");
        assert_eq!(got["item"]["arguments"]["limit"], 3);
    }

    #[test]
    fn function_call_item_restores_custom_tool_call() {
        let mut mapper = ToolNameMapper::default();
        mapper.add_custom("apply_patch");
        let tool_call = json!({
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "apply_patch",
                "arguments": "{\"input\":\"*** Begin Patch\\n*** End Patch\\n\"}"
            }
        });

        let got = function_call_item(&tool_call, &mapper).expect("custom tool call");

        assert_eq!(got["type"], "response.output_item.done");
        assert_eq!(got["item"]["type"], "custom_tool_call");
        assert_eq!(got["item"]["call_id"], "call_1");
        assert_eq!(got["item"]["name"], "apply_patch");
        assert_eq!(got["item"]["input"], "*** Begin Patch\n*** End Patch\n");
    }

    #[test]
    fn streaming_accumulator_treats_reasoning_as_stream_progress() {
        let mut accumulator = StreamingAccumulator::new(ToolNameMapper::default());

        accumulator.ingest(&json!({
            "id": "chatcmpl_1",
            "model": "deepseek-chat",
            "choices": [{
                "delta": {"reasoning_content": "planning only"}
            }]
        }));

        assert!(accumulator.has_stream_progress());
        assert!(!accumulator.has_output_items());
        let mut store = ReasoningStore::default();
        let events = accumulator.final_events(&mut store);
        assert_eq!(events.last().unwrap()["type"], "response.completed");
        assert_eq!(events.last().unwrap()["response"]["end_turn"], true);
    }

    #[test]
    fn interrupted_stream_without_finish_reason_requests_follow_up() {
        let mut accumulator = StreamingAccumulator::new(ToolNameMapper::default());

        accumulator.ingest(&json!({
            "id": "chatcmpl_1",
            "model": "deepseek-chat",
            "choices": [{
                "delta": {"content": "Now I'll rewrite"}
            }]
        }));

        let mut store = ReasoningStore::default();
        let events = accumulator.final_events_after_interruption(&mut store);
        assert_eq!(events.last().unwrap()["type"], "response.completed");
        assert_eq!(events.last().unwrap()["response"]["end_turn"], false);
    }

    #[test]
    fn interrupted_stream_with_finish_reason_does_not_force_follow_up() {
        let mut accumulator = StreamingAccumulator::new(ToolNameMapper::default());

        accumulator.ingest(&json!({
            "id": "chatcmpl_1",
            "model": "deepseek-chat",
            "choices": [{
                "delta": {"content": "done"},
                "finish_reason": "stop"
            }]
        }));

        let mut store = ReasoningStore::default();
        let events = accumulator.final_events_after_interruption(&mut store);
        assert_eq!(events.last().unwrap()["type"], "response.completed");
        assert_eq!(events.last().unwrap()["response"]["end_turn"], true);
    }

    #[test]
    fn convert_input_keeps_tool_search_history_pair() {
        let input = json!([
            {
                "type": "tool_search_call",
                "call_id": "call_1",
                "status": "completed",
                "execution": "client",
                "arguments": {"query": "browser tools"}
            },
            {
                "type": "tool_search_output",
                "call_id": "call_1",
                "status": "completed",
                "execution": "client",
                "tools": [{"type": "function", "name": "read_mcp_resource"}]
            }
        ]);
        let mut mapper = ToolNameMapper::default();

        let got = convert_input(Some(&input), &mut mapper, &ReasoningStore::default(), false);

        assert_eq!(got.len(), 2);
        assert_eq!(got[0]["role"], "assistant");
        assert_eq!(got[0]["tool_calls"][0]["function"]["name"], "tool_search");
        assert_eq!(
            got[0]["tool_calls"][0]["function"]["arguments"],
            "{\"query\":\"browser tools\"}"
        );
        assert_eq!(got[1]["role"], "tool");
        assert_eq!(got[1]["tool_call_id"], "call_1");
        assert!(got[1]["content"]
            .as_str()
            .expect("content")
            .contains("read_mcp_resource"));
    }
}
