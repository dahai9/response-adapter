use pretty_assertions::assert_eq;
use responses_adapter::adapter::{
    build_chat_body, sanitize_chat_messages, ReasoningStore, StreamingAccumulator, ToolNameMapper,
};
use responses_adapter::config::{Config, ThinkingMode};
use serde_json::json;
use std::net::SocketAddr;
use std::time::Duration;

fn config(model_override: Option<&str>, thinking: Option<ThinkingMode>) -> Config {
    Config {
        api_key: "test-key".into(),
        base_url: "https://api.example.com".into(),
        model_map: std::collections::HashMap::new(),
        model_override: model_override.map(ToOwned::to_owned),
        thinking,
        timeout: Duration::from_secs(120),
        listen: "127.0.0.1:8787".parse::<SocketAddr>().unwrap(),
        models: Vec::new(),
    }
}

#[test]
fn converts_namespace_type_tools_to_flat_function_tools() {
    let request = json!({
        "model": "test-model",
        "instructions": "You are concise.",
        "tools": [{
            "type": "namespace",
            "name": "jina-mcp-server",
            "description": "Tools in the jina-mcp-server namespace.",
            "tools": [{
                "type": "function",
                "name": "search",
                "description": "Search the web",
                "strict": false,
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }]
        }],
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();

    assert_eq!(converted.body["model"], "test-model");
    assert_eq!(converted.body["messages"][0]["role"], "system");
    assert_eq!(
        converted.body["messages"][1],
        json!( {"role": "user", "content": "hello"})
    );
    assert_eq!(
        converted.body["tools"][0]["function"]["name"],
        "jina_mcp_server__search"
    );
    assert_eq!(
        converted.body["tools"][0]["function"]["description"],
        "Search the web"
    );
}

#[test]
fn maps_messages_tools_and_reasoning_effort() {
    let request = json!({
        "model": "test-model",
        "instructions": "You are concise.",
        "reasoning": {"effort": "xhigh"},
        "tools": [{
            "type": "function",
            "namespace": "mcp/server",
            "name": "lookup",
            "description": "lookup things",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
        }],
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}]
        }]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();

    assert_eq!(converted.body["model"], "test-model");
    assert_eq!(converted.body["reasoning_effort"], "max");
    assert_eq!(converted.body["messages"][0]["role"], "system");
    assert_eq!(
        converted.body["messages"][1],
        json!({"role": "user", "content": "hello"})
    );
    assert_eq!(
        converted.body["tools"][0]["function"]["name"],
        "mcp_server__lookup"
    );
}

#[test]
fn env_model_override_wins() {
    let request = json!({"model": "test-model", "input": "ping"});
    let converted = build_chat_body(
        &request,
        &config(Some("override-model"), None),
        &ReasoningStore::default(),
    )
    .unwrap();
    assert_eq!(converted.body["model"], "override-model");
}

#[test]
fn drops_assistant_tool_call_without_matching_tool_output() {
    let messages = vec![
        json!({"role": "user", "content": "use a tool"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_missing",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}
            }]
        }),
    ];

    let sanitized = sanitize_chat_messages(messages);
    assert_eq!(
        sanitized,
        vec![json!({"role": "user", "content": "use a tool"})]
    );
}

#[test]
fn keeps_assistant_tool_call_adjacent_to_tool_output() {
    let messages = vec![
        json!({"role": "user", "content": "use a tool"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}
            }]
        }),
        json!({"role": "tool", "tool_call_id": "call_1", "content": "ok"}),
    ];

    let sanitized = sanitize_chat_messages(messages);
    assert_eq!(sanitized.len(), 3);
    assert_eq!(sanitized[1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(sanitized[2]["role"], "tool");
}

#[test]
fn attaches_reasoning_content_for_thinking_tool_subturn() {
    let mut store = ReasoningStore::default();
    store.remember_tool_reasoning(vec!["call_1".to_string()], "internal reasoning");
    let request = json!({
        "model": "test-model",
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        ]
    });

    let converted =
        build_chat_body(&request, &config(None, Some(ThinkingMode::Enabled)), &store).unwrap();

    assert_eq!(converted.body["thinking"]["type"], "enabled");
    assert_eq!(
        converted.body["messages"][0]["reasoning_content"],
        "internal reasoning"
    );
}

#[test]
fn attaches_reasoning_content_when_reasoning_effort_is_present() {
    let mut store = ReasoningStore::default();
    store.remember_tool_reasoning(vec!["call_1".to_string()], "internal reasoning");
    let request = json!({
        "model": "test-model",
        "reasoning": {"effort": "high"},
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        ]
    });

    let converted = build_chat_body(&request, &config(None, None), &store).unwrap();

    assert_eq!(converted.body["reasoning_effort"], "high");
    assert_eq!(
        converted.body["messages"][0]["reasoning_content"],
        "internal reasoning"
    );
}

#[test]
fn emits_empty_reasoning_content_when_required_but_missing() {
    let request = json!({
        "model": "test-model",
        "reasoning": {"effort": "high"},
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        ]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();

    assert_eq!(converted.body["messages"][0]["reasoning_content"], "");
}

#[test]
fn tool_call_history_always_carries_reasoning_content_field() {
    let request = json!({
        "model": "test-model",
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        ]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();

    assert_eq!(converted.body["messages"][0]["reasoning_content"], "");
}

#[test]
fn mapper_round_trips_namespace() {
    let mut mapper = ToolNameMapper::default();
    let encoded = mapper.add("run", Some("mcp/server"));
    assert_eq!(encoded, "mcp_server__run");
    assert_eq!(
        mapper.decode(&encoded),
        (Some("mcp/server".into()), "run".into())
    );
}

#[test]
fn streaming_text_starts_item_before_delta() {
    let mut accumulator = StreamingAccumulator::new(ToolNameMapper::default());
    let events = accumulator.ingest(&json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{"delta": {"content": "po"}}]
    }));

    assert_eq!(events[0]["type"], "response.output_item.added");
    assert_eq!(
        events[1],
        json!({"type": "response.output_text.delta", "delta": "po"})
    );

    let mut store = ReasoningStore::default();
    let done = accumulator.final_events(&mut store);
    assert_eq!(done[0]["type"], "response.output_item.done");
    assert_eq!(done[0]["item"]["id"], events[0]["item"]["id"]);
}
