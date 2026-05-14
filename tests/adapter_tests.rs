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
fn converts_custom_apply_patch_tool_to_chat_function() {
    let request = json!({
        "model": "test-model",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Use the apply_patch tool to edit files.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: begin_patch hunk+ end_patch"
            }
        }],
        "input": [{"type": "message", "role": "user", "content": "edit file"}]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();
    let tools = converted.body["tools"].as_array().unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "apply_patch");
    assert_eq!(tools[0]["function"]["parameters"]["required"][0], "input");
    let description = tools[0]["function"]["description"].as_str().unwrap();
    assert!(description
        .contains("Use the apply_patch tool to edit files.\nThis Responses freeform/custom tool"));
    assert!(description.contains("\"type\": \"grammar\""));
    assert!(description.contains("\"syntax\": \"lark\""));
    assert!(description.contains("\"definition\": \"start: begin_patch hunk+ end_patch\""));
    assert!(description.contains("complete raw apply_patch patch text"));
    assert!(converted.body["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("single string argument named `input`"));
    assert_eq!(converted.body["messages"][1]["role"], "user");
}

#[test]
fn preserves_developer_messages_as_system_messages() {
    let request = json!({
        "model": "test-model",
        "input": [
            {"type": "message", "role": "developer", "content": "Always use tools for file edits."},
            {"type": "message", "role": "user", "content": "edit file"}
        ]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();

    assert_eq!(converted.body["messages"][0]["role"], "system");
    assert_eq!(
        converted.body["messages"][0]["content"],
        "Always use tools for file edits."
    );
    assert_eq!(converted.body["messages"][1]["role"], "user");
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
fn custom_tool_call_history_wraps_freeform_input() {
    let request = json!({
        "model": "test-model",
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** Add File: note.txt\n+ok\n*** End Patch\n"
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "Done"
            }
        ]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();
    let message = &converted.body["messages"][0];

    assert_eq!(message["tool_calls"][0]["function"]["name"], "apply_patch");
    let args: serde_json::Value = serde_json::from_str(
        message["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        args["input"],
        "*** Begin Patch\n*** Add File: note.txt\n+ok\n*** End Patch\n"
    );
    assert_eq!(converted.body["messages"][1]["role"], "tool");
}

#[test]
fn failed_apply_patch_output_adds_retry_guidance() {
    let request = json!({
        "model": "test-model",
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** Add File: note.txt\n+new\n*** End Patch\n"
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "name": "apply_patch",
                "output": "error: file already exists"
            }
        ]
    });

    let converted =
        build_chat_body(&request, &config(None, None), &ReasoningStore::default()).unwrap();

    assert_eq!(converted.body["messages"][1]["role"], "tool");
    let content = converted.body["messages"][1]["content"].as_str().unwrap();
    assert!(content.contains("file already exists"));
    assert!(content.contains("retry with `*** Update File:`"));
    assert!(content.contains("Do not abandon apply_patch"));
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

#[test]
fn finish_reason_tool_calls_without_tool_delta_still_ends_turn() {
    let mut accumulator = StreamingAccumulator::new(ToolNameMapper::default());
    accumulator.ingest(&json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{
            "delta": {"content": "Now let me write the document."},
            "finish_reason": "tool_calls"
        }]
    }));

    let mut store = ReasoningStore::default();
    let done = accumulator.final_events(&mut store);

    assert_eq!(done.last().unwrap()["type"], "response.completed");
    assert_eq!(done.last().unwrap()["response"]["end_turn"], true);
}

#[test]
fn valid_tool_delta_requests_follow_up_turn() {
    let mut accumulator = StreamingAccumulator::new(ToolNameMapper::default());
    accumulator.ingest(&json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));

    let mut store = ReasoningStore::default();
    let done = accumulator.final_events(&mut store);

    assert_eq!(done[0]["type"], "response.output_item.done");
    assert_eq!(done[0]["item"]["type"], "function_call");
    assert_eq!(done.last().unwrap()["response"]["end_turn"], false);
}

#[test]
fn text_before_tool_call_is_marked_as_commentary() {
    let mut mapper = ToolNameMapper::default();
    mapper.add("lookup", None);
    let mut accumulator = StreamingAccumulator::new(mapper);
    accumulator.ingest(&json!({
        "id": "chatcmpl_1",
        "model": "test-model",
        "choices": [{
            "delta": {
                "content": "I will look that up.",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));

    let mut store = ReasoningStore::default();
    let done = accumulator.final_events(&mut store);

    assert_eq!(done[0]["type"], "response.output_item.done");
    assert_eq!(done[0]["item"]["phase"], "commentary");
    assert_eq!(done.last().unwrap()["response"]["end_turn"], false);
}
