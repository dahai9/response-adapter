import json
import unittest

import deepseek_responses_adapter as adapter


class AdapterTests(unittest.TestCase):
    def test_build_deepseek_body_maps_messages_and_tools(self):
        mapper = adapter.ToolNameMapper()
        body = adapter.build_deepseek_body(
            {
                "model": "deepseek-v4-pro",
                "instructions": "You are Codex.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "list files"}],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [{"type": "input_text", "text": "ok"}],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "shell_command",
                        "description": "Run shell",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "namespace",
                        "name": "mcp/sample",
                        "tools": [
                            {
                                "type": "function",
                                "name": "lookup",
                                "description": "Lookup",
                                "parameters": {"type": "object"},
                            }
                        ],
                    },
                ],
                "reasoning": {"effort": "xhigh"},
            },
            mapper,
        )

        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(body["stream"], True)
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "You are Codex."})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "list files"})
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["tools"][0]["function"]["name"], "shell_command")
        self.assertEqual(mapper.decode("shell_command"), (None, "shell_command"))
        encoded_namespace_name = body["tools"][1]["function"]["name"]
        self.assertEqual(mapper.decode(encoded_namespace_name), ("mcp/sample", "lookup"))

    def test_build_deepseek_body_allows_env_override(self):
        old_model = adapter.os.environ.get("DEEPSEEK_MODEL")
        adapter.os.environ["DEEPSEEK_MODEL"] = "deepseek-v4-flash"
        mapper = adapter.ToolNameMapper()
        try:
            body = adapter.build_deepseek_body(
                {
                    "model": "deepseek-v4-pro",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "ping"}],
                        }
                    ],
                },
                mapper,
            )
        finally:
            if old_model is None:
                adapter.os.environ.pop("DEEPSEEK_MODEL", None)
            else:
                adapter.os.environ["DEEPSEEK_MODEL"] = old_model

        self.assertEqual(body["model"], "deepseek-v4-flash")

    def test_build_deepseek_body_drops_orphan_tool_call_history(self):
        mapper = adapter.ToolNameMapper()
        body = adapter.build_deepseek_body(
            {
                "model": "deepseek-v4-pro",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_orphan",
                        "name": "shell_command",
                        "arguments": "{\"cmd\":\"ls\"}",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "retry"}],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "shell_command",
                        "parameters": {"type": "object"},
                    }
                ],
            },
            mapper,
        )

        self.assertEqual(body["messages"], [{"role": "user", "content": "retry"}])

    def test_build_deepseek_body_pairs_tool_call_with_tool_output(self):
        mapper = adapter.ToolNameMapper()
        body = adapter.build_deepseek_body(
            {
                "model": "deepseek-v4-pro",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "namespace": "mcp/sample",
                        "arguments": "{\"query\":\"x\"}",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "interleaved"}],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "result",
                    },
                ],
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp/sample",
                        "tools": [
                            {
                                "type": "function",
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            }
                        ],
                    }
                ],
            },
            mapper,
        )

        self.assertEqual(body["messages"][0]["role"], "assistant")
        self.assertEqual(body["messages"][0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(body["messages"][1], {"role": "tool", "tool_call_id": "call_1", "content": "result"})
        self.assertEqual(body["messages"][2], {"role": "user", "content": "interleaved"})
        self.assertEqual(body["messages"][0]["tool_calls"][0]["function"]["name"], body["tools"][0]["function"]["name"])

    def test_chat_tool_call_becomes_responses_function_call(self):
        mapper = adapter.ToolNameMapper()
        encoded = mapper.add("lookup", namespace="mcp/sample")
        events = adapter.chat_completion_to_response_events(
            {
                "id": "chatcmpl_1",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": encoded,
                                        "arguments": "{\"query\":\"x\"}",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
            mapper,
        )

        self.assertEqual(events[0]["type"], "response.created")
        function_event = events[1]
        self.assertEqual(function_event["type"], "response.output_item.done")
        item = function_event["item"]
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["namespace"], "mcp/sample")
        self.assertEqual(item["name"], "lookup")
        self.assertEqual(item["call_id"], "call_abc")
        self.assertEqual(json.loads(item["arguments"]), {"query": "x"})
        self.assertEqual(events[-1]["response"]["end_turn"], False)

    def test_chat_text_becomes_assistant_message(self):
        events = adapter.chat_completion_to_response_events(
            {
                "id": "chatcmpl_2",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
            },
            adapter.ToolNameMapper(),
        )

        self.assertEqual(events[1]["item"]["type"], "message")
        self.assertEqual(events[1]["item"]["content"][0]["text"], "done")
        self.assertEqual(events[-1]["response"]["end_turn"], True)

    def test_streaming_accumulator_emits_deltas_and_final_message(self):
        accumulator = adapter.StreamingAccumulator(adapter.ToolNameMapper())

        events = accumulator.ingest(
            {
                "id": "chatcmpl_stream",
                "model": "deepseek-v4-pro",
                "choices": [{"delta": {"role": "assistant", "content": "he"}}],
                "usage": None,
            }
        )
        events += accumulator.ingest(
            {
                "id": "chatcmpl_stream",
                "model": "deepseek-v4-pro",
                "choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}],
                "usage": None,
            }
        )
        events += accumulator.ingest(
            {
                "id": "chatcmpl_stream",
                "model": "deepseek-v4-pro",
                "choices": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
        )
        final_events = accumulator.final_events()

        self.assertEqual(
            [event["delta"] for event in events],
            ["he", "llo"],
        )
        self.assertEqual(final_events[0]["item"]["content"][0]["text"], "hello")
        self.assertEqual(final_events[-1]["response"]["usage"]["total_tokens"], 3)
        self.assertEqual(final_events[-1]["response"]["end_turn"], True)

    def test_streaming_accumulator_reassembles_tool_call_arguments(self):
        mapper = adapter.ToolNameMapper()
        encoded = mapper.add("shell_command")
        accumulator = adapter.StreamingAccumulator(mapper)

        accumulator.ingest(
            {
                "id": "chatcmpl_tool",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": encoded,
                                        "arguments": "{\"cmd\":",
                                    },
                                }
                            ]
                        }
                    }
                ],
            }
        )
        accumulator.ingest(
            {
                "id": "chatcmpl_tool",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": "\"ls\"}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )
        final_events = accumulator.final_events()

        call = final_events[0]["item"]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(call["call_id"], "call_1")
        self.assertEqual(call["name"], "shell_command")
        self.assertEqual(json.loads(call["arguments"]), {"cmd": "ls"})
        self.assertEqual(final_events[-1]["response"]["end_turn"], False)


if __name__ == "__main__":
    unittest.main()
