# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest
from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset, format_and_template_dataset
from utils.datasets.chat_templates import keep_renderable_chat_template

_GEMMA4_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "assets" / "chat_templates" / "gemma-4.jinja"
)

_QWEN35_TEMPLATE = """
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' + (message.content or '') }}
{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}
{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
{%- for args_name, args_value in tool_call.arguments | items %}
{{- '<parameter=' + args_name + '>\\n' + args_value | string + '\\n</parameter>\\n' }}
{%- endfor %}
{{- '</function>\\n</tool_call>' }}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- endfor %}
"""

_LLAMA3_TEMPLATE = """
{%- for message in messages %}
{%- if not (message.role == 'tool' or 'tool_calls' in message) %}
{{- '<|start_header_id|>' + message.role + '<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' }}
{%- elif 'tool_calls' in message %}
{%- if not message.tool_calls|length == 1 %}{{- raise_exception('one tool call per message') }}{%- endif %}
{%- set tool_call = message.tool_calls[0].function %}
{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n{"name": "' + tool_call.name + '", "parameters": ' + tool_call.arguments | tojson + '}<|eot_id|>' }}
{%- else %}
{{- '<|start_header_id|>ipython<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' }}
{%- endif %}
{%- endfor %}
"""

_DEEPSEEK_TEMPLATE = """
{%- for message in messages %}
{%- if message['role'] == 'user' %}{{- '<User>' + message['content'] }}{%- endif %}
{%- if message['role'] == 'assistant' and message['content'] is none %}
{%- for tool in message['tool_calls'] %}
{{- '<call>' + tool['function']['name'] + '\\n' + tool['function']['arguments'] + '</call>' }}
{%- endfor %}
{%- endif %}
{%- if message['role'] == 'assistant' and message['content'] is not none %}{{- '<Assistant>' + message['content'] }}{%- endif %}
{%- if message['role'] == 'tool' %}{{- '<output>' + message['content'] }}{%- endif %}
{%- endfor %}
"""

_CATALOG_DEEPSEEK_TEMPLATE = (
    "{%- if tools %}{%- for tool in tools %}"
    "{{- '<catalog>' + tool.function.name + '</catalog>' }}"
    "{%- endfor %}{%- endif %}"
    + _DEEPSEEK_TEMPLATE
)

_MISTRAL_TOOL_ID_TEMPLATE = """
{%- for message in messages %}
{%- if message.role == 'assistant' and message.tool_calls %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.id is undefined or tool_call.id | length != 9 %}
{{- raise_exception('tool call id must be nine characters') }}
{%- endif %}
{{- '<call id=' + tool_call.id + '>' + tool_call.function.name + ':' + (tool_call.function.arguments | tojson) + '</call>' }}
{%- endfor %}
{%- elif message.role == 'tool' %}
{%- if message.tool_call_id is undefined %}
{{- raise_exception('tool result must have a tool call id') }}
{%- endif %}
{{- '<result id=' + message.tool_call_id + ' name=' + message.name + '>' + message.content + '</result>' }}
{%- else %}{{- message.content or '' }}
{%- endif %}
{%- endfor %}
"""

_TOOLS_TEMPLATE = """
{%- if tools %}
{%- for tool in tools %}{{- '<tool>' + tool.function.name + ':' + (tool.function.parameters | tojson) + '</tool>' }}{%- endfor %}
{%- endif %}
{%- for message in messages %}
{{- '<' + message.role + '>' + (message.content or '') }}
{%- for call in message.tool_calls or [] %}{{- '<call>' + call.function.name + ':' + (call.function.arguments | tojson) + '</call>' }}{%- endfor %}
{%- endfor %}
"""

_FIRST_TOOL_ONLY_TEMPLATE = """
{%- for message in messages %}
{%- if message.role == 'assistant' and message.tool_calls %}
{%- set call = message.tool_calls[0].function %}{{- '<call>' + call.name + ':' + (call.arguments | tojson) + '</call>' }}
{%- elif message.role == 'tool' %}{{- '<result>' + message.content + '</result>' }}
{%- else %}{{- '<' + message.role + '>' + (message.content or '') }}
{%- endif %}
{%- endfor %}
"""

_VLM_TOOL_TEMPLATE = """
{%- for message in messages %}
{%- if message.role == 'assistant' and message.tool_calls %}
{%- for call in message.tool_calls %}{{- '<call>' + call.function.name + ':' + call.function.arguments + '</call>' }}{%- endfor %}
{%- elif message.role == 'tool' %}{{- '<result>' + message.content + '</result>' }}
{%- else %}{{- '<' + message.role + '>' + message.content[0].text }}
{%- endif %}
{%- endfor %}
"""

_REASONING_TOOL_TEMPLATE = """
{%- for message in messages %}
{%- if message.role == 'assistant' and message.tool_calls %}
{{- '<reasoning>' + (message.reasoning_content or '') + '</reasoning>' }}
{%- for call in message.tool_calls %}{{- '<call>' + call.function.name + ':' + (call.function.arguments | tojson) + '</call>' }}{%- endfor %}
{%- elif message.role == 'tool' %}{{- '<result>' + message.content + '</result>' }}
{%- else %}{{- '<' + message.role + '>' + (message.content or '') }}
{%- endif %}
{%- endfor %}
"""


class _JinjaTokenizer:
    eos_token = ""

    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(
        self,
        conversation,
        tokenize = False,
        add_generation_prompt = False,
        **_kwargs,
    ):
        jinja2 = pytest.importorskip("jinja2")
        sandbox = pytest.importorskip("jinja2.sandbox")

        def _raise(message):
            raise jinja2.exceptions.TemplateError(message)

        env = sandbox.ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
        env.filters["tojson"] = lambda value, **opts: json.dumps(value, **opts)
        env.globals["raise_exception"] = _raise
        return env.from_string(self.chat_template).render(
            messages = conversation,
            tools = _kwargs.get("tools"),
            add_generation_prompt = add_generation_prompt,
            bos_token = "",
        )


class _VLMJinjaTokenizer(_JinjaTokenizer):
    image_processor = object()


def _tool_call_row(arguments):
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "content": "21C", "tool_call_id": "call_0"},
        {"role": "assistant", "content": "It is 21C in Paris."},
    ]


def _plain_row():
    return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]


def _format(rows, template):
    dataset_info = {
        "dataset": Dataset.from_list([{"messages": row} for row in rows]),
        "detected_format": "chatml_messages",
        "final_format": "chatml_messages",
        "chat_column": "messages",
        "is_standardized": True,
        "warnings": [],
    }
    return apply_chat_template_to_dataset(dataset_info, _JinjaTokenizer(template), num_proc = 1)


def test_json_string_tool_arguments_are_rendered_as_parameters():
    result = _format([_tool_call_row('{"city": "Paris", "unit": null}')], _QWEN35_TEMPLATE)

    assert result["success"] is True
    text = result["dataset"][0]["text"]
    assert "<parameter=city>\nParis\n</parameter>" in text
    assert "<parameter=unit>\nNone\n</parameter>" in text


def test_gemma4_learns_its_own_tool_call_format():
    result = _format(
        [_tool_call_row('{"city": "Paris"}')], _GEMMA4_TEMPLATE.read_text(encoding = "utf-8")
    )

    assert result["success"] is True
    assert 'call:get_weather{city:<|"|>Paris<|"|>}' in result["dataset"][0]["text"]


def test_tool_call_dataset_with_null_filled_keys_renders_on_llama3():
    rows = [_plain_row(), _tool_call_row('{"city": "Paris"}')]

    result = _format(rows, _LLAMA3_TEMPLATE)

    assert result["success"] is True, result["errors"]
    texts = result["dataset"]["text"]
    assert len(texts) == 2
    assert "<|start_header_id|>assistant<|end_header_id|>\n\nHello!" in texts[0]
    assert '{"name": "get_weather", "parameters": {"city": "Paris"}}' in texts[1]


def test_dict_arguments_do_not_gain_other_tools_null_parameters():
    weather = _tool_call_row({"city": "Paris"})
    search = _tool_call_row({"query": "cats"})
    search[1]["tool_calls"][0]["function"]["name"] = "web_search"

    result = _format([weather, search], _QWEN35_TEMPLATE)

    assert result["success"] is True
    weather_text, search_text = result["dataset"]["text"]
    assert "<parameter=query>" not in weather_text
    assert "<parameter=city>" not in search_text


def test_non_json_tool_arguments_are_kept_verbatim():
    result = _format([_tool_call_row("not json")], _LLAMA3_TEMPLATE)

    assert result["success"] is True
    assert '"parameters": "not json"' in result["dataset"][0]["text"]


def test_tool_call_turn_with_null_content_and_string_arguments_still_renders():
    row = _tool_call_row('{"city": "Paris"}')
    row[1]["content"] = None

    result = _format([row, _plain_row()], _DEEPSEEK_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert len(result["dataset"]) == 2
    assert '<call>get_weather\n{"city": "Paris"}</call>' in result["dataset"][0]["text"]


def test_null_content_is_dropped_before_the_row_as_loaded_is_tried():
    template = (
        "{%- for message in messages %}"
        "{%- if message.content is defined %}[{{ message.content }}]{%- else %}-{%- endif %}"
        "{%- for call in message.tool_calls or [] %}<call>{{ call.function.name }}</call>{%- endfor %}"
        "{%- endfor %}"
    )
    null_content = _tool_call_row('{"city": "Paris"}')
    null_content[1]["content"] = None
    null_content[1]["tool_calls"][0]["function"]["arguments"] = {}
    empty_content = _tool_call_row('{"city": "Paris"}')
    empty_content[1]["tool_calls"][0]["function"]["arguments"] = {}

    result = _format([null_content, empty_content], template)

    assert result["dataset"]["text"] == [
        "[Weather in Paris?]-<call>get_weather</call>[21C][It is 21C in Paris.]",
        "[Weather in Paris?][]<call>get_weather</call>[21C][It is 21C in Paris.]",
    ]


def test_parallel_tool_calls_fall_back_to_single_call_turns():
    parallel = _tool_call_row('{"city": "Paris"}')
    parallel[1]["tool_calls"].append({**parallel[1]["tool_calls"][0], "id": "call_1"})

    result = _format([_plain_row(), parallel], _LLAMA3_TEMPLATE)

    assert result["success"] is True
    assert result["dropped_rows_warning"] is None
    assert len(result["dataset"]) == 2
    assert result["dataset"][1]["text"].count('"name": "get_weather"') == 2


def test_template_probe_counts_rows_after_cleaning():
    tokenizer = _JinjaTokenizer(_LLAMA3_TEMPLATE)
    dataset = Dataset.from_list(
        [{"messages": _plain_row()}, {"messages": _tool_call_row('{"city": "Paris"}')}]
    )

    note = keep_renderable_chat_template(tokenizer, dataset, "messages", _QWEN35_TEMPLATE)

    assert note is None
    assert tokenizer.chat_template == _LLAMA3_TEMPLATE


def _sharegpt_tool_row(call):
    return {
        "conversations": [
            {"from": "human", "value": "Weather in Paris?"},
            {"from": "function_call", "value": call},
            {"from": "observation", "value": '{"temp": 18}'},
            {"from": "gpt", "value": "It is 18C in Paris."},
        ]
    }


def _format_sharegpt(
    rows,
    template,
    model_name = "stub-model",
):
    return format_and_template_dataset(
        Dataset.from_list(rows),
        model_name = model_name,
        tokenizer = _JinjaTokenizer(template),
        num_proc = 1,
    )


def test_sharegpt_function_call_and_observation_train_as_tool_turns():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], _LLAMA3_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert (
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        '{"name": "get_weather", "parameters": {"city": "Paris"}}' in text
    )
    assert '<|start_header_id|>ipython<|end_header_id|>\n\n{"temp": 18}' in text
    assert "function_call" not in text
    assert "observation" not in text


def test_template_that_ignores_tool_calls_drops_the_row():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    template = (
        "{%- for message in messages %}"
        "{{- '<' + message.role + '>' + (message.content or '') }}"
        "{%- endfor %}"
    )

    result = _format_sharegpt([_sharegpt_tool_row(call)], template)

    assert result["success"] is False
    assert "did not serialize every tool call" in result["errors"][0]


def test_template_that_drops_nonempty_arguments_drops_the_row():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    template = """
{%- for message in messages %}
{%- for call in message.tool_calls or [] %}{{- '<call>' + call.function.name + '</call>' }}{%- endfor %}
{%- endfor %}
"""

    result = _format_sharegpt([_sharegpt_tool_row(call)], template)

    assert result["success"] is False
    assert "did not serialize every tool call" in result["errors"][0]


def test_template_that_drops_tool_results_drops_the_row():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    template = """
{%- for message in messages %}
{%- for call in message.tool_calls or [] %}
{{- '<call>' + call.function.name + ':' + (call.function.arguments | tojson) + '</call>' }}
{%- endfor %}
{%- endfor %}
"""

    result = _format_sharegpt([_sharegpt_tool_row(call)], template)

    assert result["success"] is False
    assert "did not serialize every tool result" in result["errors"][0]


def test_overlapping_call_names_do_not_mask_a_dropped_call():
    call = json.dumps(
        [
            {"name": "search", "arguments": {"query": "weather"}},
            {"name": "search_web", "arguments": {"query": "weather"}},
        ]
    )
    row = _sharegpt_tool_row(call)
    row["conversations"].insert(3, {"from": "observation", "value": '{"hits": 1}'})
    template = """
{%- for message in messages %}
{%- for call in message.tool_calls or [] %}
{%- if call.function.name == 'search_web' %}{{- '<call>' + call.function.name + '</call>' }}{%- endif %}
{%- endfor %}
{%- endfor %}
"""

    result = _format_sharegpt([row], template)

    assert result["success"] is False
    assert "did not serialize every tool call" in result["errors"][0]


def test_probe_marker_does_not_overlap_the_function_name():
    call = json.dumps({"name": "call", "arguments": {"city": "Paris"}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], _QWEN35_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert "<function=call>" in result["dataset"][0]["text"]


def test_sharegpt_function_call_list_trains_every_call():
    call = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": '{"city": "Rome"}'},
        ]
    )

    result = _format_sharegpt([_sharegpt_tool_row(call)], _QWEN35_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert "<|im_start|>assistant\n<tool_call>\n<function=get_weather>" in text
    assert "<parameter=city>\nParis\n</parameter>" in text
    assert "<parameter=city>\nRome\n</parameter>" in text
    assert '<|im_start|>tool\n{"temp": 18}' in text


def test_sharegpt_function_call_list_falls_back_for_single_call_templates():
    call = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Rome"}},
        ]
    )
    row = _sharegpt_tool_row(call)
    row["conversations"].insert(3, {"from": "observation", "value": '{"temp": 24}'})

    result = _format_sharegpt([row], _LLAMA3_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert text.count("<|start_header_id|>assistant<|end_header_id|>") == 3
    assert text.count("<|start_header_id|>ipython<|end_header_id|>") == 2
    assert '"parameters": {"city": "Paris"}' in text
    assert '"parameters": {"city": "Rome"}' in text


def test_gpt_oss_splits_parallel_calls_before_a_silent_first_call_render():
    call = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Rome"}},
        ]
    )
    row = _sharegpt_tool_row(call)
    row["conversations"].insert(3, {"from": "observation", "value": '{"temp": 24}'})

    result = _format_sharegpt([row], _FIRST_TOOL_ONLY_TEMPLATE, model_name = "openai/gpt-oss-20b")

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert text.count("<call>get_weather:") == 2
    assert '"city": "Paris"' in text
    assert '"city": "Rome"' in text
    assert text.count("<result>") == 2


def test_parallel_call_split_does_not_repeat_reasoning():
    call = '<think>Compare both cities.</think>' + json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Rome"}},
        ]
    )
    row = _sharegpt_tool_row(call)
    row["conversations"].insert(3, {"from": "observation", "value": '{"temp": 24}'})

    result = _format_sharegpt(
        [row], _REASONING_TOOL_TEMPLATE, model_name = "openai/gpt-oss-20b"
    )

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert text.count("<reasoning>Compare both cities.</reasoning>") == 1
    assert text.count("<call>get_weather:") == 2


def test_vlm_processor_unwraps_sharegpt_tool_text_blocks():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})

    result = format_and_template_dataset(
        Dataset.from_list([_sharegpt_tool_row(call)]),
        model_name = "stub-vlm",
        tokenizer = _VLMJinjaTokenizer(_VLM_TOOL_TEMPLATE),
        num_proc = 1,
    )

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert '<call>get_weather:{"city": "Paris"}</call>' in text
    assert '<result>{"temp": 18}</result>' in text


@pytest.mark.parametrize(
    "wrapped_call, reasoning",
    [
        (
            '<think>Check the requested city.</think>{"name":"get_weather","arguments":{"city":"Paris"}}',
            "Check the requested city.",
        ),
        (
            'Check the requested city.<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            "Check the requested city.",
        ),
    ],
)
def test_reasoning_wrapped_sharegpt_function_calls_are_decoded(wrapped_call, reasoning):
    result = _format_sharegpt([_sharegpt_tool_row(wrapped_call)], _REASONING_TOOL_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert f"<reasoning>{reasoning}</reasoning>" in text
    assert '<call>get_weather:{"city": "Paris"}</call>' in text
    assert '<result>{"temp": 18}</result>' in text


def test_content_only_template_preserves_wrapped_tool_reasoning():
    call = (
        '<think>Paris</think>'
        '{"name":"get_weather","arguments":{"city":"Paris"}}'
    )

    result = _format_sharegpt([_sharegpt_tool_row(call)], _QWEN35_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert "<|im_start|>assistant\nParis<tool_call>" in text
    assert "<function=get_weather>" in text


def test_catalog_name_does_not_mask_a_dropped_tool_call():
    call = (
        '<think>Check the requested city.</think>'
        '{"name":"get_weather","arguments":{"city":"Paris"}}'
    )
    row = _sharegpt_tool_row(call)
    row["tools"] = json.dumps(
        [{"name": "get_weather", "parameters": {"type": "object"}}]
    )

    result = _format_sharegpt([row], _CATALOG_DEEPSEEK_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert '<call>get_weather\n{"city": "Paris"}</call>' in text


@pytest.mark.parametrize("tag", ["think", "tool_call"])
def test_literal_reasoning_tags_inside_tool_arguments_are_preserved(tag):
    call = json.dumps(
        {"name": "web_search", "arguments": {"query": f"<{tag}>draft</{tag}>"}}
    )

    result = _format_sharegpt([_sharegpt_tool_row(call)], _DEEPSEEK_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert (
        f'<call>web_search\n{{"query": "<{tag}>draft</{tag}>"}}</call>'
        in result["dataset"][0]["text"]
    )


def test_sharegpt_function_call_keeps_null_content_for_deepseek_templates():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], _DEEPSEEK_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert '<call>get_weather\n{"city": "Paris"}</call>' in result["dataset"][0]["text"]


def test_sharegpt_function_call_preserves_unicode_arguments():
    call = json.dumps(
        {"name": "get_weather", "arguments": {"city": "München"}},
        ensure_ascii = False,
    )

    result = _format_sharegpt([_sharegpt_tool_row(call)], _DEEPSEEK_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert '"city": "München"' in text
    assert "\\u00fc" not in text


def test_sharegpt_row_tool_catalog_is_decoded_and_rendered():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    row = _sharegpt_tool_row(call)
    row["tools"] = json.dumps(
        [
            {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "default": None}},
                    "required": ["city"],
                },
            }
        ]
    )

    result = _format_sharegpt([row], _TOOLS_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert '<tool>get_weather:{"type": "object"' in text
    assert '"default": null' in text
    assert "<assistant>" in text

    gemma_result = _format_sharegpt([row], _GEMMA4_TEMPLATE.read_text(encoding = "utf-8"))
    assert gemma_result["success"] is True, gemma_result["errors"]
    assert "declaration:get_weather" in gemma_result["dataset"][0]["text"]


def test_mixed_flat_and_wrapped_tool_catalogs_ignore_arrow_null_fields():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    schema = {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    flat = _sharegpt_tool_row(call)
    flat["tools"] = [schema]
    wrapped = _sharegpt_tool_row(call)
    wrapped["tools"] = [{"type": "function", "function": schema}]

    result = _format_sharegpt([flat, wrapped], _TOOLS_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert len(result["dataset"]) == 2
    assert all("<tool>get_weather:" in text for text in result["dataset"]["text"])


def test_structured_tool_catalogs_drop_arrow_null_schema_fields():
    weather = _sharegpt_tool_row(
        json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    )
    weather["tools"] = [
        {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    search = _sharegpt_tool_row(
        json.dumps({"name": "web_search", "arguments": {"query": "weather"}})
    )
    search["tools"] = [
        {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    result = _format_sharegpt([weather, search], _TOOLS_TEMPLATE)

    assert result["success"] is True, result["errors"]
    weather_text, search_text = result["dataset"]["text"]
    assert '"query": null' not in weather_text
    assert '"city": null' not in search_text


def test_sharegpt_tool_result_gets_the_call_id_and_name_required_by_mistral():
    call = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})

    result = _format_sharegpt([_sharegpt_tool_row(call)], _MISTRAL_TOOL_ID_TEMPLATE)

    assert result["success"] is True, result["errors"]
    text = result["dataset"][0]["text"]
    assert '<call id=call00000>get_weather:{"city": "Paris"}</call>' in text
    assert '<result id=call00000 name=get_weather>{"temp": 18}</result>' in text


def test_sharegpt_function_call_that_is_not_json_is_kept_as_written():
    result = _format_sharegpt([_sharegpt_tool_row("get_weather(Paris)")], _QWEN35_TEMPLATE)

    assert result["success"] is True, result["errors"]
    assert "<|im_start|>function_call\nget_weather(Paris)" in result["dataset"][0]["text"]
