#!/bin/bash

set -e
jinja2_template=""

# Script to generate jinja2 template for chat models
jinja2_template+="{%- if not date_string is defined %}"
jinja2_template+="{%- set date_string = \"01 Oct 2025\" %}"
jinja2_template+="{%- endif %}"
jinja2_template+="{%- if not tools is defined %}"
jinja2_template+="{%- set tools = none %}"
jinja2_template+="{%- endif %}"

jinja2_template+="{%- if messages[0]['role'] == 'system' %}"
jinja2_template+="{%- set system_message = messages[0]['content'] | trim %}"
jinja2_template+="{%- set messages = messages[1:] %}"
jinja2_template+="{%- else %}"
jinja2_template+="{%- set system_message = \"You are a helpful assistant.\" %}"
jinja2_template+="{%- endif %}"


# This script generates the chat_template jinja2 template based on the provided configuration file.
jinja2_template+="{{- bos_token }}"
jinja2_template+="{{- \"<|start_header_id|>system<|end_header_id|>\" }}"
jinja2_template+="{{- \"Cutting Knowledge Date: 01 Oct 2025\n\" }}"
jinja2_template+="{{- \"Today Date: \" + date_string + \"\n\" }}"

jinja2_template+="{%- if tools is not none and tools|length > 0 %}"
jinja2_template+="{{- \"You may call one or more functions to assist with the user query.\n\" }}"
jinja2_template+="{{- \"You are provided with function signatures within <tools></tools> XML tags:\n\" }}"
jinja2_template+="{{- \"<tools>\n\" }}"
jinja2_template+="{%- for tool in tools %}"
jinja2_template+="{{- tool | tojson + \"\n\" }}"
jinja2_template+="{{- \"\n\n\" }}"
jinja2_template+="{%- endfor %}"
jinja2_template+="{{- \"</tools>\n\n\" }}"
jinja2_template+="{{- \"For each function call, return a json object with function name and arguments within <tool_calls></tool_calls> XML tags:\n\" }}"
jinja2_template+="{{- \"<tool_calls>\n\" }}"
jinja2_template+="{{- '{\"name\": <function-name>, \"arguments\": <args-json-object>}' }}"
jinja2_template+="{{- \"</tool_calls>\n\n\" }}"
jinja2_template+="{%- endif %}"

jinja2_template+="{{- system_message + \"<|eot_id|>\" }}"

# Iterate over messages and format them
jinja2_template+="{%- for message in messages %}"
jinja2_template+="{%- if (message['role'] == 'user') or (message['role'] == 'system' and not loop.first ) or (message['role'] == 'assistant' and not message.tool_calls) %}"
jinja2_template+="<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
jinja2_template+="{%- if message['content'] %}"
jinja2_template+="{{ message['content'] | trim }}"
jinja2_template+="{% else %}"
jinja2_template+="{%- for content in message['content'] %}"
jinja2_template+="{%- if 'text' in content %}{{ content['text'] }}{%- endif %}"
jinja2_template+="{%- endfor %}"
jinja2_template+="{%- endif %}"
jinja2_template+="<|eot_id|>"
jinja2_template+="{%- elif message['role'] == 'assistant' %}"
jinja2_template+="{{- \"<|start_header_id|>assistant<|end_header_id|>\n\n\" }}"
jinja2_template+="{%- if message.content %}"
jinja2_template+="{{ message['content'] }}"
jinja2_template+="{%- endif %}"
jinja2_template+="{%- for tool_call in message.tool_calls %}"
jinja2_template+="{%- if tool_call.function is defined %}{%- set tool_call=tool_call.function %}{%- endif %}"
jinja2_template+="<tool_call>\n"
jinja2_template+="{\"name\": \"{{ tool_call.name }}\", \"arguments\": {{ tool_call.arguments | tojson }}} {{- "\n" }}"
jinja2_template+="</tool_call>\n"
jinja2_template+="{%- endfor %}"

jinja2_template+="{%- endif %}"
jinja2_template+="{%- endfor %}"
# jinja2_template+="{{- \"<|start_header_id|>\" + message['role'] + \"<|end_header_id|>\n\n\" + message['content'] | trim + \"<|eot_id|>\" }}"

# If should add generation prompt, add it
jinja2_template+="{%- if add_generation_prompt %}"
jinja2_template+="{{- \"<|start_header_id|>assistant<|end_header_id|>\n\n\" }}"
jinja2_template+="{%- endif %}"
jinja2_template="  $jinja2_template"

# echo "  $jinja2_template"

if [ "$#" -ne 1 ]; then
    echo $jinja2_template
else
    escaped=$(printf '%s\n' "$jinja2_template" | sed -e 's/[\/&]/\\&/g')
    awk -v jinja2_template="$escaped" 'NR==2{$0=jinja2_template}1' $1 > tmpfile && mv tmpfile $1
fi
