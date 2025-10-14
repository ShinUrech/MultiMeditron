#!/bin/bash

set -e

# Set the multline jinja2_template variable
# For information on the template check `sglang/srt/function_call/llama32_detector.py` for details
# on the expected tool call format
jinja2_template=$(cat << 'EOF'
; ============= Jinja2 Template for Chat Models =============
{%- if not date_string is defined %}
  {%- set date_string = "01 Oct 2025" %}
{%- endif %}
{%- if not tools is defined %}
  {%- set tools = none %}
{%- endif %}
{%- if messages[0]['role'] == 'system' %}
  {%- set system_message = messages[0]['content'] | trim %}
  {%- set messages = messages[1:] %}
{%- else %}
  {%- set system_message = "You are a helpful assistant." %}
{%- endif %}

; ============= System Prompt =============
<|start_header_id|>system<|end_header_id|>\n
Cutting Knowledge Date: 01 Oct 2025\n
Today Date: {{ date_string }}\n
{{ "\\n" + system_message | trim + "\\n\\n" }}
{%- if tools is not none and tools|length > 0 %}
  # Tools\n
  You may call one or more functions to assist with the user query.\n
  Here are the available tools:
  {{ "\\n```\\n[\\n" }}
  {%- for tool in tools %}
  {{ tool | tojson | indent(2, true) + ",\\n" }}
  {%- endfor %}
  {{ "]\\n```\\n\\n" }}
  For each function call, start with the <|python_tag|> tag, then provide a JSON object with the function name and arguments. Separate multiple function calls with a semicolon `;`.\n
  <|python_tag|>{"name": <function-name>, "arguments": <args-json-object>}<|eot_id|>\n
{%- endif %}

; ============= Chat Messages =============
{%- for message in messages %}
  {%- if (message['role'] == 'user') or (message['role'] == 'system') or (message['role'] == 'assistant' and not message.tool_calls) %}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>{{ "\\n\\n" }}
    {%- if message['content'] %}
      {{ message['content'] | trim }}
    {%- else %}
      {%- for content in message['content'] %}
        {%- if 'text' in content %}{{ content['text'] + "\\n" }}{%- endif %}
      {%- endfor %}
    {%- endif %}<|eot_id|>
  {%- elif message['role'] == 'assistant' %}
    <|start_header_id|>assistant<|end_header_id|>{{ "\\n\\n" }}
    {%- if message.content and message.content | trim | length > 0 %}
      {{ message['content'] | trim + "\\n\\n" }}
    {%- endif %}
    {%- for tool_call in message.tool_calls %}
      {%- if tool_call.function is defined %}
        {%- set tool_call=tool_call.function %}
      {%- endif %}
      <|python_tag|>{"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments | tojson }}}
      {%- if not loop.last %};{%- endif %}
    {%- endfor %}
    <|eot_id|>
  {%- elif message['role'] == 'tool' %}
    <|start_header_id|>ipython<|end_header_id|>{{ "\\n\\n" }}
    {%- if message.content is mapping or message.content is iterable %}
      {{ message['content'] | tojson }}
    {%- else %}
      {{ message['content'] | trim }}
    {%- endif %}<|eot_id|>
  {%- endif %}
{%- endfor %}

; ============= Generation Prompt =============
{%- if add_generation_prompt %}
  <|start_header_id|>assistant<|end_header_id|>{{ "\\n\\n" }}
{%- endif %}

EOF
)

# Remove all lines starting with ;
jinja2_template=$(echo "$jinja2_template" | sed '/^;/d')

# Remove all leading spaces
jinja2_template=$(echo "$jinja2_template" | sed 's/^[ \t]*//')

# Remove all literal new lines
jinja2_template=$(echo "$jinja2_template" | tr -d '\n')

# Escape all quotes and single quotes
jinja2_template=$(echo "$jinja2_template" | sed 's/"/\\"/g')

# A small python snippet to test rendering the jinja2 template
python_code=$(cat << 'PYTHON_EOF'
import jinja2, os

jinja2_template = os.environ['jinja2_template']
template = jinja2.Template(jinja2_template)
rendered = template.render(
    date_string="27 Apr 2024",
    tools=[
        {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}},
    ],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in Boston?"},
        {"role": "assistant", "content": "I can help with that! Let me check the weather for you.", "tool_calls": [{"function": {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "celsius"}}}, {"function": {"name": "pretty_display", "arguments": {"data": {"temperature": 22}}}}]},
        {"role": "tool", "content": {"temperature": 22, "unit": "celsius", "description": "clear sky"}},
        {"role": "assistant", "content": "The current weather in Boston is 22 degrees Celsius with clear skies."},
    ],
    add_generation_prompt=True,
    bos_token="",
)
print(rendered)
PYTHON_EOF
)

if [ "$#" -ne 1 ]; then
    printf "%s" "$jinja2_template"
    echo

    # Render the jinja2 template to verify it's correct
    echo "----- Rendered Template Preview -----"

    # Replace all '\n' with actual new lines for rendering
    jinja2_template=$(printf "%b" "$jinja2_template" | sed 's/\\"/"/g')

    export jinja2_template
    python3 -c "$python_code"

    echo "----- End of Rendered Template Preview -----"
else
    escaped=$(printf 'custom_chat_template: "%s"' "$jinja2_template" | sed -e 's/[\/&]/\\&/g')
    printf '%s\n' "$escaped"
    awk -v jinja2_template="$escaped" 'NR==1{$0=jinja2_template}1' $1 > tmpfile && mv tmpfile $1
fi
