"""Data Provider Agent — generates test data for form fills.

Converted to ADK LlmAgent (Agent) so the LLM call is auto-traced.

The orchestrator sets session state before invoking:
    - dp_action: ActionDoc
    - dp_description: str
    - dp_target_url: str

Output stored via output_key="dp_result_raw".
"""

import json

from google.adk.agents import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.config import config
from src.models.action import ActionType
from src.utils import strip_code_fences


DATA_INSTRUCTION = """<role>You are a test data generator producing contextually appropriate values for automated UI testing.</role>

<task>Generate a single fill value for the field described in the user message that is contextually appropriate given the page purpose and field type.</task>

<thinking_process>
Reason through these steps:
1. PAGE UNDERSTANDING — What is this page for? What kind of data does it expect from users?
2. FIELD PURPOSE — Based on the label, role, and selector, what type of data does this specific field accept?
3. SCENARIO APPLICATION — Apply the test scenario to determine what kind of value to generate.
</thinking_process>

<scenario_rules>
- "positive" — Generate valid data that would pass typical validation rules. Use realistic formats appropriate for the field type (valid email format for email fields, reasonable length for name fields, properly formatted numbers for numeric fields).
- "negative" — Generate data designed to trigger validation errors. Use strategies such as: empty strings, values exceeding expected length, incorrect formats (letters in numeric fields, malformed emails), boundary values (zero, negative numbers), or special characters that test input handling.
- "neutral" — Generate reasonable default data that represents typical usage.
</scenario_rules>

<output_format>
Return ONLY: {{"fill_value": "<the generated value>"}}
</output_format>"""


def _dp_before_model_callback(callback_context, llm_request):
    """Inject field context as the user message."""
    state = callback_context.state
    action = state.get("dp_action")
    screen_description = state.get("dp_description", "")
    target_url = state.get("dp_target_url", "")

    if isinstance(action, dict):
        from src.models.action import ActionDoc
        action = ActionDoc(**action)
        state["dp_action"] = action

    if not action or action.type != ActionType.FILL:
        skip = json.dumps({"fill_value": ""})
        state["dp_result_raw"] = skip
        return LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=skip)]
            ),
        )

    if action.fill_value:
        skip = json.dumps({"fill_value": action.fill_value})
        state["dp_result_raw"] = skip
        state["dp_result"] = action.model_dump(mode="json")
        return LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=skip)]
            ),
        )

    elem = action.element_info
    context_msg = (
        f"Page: {screen_description}\n"
        f"URL: {target_url}\n"
        f"Field label: {elem.text or 'unknown field'}\n"
        f"Field role: {elem.role or 'input'}\n"
        f"Field selector: {elem.selector or 'unknown'}\n"
        f"Test scenario: {action.scenario.value}"
    )

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


def _dp_after_model_callback(callback_context, llm_response):
    """Parse the LLM response and store the updated ActionDoc."""
    state = callback_context.state
    action = state.get("dp_action")

    raw = ""
    if llm_response.content and llm_response.content.parts:
        raw = llm_response.content.parts[0].text or ""
    raw = strip_code_fences(raw)

    try:
        data = json.loads(raw)
        fill_value = str(data.get("fill_value", ""))
    except (json.JSONDecodeError, AttributeError):
        fill_value = raw.strip('"').strip("'")

    if isinstance(action, dict):
        from src.models.action import ActionDoc
        action = ActionDoc(**action)

    if action:
        state["dp_result"] = action.model_copy(
            update={"fill_value": fill_value}
        ).model_dump(mode="json")
    return None


data_provider_agent = Agent(
    name="data_provider",
    description="Generates contextually appropriate test data for form fills.",
    model=config.GEMINI_LITE_MODEL,
    instruction=DATA_INSTRUCTION,
    output_key="dp_result_raw",
    before_model_callback=_dp_before_model_callback,
    after_model_callback=_dp_after_model_callback,
)
