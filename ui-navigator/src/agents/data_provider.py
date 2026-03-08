import json

from src.config import config
from src.utils import gemini_client, strip_code_fences
from src.models.action import ActionDoc, ActionType

DATA_PROMPT = """<role>You are a test data generator producing contextually appropriate values for automated UI testing.</role>

<context>
Page: {screen_description}
URL: {target_url}
Field label: {label}
Field role: {role}
Field selector: {selector}
Test scenario: {scenario}
</context>

<task>Generate a single fill value for this field that is contextually appropriate given the page purpose and field type.</task>

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



async def provide_fill_value(
    action: ActionDoc,
    screen_description: str,
    target_url: str,
) -> ActionDoc:
    """Generate contextually relevant test data for a FILL action.

    Called by the orchestrator before passing a FILL action to the Navigator.
    Uses screen_description from Page Analyzer to understand page context.
    """
    if action.type != ActionType.FILL:
        return action

    if action.fill_value:
        return action

    elem = action.element_info
    prompt = DATA_PROMPT.format(
        label=elem.text or "unknown field",
        role=elem.role or "input",
        selector=elem.selector or "unknown",
        scenario=action.scenario.value,
        screen_description=screen_description or "unknown page",
        target_url=target_url,
    )

    response = gemini_client.models.generate_content(
        model=config.GEMINI_LITE_MODEL,
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
    )

    raw = strip_code_fences(response.text)
    try:
        data = json.loads(raw)
        fill_value = str(data.get("fill_value", ""))
    except (json.JSONDecodeError, AttributeError):
        fill_value = raw.strip('"').strip("'")

    return action.model_copy(update={"fill_value": fill_value})
