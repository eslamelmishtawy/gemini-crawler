"""Page Analyzer — dual vision + DOM analysis with ParallelAgent.

Architecture:
    page_analyzer (SequentialAgent)
    ├── parallel_extraction (ParallelAgent)
    │   ├── vision_analyzer (Agent — multimodal via before_model_callback)
    │   └── dom_analyzer (Agent/LlmAgent)
    └── merger (Agent/LlmAgent)

State keys set by orchestrator before invoking:
    - pa_screenshot: bytes
    - pa_page_source: str

State keys set after completion:
    - pa_screen: ScreenDoc
    - pa_description: str
    - pa_actions: list[ActionDoc]
"""

import base64
import json
import uuid

from google.adk.agents import (
    Agent, ParallelAgent, SequentialAgent,
)
from google.genai import types

from src.config import config
from src.utils import strip_code_fences
from src.models.screen import ScreenDoc, ScreenStatus
from src.models.action import (
    ActionDoc, ActionType, ActionStatus, ActionScenario, ElementInfo,
)

VALID_ACTION_TYPES = [t.value for t in ActionType]
VALID_SCENARIOS = [s.value for s in ActionScenario]


# -------------------------------------------------------------------
# Vision Analyzer — Agent (multimodal via before_model_callback)
# -------------------------------------------------------------------

VISION_PROMPT = f"""<role>You are a UI element extraction specialist \
analyzing a screenshot of a web or mobile screen.</role>

<task>Extract EVERY visible element from this screenshot. \
Achieve exhaustive coverage — enumerate all elements without \
skipping any, even if they appear similar to each other.</task>

<categories>
Classify each element into one of two categories:
1. INTERACTIVE — buttons, inputs, links, toggles, dropdowns, \
tabs, navigation items, or any element a user can act on.
2. ASSERTION — page titles, headings, labels, status text, \
error messages, prices, counts, or any element that displays \
information.
</categories>

<element_fields>
For each element, return:
- type: one of {VALID_ACTION_TYPES}
- label: a human-readable description of the element's purpose
- text: the actual text content displayed on screen
- bounds: {{x, y, width, height}} in pixels relative to the screenshot
- visible: true if clearly visible and interactable, false if \
obscured or disabled
- scenario: one of {VALID_SCENARIOS}
</element_fields>

<scenario_logic>
Determine the scenario by reasoning about how this element would \
be used in testing:
- "positive" — interaction that follows the intended happy path
- "negative" — interaction designed to test error handling
- "neutral" — no pass/fail implication

Apply these rules:
- Assertion elements are always "neutral".
- Input fields should generate BOTH a "positive" and a "negative" \
entry as separate elements.
- Buttons and links are typically "positive" unless their purpose \
is explicitly error-related.
</scenario_logic>

<additional_fields>
Also return:
- screen_description: a structured description using this exact \
format: "[Page Title] | [Screen Type] | [Key Structural Elements] \
| [User Purpose]"
- resolution: {{width, height}} of the screenshot in pixels
</additional_fields>

<output_format>
Return ONLY valid JSON matching this structure:
{{
  "screen_description": "<Page Title | Screen Type | Key Elements \
| Purpose>",
  "resolution": {{"width": <int>, "height": <int>}},
  "elements": [
    {{
      "type": "<element type>",
      "label": "<human-readable description>",
      "text": "<displayed text>",
      "bounds": {{"x": <int>, "y": <int>, "width": <int>, \
"height": <int>}},
      "visible": <bool>,
      "scenario": "<scenario value>"
    }}
  ]
}}
</output_format>"""


def _vision_before_model_callback(callback_context, llm_request):
    """Inject screenshot as inline image into the LLM request."""
    state = callback_context.state
    raw = state["pa_screenshot"]
    # State stores base64 string; if somehow still bytes, encode it
    if isinstance(raw, bytes):
        screenshot_b64 = base64.b64encode(raw).decode("utf-8")
    else:
        screenshot_b64 = raw

    llm_request.contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=screenshot_b64,
                    )
                ),
                types.Part(text=VISION_PROMPT),
            ],
        )
    ]
    return None


vision_analyzer_agent = Agent(
    name="vision_analyzer",
    description="Extracts UI elements from screenshot via vision.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=VISION_PROMPT,
    output_key="pa_vision_raw",
    before_model_callback=_vision_before_model_callback,
)


# -------------------------------------------------------------------
# DOM Analyzer — LlmAgent (text-only, fully traced)
# -------------------------------------------------------------------

DOM_INSTRUCTION = f"""<role>You are a DOM analysis specialist extracting interactive and assertable elements from HTML source code.</role>

<task>Parse the HTML source and extract EVERY element that is either interactive or carries meaningful text content. Achieve exhaustive coverage of the DOM.</task>

<categories>
1. INTERACTIVE — buttons, inputs, links, selects, toggles, textareas, and any element with onclick, href, role="button", or similar interactive attributes.
2. ASSERTION — headings (h1-h6), labels, title elements, status text, error containers, aria-live regions, and any element displaying meaningful information.
</categories>

<element_fields>
For each element, return:
- type: one of {VALID_ACTION_TYPES}
- label: a human-readable description of the element's purpose
- selector: a precise CSS selector that uniquely identifies this element in the DOM
- text: the visible text content of the element
- role: the HTML tag name or ARIA role
- scenario: one of {VALID_SCENARIOS}
</element_fields>

<scenario_logic>
Apply the same reasoning as visual analysis:
- Assertion elements → always "neutral"
- Input fields → generate both "positive" and "negative" entries
- Buttons/links → typically "positive" unless explicitly error-related
</scenario_logic>

<output_format>
Return ONLY valid JSON matching this structure:
{{
  "elements": [
    {{
      "type": "<element type>",
      "label": "<human-readable description>",
      "selector": "<unique CSS selector>",
      "text": "<visible text>",
      "role": "<HTML tag or ARIA role>",
      "scenario": "<scenario value>"
    }}
  ]
}}
</output_format>"""


def _dom_before_model_callback(callback_context, llm_request):
    """Inject page source as user message."""
    state = callback_context.state
    page_source = state.get("pa_page_source", "")

    context_msg = (
        f"HTML SOURCE:\n```html\n{page_source[:15000]}\n```"
    )

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


dom_analyzer_agent = Agent(
    name="dom_analyzer",
    description="Extracts elements from HTML DOM.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=DOM_INSTRUCTION,
    output_key="pa_dom_raw",
    before_model_callback=_dom_before_model_callback,
)


# -------------------------------------------------------------------
# Merger — LlmAgent (text-only, fully traced)
# -------------------------------------------------------------------

MERGE_INSTRUCTION = """<role>You are a UI analysis merger combining visual and DOM-based element extraction results.</role>

<task>Merge two independent analyses of the same screen into a single unified element list:
1. VISION analysis (from screenshot): contains bounds, visibility, and visual context.
2. DOM analysis (from HTML): contains CSS selectors and structural information.
</task>

<thinking_process>
For each element, reason through these steps:
1. MATCH — Find corresponding elements between the two lists by comparing label, text, type, and position.
2. COMBINE — Merge matched pairs, taking the best attributes from each source.
3. RESOLVE CONFLICTS — When both sources provide the same field, prefer DOM for selectors and roles, prefer VISION for bounds and visibility.
4. HANDLE UNMATCHED — DOM elements with no vision match → set visible=false. Vision elements with no DOM match → set selector="".
</thinking_process>

<screen_description_task>
Produce a screen_description using this exact structured format:
  "[Page Title] | [Screen Type] | [Key Structural Elements] | [User Purpose]"

Rules:
- Page Title: the main heading or title visible on the screen (e.g. "Products", "Your Cart", "Checkout: Your Information")
- Screen Type: the category of screen (e.g. "Product listing grid", "Shopping cart", "Checkout form", "Single product detail")
- Key Structural Elements: the distinctive UI components that make this screen unique (e.g. "6 item cards, sort dropdown" vs "single product image, back button")
- User Purpose: what the user can accomplish here

Use BOTH the visual understanding AND the DOM structure to produce the most accurate description.
Two screens with similar purposes MUST still be distinguishable by their title and structural elements.
</screen_description_task>

<output_format>
Return ONLY valid JSON:
{
  "screen_description": "<Page Title | Screen Type | Key Elements | Purpose>",
  "elements": [
    {
      "type": "...",
      "label": "...",
      "selector": "...",
      "text": "...",
      "role": "...",
      "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
      "visible": true,
      "scenario": "..."
    }
  ]
}
</output_format>"""


def _merger_before_model_callback(callback_context, llm_request):
    """Inject vision + DOM results as user message."""
    state = callback_context.state
    vision_raw = state.get("pa_vision_raw", "{}")
    dom_raw = state.get("pa_dom_raw", "{}")

    vision_text = strip_code_fences(vision_raw)
    dom_text = strip_code_fences(dom_raw)

    context_msg = (
        f"VISION ANALYSIS:\n{vision_text}\n\n"
        f"DOM ANALYSIS:\n{dom_text}"
    )

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


def _parse_action_type(type_str: str) -> ActionType:
    try:
        return ActionType(type_str)
    except ValueError:
        return ActionType.CLICK


def _parse_scenario(scenario_str: str) -> ActionScenario:
    try:
        return ActionScenario(scenario_str)
    except ValueError:
        return ActionScenario.NEUTRAL


def _merger_after_model_callback(callback_context, llm_response):
    """Parse merged result and build ScreenDoc + ActionDocs."""
    state = callback_context.state

    raw = ""
    if llm_response.content and llm_response.content.parts:
        raw = llm_response.content.parts[0].text or ""
    raw = strip_code_fences(raw)

    try:
        merged = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        merged = {"screen_description": "", "elements": []}

    vision_raw = state.get("pa_vision_raw", "{}")
    try:
        vision_data = json.loads(strip_code_fences(vision_raw))
    except (json.JSONDecodeError, AttributeError):
        vision_data = {}

    screen_description = (
        merged.get("screen_description")
        or vision_data.get("screen_description", "")
    )

    screen_id = f"scr_{uuid.uuid4().hex[:8]}"

    actions = [
        ActionDoc(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            screen_id=screen_id,
            type=_parse_action_type(elem.get("type", "click")),
            scenario=_parse_scenario(
                elem.get("scenario", "neutral")
            ),
            element_info=ElementInfo(
                selector=elem.get("selector", ""),
                text=elem.get("text", ""),
                role=elem.get("role", ""),
                bounds=elem.get("bounds"),
                visible=elem.get("visible", True),
            ),
            status=ActionStatus.PENDING,
        )
        for elem in merged.get("elements", [])
    ]

    screen = ScreenDoc(
        screen_id=screen_id,
        screen_description=screen_description,
        status=ScreenStatus.ANALYZED,
        resolution=vision_data.get(
            "resolution", {"width": 1280, "height": 720}
        ),
        actions_summary={
            "total": len(actions),
            "pending": len(actions),
            "completed": 0, "failed": 0, "skipped": 0,
        },
    )

    state["pa_screen"] = screen.model_dump(mode="json")
    state["pa_description"] = screen_description
    state["pa_actions"] = [a.model_dump(mode="json") for a in actions]
    return None


merger_agent = Agent(
    name="merger",
    description="Merges vision and DOM analysis into unified list.",
    model=config.GEMINI_LITE_MODEL,
    instruction=MERGE_INSTRUCTION,
    output_key="pa_merge_raw",
    before_model_callback=_merger_before_model_callback,
    after_model_callback=_merger_after_model_callback,
)


# -------------------------------------------------------------------
# Page Analyzer Pipeline
# -------------------------------------------------------------------

page_analyzer_agent = SequentialAgent(
    name="page_analyzer",
    description="Dual vision + DOM analysis to extract UI elements.",
    sub_agents=[
        ParallelAgent(
            name="parallel_extraction",
            description="Runs vision and DOM extraction concurrently.",
            sub_agents=[
                vision_analyzer_agent,
                dom_analyzer_agent,
            ],
        ),
        merger_agent,
    ],
)
