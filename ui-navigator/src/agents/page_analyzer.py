import asyncio
import base64
import json
import uuid

from src.config import config
from src.utils import gemini_client, strip_code_fences
from src.models.screen import ScreenDoc, ScreenStatus
from src.models.action import (
    ActionDoc, ActionType, ActionStatus, ActionScenario, ElementInfo,
)

# Derived from enums — keeps prompts in sync with models
VALID_ACTION_TYPES = [t.value for t in ActionType]
VALID_SCENARIOS = [s.value for s in ActionScenario]

VISION_PROMPT = f"""<role>You are a UI element extraction specialist analyzing a screenshot of a web or mobile screen.</role>

<task>Extract EVERY visible element from this screenshot. Achieve exhaustive coverage — enumerate all elements without skipping any, even if they appear similar to each other.</task>

<categories>
Classify each element into one of two categories:
1. INTERACTIVE — buttons, inputs, links, toggles, dropdowns, tabs, navigation items, or any element a user can act on.
2. ASSERTION — page titles, headings, labels, status text, error messages, prices, counts, or any element that displays information.
</categories>

<element_fields>
For each element, return:
- type: one of {VALID_ACTION_TYPES}
- label: a human-readable description of the element's purpose
- text: the actual text content displayed on screen
- bounds: {{x, y, width, height}} in pixels relative to the screenshot
- visible: true if clearly visible and interactable, false if obscured or disabled
- scenario: one of {VALID_SCENARIOS}
</element_fields>

<scenario_logic>
Determine the scenario by reasoning about how this element would be used in testing:
- "positive" — interaction that follows the intended happy path (valid data, expected usage)
- "negative" — interaction designed to test error handling (empty submission, invalid format, boundary values)
- "neutral" — no pass/fail implication (reading text, checking layout, visual verification)

Apply these rules:
- Assertion elements are always "neutral" — they are read-only observations.
- Input fields should generate BOTH a "positive" and a "negative" entry as separate elements, since the same field can be tested both ways.
- Buttons and links are typically "positive" unless their purpose is explicitly error-related.
</scenario_logic>

<additional_fields>
Also return:
- screen_description: a structured description using this exact format:
  "[Page Title] | [Screen Type] | [Key Structural Elements] | [User Purpose]"
  Example: "Products | Product listing grid | 6 item cards, sort dropdown, cart icon, hamburger menu | Browse and add products to cart"
  Example: "Sauce Labs Backpack | Single product detail | product image, price, description, back button, add-to-cart | View product details and add to cart"
  Example: "Login | Authentication form | username input, password input, login button, credentials list | Sign in to access the store"
  The page title and structural elements make each screen uniquely identifiable even when purposes overlap.
- resolution: {{width, height}} of the screenshot in pixels
</additional_fields>

<output_format>
Return ONLY valid JSON matching this structure:
{{
  "screen_description": "<Page Title | Screen Type | Key Elements | Purpose>",
  "resolution": {{"width": <int>, "height": <int>}},
  "elements": [
    {{
      "type": "<element type>",
      "label": "<human-readable description>",
      "text": "<displayed text>",
      "bounds": {{"x": <int>, "y": <int>, "width": <int>, "height": <int>}},
      "visible": <bool>,
      "scenario": "<scenario value>"
    }}
  ]
}}
</output_format>"""

DOM_PROMPT = f"""<role>You are a DOM analysis specialist extracting interactive and assertable elements from HTML source code.</role>

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

MERGE_PROMPT = """<role>You are a UI analysis merger combining visual and DOM-based element extraction results.</role>

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


async def analyze_screen(screenshot_bytes: bytes, page_source: str) -> dict:
    """Dual analysis: Vision (coordinates) + DOM (selectors) → merged elements."""
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    # Vision + DOM run in parallel — independent analyses
    vision_response, dom_response = await asyncio.gather(
        gemini_client.aio.models.generate_content(
            model=config.GEMINI_FLASH_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}},
                    {"text": VISION_PROMPT},
                ],
            }],
        ),
        gemini_client.aio.models.generate_content(
            model=config.GEMINI_FLASH_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {"text": f"HTML SOURCE:\n```html\n{page_source[:15000]}\n```\n\n{DOM_PROMPT}"},
                ],
            }],
        ),
    )

    vision_text = strip_code_fences(vision_response.text)
    dom_text = strip_code_fences(dom_response.text)

    merge_response = await gemini_client.aio.models.generate_content(
        model=config.GEMINI_LITE_MODEL,
        contents=[{
            "role": "user",
            "parts": [{
                "text": (
                    f"VISION ANALYSIS:\n{vision_text}\n\n"
                    f"DOM ANALYSIS:\n{dom_text}\n\n"
                    f"{MERGE_PROMPT}"
                ),
            }],
        }],
    )

    merged = json.loads(strip_code_fences(merge_response.text))
    vision_data = json.loads(vision_text)

    # screen_description from merged understanding (vision + DOM combined)
    # Falls back to vision-only description if merge didn't produce one
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
            scenario=_parse_scenario(elem.get("scenario", "neutral")),
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
        resolution=vision_data.get("resolution", {"width": 1280, "height": 720}),
        actions_summary={
            "total": len(actions),
            "pending": len(actions),
            "completed": 0, "failed": 0, "skipped": 0,
        },
    )

    return {
        "screen": screen,
        "screen_description": screen_description,
        "actions": actions,
    }
