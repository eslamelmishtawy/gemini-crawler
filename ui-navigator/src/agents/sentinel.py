"""Sentinel Agent — state change detection + screen matching.

Split into two agents:
- sentinel_compare: Agent (multimodal via before_model_callback)
- sentinel_match: Agent — text-only screen matching
"""

import base64
import json

from google.adk.agents import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.config import config
from src.utils import strip_code_fences


class StateVerdict:
    SAME_SCREEN = "same_screen"
    NEW_SCREEN = "new_screen"
    MODAL = "modal"
    ERROR = "error"


# -------------------------------------------------------------------
# Sentinel Compare — Agent (multimodal via before_model_callback)
# -------------------------------------------------------------------

COMPARE_PROMPT = """<role>You are a UI state change detector comparing two screenshots taken before and after an action was executed.</role>

<task>Analyze the visual differences between Screenshot 1 (BEFORE) and Screenshot 2 (AFTER) to determine what type of state change occurred.</task>

<thinking_process>
Reason through these steps:
1. LAYOUT COMPARISON — Has the overall page structure changed? Different navigation, different content sections, different page?
2. CONTENT COMPARISON — Are the headings, text blocks, and primary content the same or different?
3. OVERLAY DETECTION — Is there a dialog, popup, sidebar, or toast appearing on top of the existing page?
4. ERROR DETECTION — Are there error indicators: red borders, error messages, alert banners, HTTP error pages?
5. MINOR CHANGE CHECK — Are the differences limited to field values, dropdown states, button highlights, or other in-place updates?
</thinking_process>

<verdicts>
Based on your analysis, classify the change as exactly one of:
- "same_screen" — The page is fundamentally the same. Minor changes like filled fields, opened dropdowns, color changes, or text appearing in inputs do NOT constitute a new screen. Same layout, same navigation, same structure → same_screen.
- "new_screen" — The page has completely changed. Different layout, different content structure, different navigation. This is a full page navigation.
- "modal" — An overlay, dialog, sidebar, popup, or toast appeared ON TOP of the same page. The underlying page is still visible or partially visible behind it.
- "error" — An error state appeared: error toast, validation messages, error page, or crash screen. This can manifest as a modal OR as inline errors on the same screen.
</verdicts>

<output_format>
Return ONLY valid JSON: {"verdict": "<one of the four values>", "reason": "<brief explanation of what changed>"}
</output_format>"""


def _compare_before_model_callback(callback_context, llm_request):
    """Inject before/after screenshots as inline images."""
    state = callback_context.state
    raw_before = state.get("sentinel_screenshot_before", "")
    raw_after = state.get("sentinel_screenshot_after", "")
    # Ensure raw bytes for Blob — state stores base64 strings
    if isinstance(raw_before, bytes):
        before_bytes = raw_before
    else:
        before_bytes = base64.b64decode(raw_before) if raw_before else b""
    if isinstance(raw_after, bytes):
        after_bytes = raw_after
    else:
        after_bytes = base64.b64decode(raw_after) if raw_after else b""

    llm_request.contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=COMPARE_PROMPT),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=before_bytes,
                    )
                ),
                types.Part(
                    text="Screenshot 1 (BEFORE) is above. "
                    "Screenshot 2 (AFTER) is below."
                ),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=after_bytes,
                    )
                ),
            ],
        )
    ]
    return None


def _compare_after_model_callback(callback_context, llm_response):
    """Parse verdict and store in state."""
    state = callback_context.state
    raw = ""
    if llm_response.content and llm_response.content.parts:
        raw = llm_response.content.parts[0].text or ""
    raw = strip_code_fences(raw)

    try:
        result = json.loads(raw)
        verdict = result.get("verdict", "same_screen")
        if verdict not in (
            StateVerdict.SAME_SCREEN,
            StateVerdict.NEW_SCREEN,
            StateVerdict.MODAL,
            StateVerdict.ERROR,
        ):
            verdict = StateVerdict.SAME_SCREEN
        state["sentinel_verdict"] = {
            "verdict": verdict,
            "reason": result.get("reason", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        state["sentinel_verdict"] = {
            "verdict": StateVerdict.SAME_SCREEN,
            "reason": "Failed to parse response",
        }
    return None


sentinel_compare_agent = Agent(
    name="sentinel_compare",
    description="Detects state changes via screenshot comparison.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=COMPARE_PROMPT,
    output_key="sentinel_compare_raw",
    before_model_callback=_compare_before_model_callback,
    after_model_callback=_compare_after_model_callback,
)


# -------------------------------------------------------------------
# Sentinel Match — LlmAgent (text-only, fully traced)
# -------------------------------------------------------------------

MATCH_INSTRUCTION = """<role>You are a screen identity matcher determining whether a newly discovered screen corresponds to any previously seen screen.</role>

<description_format>
Screen descriptions follow a structured format: "Page Title | Screen Type | Key Structural Elements | Purpose"
Use ALL four parts for matching — especially Page Title and Screen Type, which are the strongest identity signals.
</description_format>

<thinking_process>
Reason through these steps:
1. TITLE MATCH — Does the new screen share the same page title as any known screen? Different titles almost always mean different screens.
2. SCREEN TYPE MATCH — Is the screen type the same? A list/grid view is NOT the same as a detail/single-item view, even if they belong to the same domain.
3. STRUCTURAL MATCH — Do the key structural elements match? Compare the number and types of UI components. Screens with fundamentally different layouts (e.g., a grid of cards vs a single detailed view) are different screens.
4. DATA INDEPENDENCE — Two screens with the SAME title, type, and structure but showing DIFFERENT data instances ARE the same screen.
</thinking_process>

<critical_rule>
Screens that show DIFFERENT LEVELS OF DETAIL about the same entity are DIFFERENT screens.
A screen that displays a COLLECTION of items and a screen that displays a SINGLE item in detail are structurally different — never match them.
When in doubt, return no match. A false negative (treating a known screen as new) simply causes re-analysis. A false positive (matching two different screens) corrupts the navigation graph and is much more harmful.
</critical_rule>

<output_format>
If the new screen matches a known screen, return: {{"match": true, "screen_id": "<the matching screen_id>"}}
If no match, return: {{"match": false, "screen_id": null}}
</output_format>"""


def _match_before_model_callback(callback_context, llm_request):
    """Inject screen descriptions as user message."""
    state = callback_context.state
    new_desc = state.get("sentinel_new_desc", "")
    known_screens = state.get("sentinel_known_screens", {})

    if not known_screens:
        no_match = json.dumps({"match": False, "screen_id": None})
        state["sentinel_match"] = {"match": False, "screen_id": None}
        state["sentinel_match_raw"] = no_match
        return LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=no_match)]
            ),
        )

    screens_text = "\n".join(
        f'- {sid}: "{desc}"'
        for sid, desc in known_screens.items()
    )

    context_msg = (
        f"NEW SCREEN DESCRIPTION:\n{new_desc}\n\n"
        f"KNOWN SCREENS:\n{screens_text}"
    )

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


def _match_after_model_callback(callback_context, llm_response):
    """Parse and store match result."""
    state = callback_context.state
    raw = ""
    if llm_response.content and llm_response.content.parts:
        raw = llm_response.content.parts[0].text or ""
    raw = strip_code_fences(raw)

    try:
        result = json.loads(raw)
        state["sentinel_match"] = {
            "match": bool(result.get("match", False)),
            "screen_id": result.get("screen_id"),
        }
    except (json.JSONDecodeError, AttributeError):
        state["sentinel_match"] = {
            "match": False, "screen_id": None,
        }
    return None


sentinel_match_agent = Agent(
    name="sentinel_match",
    description="Matches new screens against known screens.",
    model=config.GEMINI_LITE_MODEL,
    instruction=MATCH_INSTRUCTION,
    output_key="sentinel_match_raw",
    before_model_callback=_match_before_model_callback,
    after_model_callback=_match_after_model_callback,
)
