import base64
import json

from src.config import config
from src.utils import gemini_client, strip_code_fences


class StateVerdict:
    SAME_SCREEN = "same_screen"
    NEW_SCREEN = "new_screen"
    MODAL = "modal"
    ERROR = "error"


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


MATCH_PROMPT = """<role>You are a screen identity matcher determining whether a newly discovered screen corresponds to any previously seen screen.</role>

<task>Compare the NEW screen description against the list of KNOWN screens and determine if it represents the same page or view as any of them.</task>

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

<input>
NEW SCREEN DESCRIPTION:
{new_description}

KNOWN SCREENS:
{known_screens}
</input>

<output_format>
If the new screen matches a known screen, return: {{"match": true, "screen_id": "<the matching screen_id>"}}
If no match, return: {{"match": false, "screen_id": null}}
</output_format>"""



async def compare_states(
    screenshot_before: bytes,
    screenshot_after: bytes,
) -> dict:
    """Compare before/after screenshots to determine what state change occurred.

    Uses Gemini FLASH with vision — this is a complex visual reasoning task.

    Returns:
        {"verdict": "same_screen|new_screen|modal|error", "reason": "..."}
    """
    before_b64 = base64.b64encode(screenshot_before).decode()
    after_b64 = base64.b64encode(screenshot_after).decode()

    response = gemini_client.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=[{
            "role": "user",
            "parts": [
                {"text": COMPARE_PROMPT},
                {"inline_data": {"mime_type": "image/png", "data": before_b64}},
                {"text": "Screenshot 1 (BEFORE) is above. Screenshot 2 (AFTER) is below."},
                {"inline_data": {"mime_type": "image/png", "data": after_b64}},
            ],
        }],
    )

    raw = strip_code_fences(response.text)
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
        return {"verdict": verdict, "reason": result.get("reason", "")}
    except (json.JSONDecodeError, AttributeError):
        return {"verdict": StateVerdict.SAME_SCREEN, "reason": "Failed to parse response"}


async def match_screen(
    new_description: str,
    known_screens: dict[str, str],
) -> dict:
    """Check if a new screen description matches any known screen.

    Uses Gemini LITE — this is a simple text comparison task.

    Args:
        new_description: screen_description from Page Analyzer for the new screen.
        known_screens: dict of {screen_id: screen_description} for all known screens.

    Returns:
        {"match": True, "screen_id": "scr_001"} or {"match": False, "screen_id": None}
    """
    if not known_screens:
        return {"match": False, "screen_id": None}

    screens_text = "\n".join(
        f"- {sid}: \"{desc}\"" for sid, desc in known_screens.items()
    )

    prompt = MATCH_PROMPT.format(
        new_description=new_description,
        known_screens=screens_text,
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
        result = json.loads(raw)
        return {
            "match": bool(result.get("match", False)),
            "screen_id": result.get("screen_id"),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"match": False, "screen_id": None}
