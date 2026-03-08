from src.config import config
from src.utils import gemini_client, strip_code_fences
from src.models.action import ActionDoc, ActionType, ActionStatus
from src.sandbox.session_manager import SessionManager


def _build_code_gen_prompt(action: ActionDoc) -> str:
    """Build prompt for Gemini to generate executable Playwright code."""
    elem = action.element_info
    action_type = action.type.value

    selector_info = f"CSS selector: {elem.selector}" if elem.selector else "No CSS selector available"
    fill_info = f"\nFill value: {action.fill_value}" if action.fill_value else ""

    return f"""<role>You are a Playwright code generator producing executable Python code for browser automation.</role>

<task>Generate Playwright Python code to perform one specific action on a web page element.</task>

<target_element>
Action type: {action_type}
Element description: {action.element_info.text or "N/A"}
{selector_info}
HTML role: {elem.role or "N/A"}{fill_info}
</target_element>

<constraints>
- Write ONLY the body of an async function that receives `page` (a Playwright Page object).
- Do NOT include function definitions, imports, or try/except blocks — only the action lines.
- Use `await` for all Playwright calls.
- Use the CSS selector to locate the element.
- For "fill" actions: use EXACTLY the fill value provided above. Do NOT generate or modify the value.
- End with `await page.wait_for_timeout(500)`.
- Interact with ONLY the single element described above. Do NOT touch any other element.
</constraints>

<output_format>
Return ONLY raw Python code lines. No markdown fences, no explanations, no comments.
</output_format>"""


def _build_retry_prompt(action: ActionDoc, error: str, attempt: int) -> str:
    """Build retry prompt with error context."""
    elem = action.element_info
    selector_info = f"CSS selector: {elem.selector}" if elem.selector else "No CSS selector"
    bounds_info = ""
    if elem.bounds:
        b = elem.bounds
        bounds_info = f"\nBounding box: x={b['x']}, y={b['y']}, width={b['width']}, height={b['height']}"

    return f"""<role>You are a Playwright code generator recovering from a failed action attempt.</role>

<task>The previous code failed. Generate a NEW approach using a different strategy to interact with the same element.</task>

<target_element>
Action type: {action.type.value}
Element description: {elem.text or "N/A"}
{selector_info}{bounds_info}
</target_element>

<failure_context>
Previous error: {error}
Attempt: {attempt} of {config.ACTION_RETRY_LIMIT}
</failure_context>

<thinking_process>
Analyze the error and select the most appropriate recovery strategy:
1. If the selector was not found → try using coordinates from the bounding box (click at center point).
2. If coordinates also failed → try an alternative selector strategy (match by text content, by role, or by aria-label).
3. If the element was not found at all → add an explicit wait: `await page.wait_for_selector("selector", timeout=5000)`.
4. If a fill operation failed → try clicking the element first to focus it, then use `page.keyboard.type()` instead.
</thinking_process>

<constraints>
- Interact with ONLY the single element described above. Do NOT touch any other element.
- Use a DIFFERENT strategy than the one that failed.
</constraints>

<output_format>
Return ONLY raw Python code lines. No markdown fences, no explanations.
</output_format>"""



async def navigate_action(
    session: SessionManager,
    action: ActionDoc,
) -> dict:
    """Generate and execute Playwright code for a single action.

    No screenshot needed — ActionDoc has selector + bounds for element location.

    Returns:
        {
            "success": bool,
            "action": ActionDoc (updated status),
            "screenshot_after": bytes,
            "generated_code": str,
            "attempts": int,
        }
    """
    if action.type == ActionType.ASSERTION:
        screenshot = await session.screenshot()
        return {
            "success": True,
            "action": action.model_copy(update={"status": ActionStatus.COMPLETED}),
            "screenshot_after": screenshot,
            "generated_code": "# assertion — no execution needed",
            "attempts": 0,
        }

    prompt = _build_code_gen_prompt(action)

    response = gemini_client.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
    )

    generated_code = strip_code_fences(response.text)
    last_error = None

    for attempt in range(1, config.ACTION_RETRY_LIMIT + 1):
        result = await session.execute_code(generated_code)

        if result["success"]:
            updated = action.model_copy(update={"status": ActionStatus.COMPLETED})
            return {
                "success": True,
                "action": updated,
                "screenshot_after": result["screenshot"],
                "generated_code": generated_code,
                "attempts": attempt,
            }

        last_error = result.get("error", "Unknown error")
        if attempt >= config.ACTION_RETRY_LIMIT:
            break

        retry_prompt = _build_retry_prompt(action, last_error, attempt + 1)

        retry_response = gemini_client.models.generate_content(
            model=config.GEMINI_FLASH_MODEL,
            contents=[{
                "role": "user",
                "parts": [{"text": retry_prompt}],
            }],
        )
        generated_code = strip_code_fences(retry_response.text)

    # All retries exhausted
    updated = action.model_copy(update={
        "status": ActionStatus.FAILED,
        "error": last_error,
        "retry_count": config.ACTION_RETRY_LIMIT,
    })
    return {
        "success": False,
        "action": updated,
        "screenshot_after": result["screenshot"],
        "generated_code": generated_code,
        "attempts": config.ACTION_RETRY_LIMIT,
    }
