"""Navigator Agent — generates and executes Playwright code.

Architecture:
    navigator (LoopAgent, max_iterations=ACTION_RETRY_LIMIT)
    └── code_gen_and_execute (Agent + execute_playwright_code tool)

The orchestrator sets session state before invoking:
    - nav_session: SessionManager
    - nav_action: ActionDoc

After completion:
    - nav_result: dict
"""

from datetime import datetime

from google.adk.agents import Agent, LoopAgent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.config import config
from src.models.action import ActionDoc, ActionType, ActionStatus
from src.utils import strip_code_fences, bytes_to_state


# -------------------------------------------------------------------
# FunctionTool: execute_playwright_code
# -------------------------------------------------------------------

async def execute_playwright_code(
    code: str,
    tool_context,
) -> dict:
    """Execute generated Playwright Python code on the browser page.

    Args:
        code: The Playwright Python code to execute. Must be the body
            of an async function that receives `page`.

    Returns:
        dict with keys:
        - success: bool indicating if the code ran without errors
        - error: error message if failed, empty string if success
        - screenshot_available: whether a screenshot was captured
    """
    state = tool_context.state
    session = state["nav_session"]
    action_raw = state["nav_action"]
    action: ActionDoc = ActionDoc(**action_raw) if isinstance(action_raw, dict) else action_raw

    clean_code = strip_code_fences(code)
    result = await session.execute_code(clean_code)

    attempt = state.get("_nav_attempt", 0) + 1
    state["_nav_attempt"] = attempt

    if result["success"]:
        screenshot = await session.screenshot()
        updated = action.model_copy(update={
            "status": ActionStatus.COMPLETED,
            "executed_at": datetime.utcnow().isoformat(),
        })
        state["nav_result"] = {
            "success": True,
            "action": updated.model_dump(mode="json"),
            "screenshot_after": bytes_to_state(screenshot),
            "generated_code": clean_code,
            "attempts": attempt,
        }
        # Don't escalate here — the before_model_callback will detect
        # nav_result on the next loop iteration and escalate locally,
        # preventing the escalation from propagating up to the
        # exploration_loop LoopAgent.
        return {
            "success": True,
            "error": "",
            "screenshot_available": True,
        }

    error_msg = result.get("error", "Unknown error")
    state["_nav_last_error"] = error_msg

    # If this is the last attempt, store failure result
    if attempt >= config.ACTION_RETRY_LIMIT:
        updated = action.model_copy(update={
            "status": ActionStatus.FAILED,
            "error": error_msg,
            "retry_count": config.ACTION_RETRY_LIMIT,
            "executed_at": datetime.utcnow().isoformat(),
        })
        state["nav_result"] = {
            "success": False,
            "action": updated.model_dump(mode="json"),
            "screenshot_after": bytes_to_state(result.get("screenshot", b"")),
            "generated_code": clean_code,
            "attempts": attempt,
        }

    return {
        "success": False,
        "error": error_msg,
        "screenshot_available": False,
    }


# -------------------------------------------------------------------
# Code Gen + Execute Agent (runs inside LoopAgent)
# -------------------------------------------------------------------

NAVIGATOR_INSTRUCTION = """\
<role>You are a Playwright code generator and executor for browser \
automation.</role>

<task>Generate Playwright Python code to perform the requested UI \
action, then call the execute_playwright_code tool to run it.</task>

<constraints>
- Write ONLY the body of an async function that receives `page` \
(a Playwright Page object).
- Do NOT include function definitions, imports, or try/except blocks.
- Use `await` for all Playwright calls.
- Use the CSS selector to locate the element.
- For "fill" actions: use EXACTLY the fill value provided.
- End with `await page.wait_for_timeout(500)`.
- Interact with ONLY the single element described.
</constraints>

<retry_strategy>
If the tool returns an error, analyze it and try a DIFFERENT approach:
1. Selector not found → use coordinates from bounding box
2. Coordinates failed → try alternative selector (text, role, \
aria-label)
3. Element not found → add explicit wait: \
`await page.wait_for_selector("selector", timeout=5000)`
4. Fill failed → click first to focus, then use \
`page.keyboard.type()` instead
</retry_strategy>

<workflow>
1. Generate the Playwright code for the action
2. Call execute_playwright_code with the generated code
3. If it succeeds, you are done
4. If it fails, generate new code with a different strategy and \
call the tool again
</workflow>"""


def _nav_before_model_callback(callback_context, llm_request):
    """Inject action details and any previous error context."""
    state = callback_context.state
    action_raw = state.get("nav_action")

    if not action_raw:
        return None

    # Navigation already completed — don't wipe history, just escalate
    nav_result = state.get("nav_result")
    if nav_result and (nav_result.get("success") or
                       state.get("_nav_attempt", 0) >= config.ACTION_RETRY_LIMIT):
        callback_context.actions.escalate = True
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Navigation complete.")],
            ),
        )

    # Already completed (success or max retries reached) — stop the loop
    nav_result = state.get("nav_result")
    if nav_result is not None:
        callback_context.actions.escalate = True
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Navigation already completed.")],
            ),
        )

    action = ActionDoc(**action_raw) if isinstance(action_raw, dict) else action_raw

    # Handle assertion actions — skip LLM, set result directly
    if action.type == ActionType.ASSERTION:
        state["nav_result"] = {
            "success": True,
            "action": action.model_copy(
                update={"status": ActionStatus.COMPLETED}
            ).model_dump(mode="json"),
            "screenshot_after": "",
            "generated_code": "# assertion — no execution",
            "attempts": 0,
        }
        # Skip LLM, escalate to exit loop
        callback_context.actions.escalate = True
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(
                    text="Assertion action — no execution needed."
                )],
            ),
        )

    elem = action.element_info
    selector_info = (
        f"CSS selector: {elem.selector}"
        if elem.selector else "No CSS selector available"
    )
    fill_info = (
        f"\nFill value: {action.fill_value}"
        if action.fill_value else ""
    )

    last_error = state.get("_nav_last_error")
    attempt = state.get("_nav_attempt", 0) + 1

    if last_error:
        bounds_info = ""
        if elem.bounds:
            b = elem.bounds
            bounds_info = (
                f"\nBounding box: x={b['x']}, y={b['y']}, "
                f"width={b['width']}, height={b['height']}"
            )

        context_msg = (
            f"Previous code FAILED. Use a DIFFERENT strategy.\n\n"
            f"Action type: {action.type.value}\n"
            f"Element: {elem.text or 'N/A'}\n"
            f"{selector_info}{bounds_info}{fill_info}\n\n"
            f"Previous error: {last_error}\n"
            f"Attempt: {attempt} of {config.ACTION_RETRY_LIMIT}"
        )
    else:
        context_msg = (
            f"Action type: {action.type.value}\n"
            f"Element: {elem.text or 'N/A'}\n"
            f"{selector_info}\n"
            f"HTML role: {elem.role or 'N/A'}{fill_info}"
        )

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


code_gen_and_execute_agent = Agent(
    name="code_gen_and_execute",
    description="Generates and executes Playwright code for a UI action.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=NAVIGATOR_INSTRUCTION,
    tools=[execute_playwright_code],
    before_model_callback=_nav_before_model_callback,
)


# -------------------------------------------------------------------
# Navigator — LoopAgent (hard retry limit)
# -------------------------------------------------------------------

navigator_agent = LoopAgent(
    name="navigator",
    description="Generates and executes Playwright code with retries.",
    max_iterations=config.ACTION_RETRY_LIMIT,
    sub_agents=[code_gen_and_execute_agent],
)
