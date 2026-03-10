"""Scout Agent — picks the best next action from pending actions.

Converted to ADK LlmAgent (Agent) so every LLM call is automatically
traced and visible in the ADK Dev UI.

The orchestrator sets session state keys before invoking:
    - scout_actions: list[dict]
    - scout_description: str
    - scout_known_screens: dict
    - scout_guidance: str

Output stored via output_key="scout_result_raw".
"""

import json

from google.adk.agents import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.config import config


def _format_actions(actions: list[dict], include_status: bool = False) -> str:
    lines = []
    for a in actions:
        elem = a.get("element_info", {})
        line = (
            f"- action_id: {a['action_id']} | "
            f"type: {a['type']} | "
            f"text: \"{elem.get('text', 'N/A')}\" | "
            f"role: {elem.get('role', 'N/A')} | "
            f"selector: {elem.get('selector', 'N/A')}"
        )
        if include_status:
            line += f" | status: {a.get('status', 'unknown')}"
        if a.get("fill_value"):
            line += f" | fill_value: \"{a['fill_value']}\""
        if a.get("scenario"):
            line += f" | scenario: {a['scenario']}"
        if a.get("actual_outcome"):
            outcome = a["actual_outcome"]
            if isinstance(outcome, dict) and outcome.get("result"):
                line += f" | outcome: {outcome['result']}"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"


SCOUT_INSTRUCTION = """<role>You are the action selection tactician for an autonomous UI exploration agent. You pick the single best pending action to execute next.</role>

<thinking_process>
Reason through these steps before choosing:
1. HISTORY REVIEW — What actions were already tried? What were their outcomes? Identify patterns in successes and failures.
2. FORM DETECTION — If this screen contains a form pattern (input fields + submit button), ensure all inputs are filled before selecting the submit action.
3. FAILURE AVOIDANCE — Do any pending actions resemble actions that already failed? Avoid selecting actions likely to fail for the same reason.
4. DISCOVERY POTENTIAL — Which pending actions are most likely to lead to undiscovered screens? Navigation links and buttons that suggest page transitions are high value.
5. GUIDANCE ALIGNMENT — Does the orchestrator's guidance suggest a particular direction? Prioritize actions that align with the strategic goal.
6. SELECTION — Pick the single best action_id from the PENDING list only.
</thinking_process>

<constraints>
- You MUST pick an action_id from the PENDING list. Do NOT select completed or failed actions.
- Pick exactly ONE action.
</constraints>

<output_format>
Return ONLY valid JSON:
{{"action_id": "<chosen action_id from pending>", "reasoning": "<why this action, given what was already tried>"}}
</output_format>"""


def _make_skip_response(text: str) -> LlmResponse:
    """Build an LlmResponse to short-circuit the LLM call."""
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        ),
    )


def _scout_before_model_callback(callback_context, llm_request):
    """Inject the formatted scout context as the user message."""
    state = callback_context.state

    screen_actions = state.get("scout_actions", [])
    screen_description = state.get("scout_description", "")
    known_screens = state.get("scout_known_screens", {})
    scout_guidance = state.get("scout_guidance", "")

    pending = [a for a in screen_actions if a.get("status") == "pending"]
    completed = [a for a in screen_actions if a.get("status") == "completed"]
    failed = [a for a in screen_actions if a.get("status") == "failed"]

    # Short-circuit: only one pending and no history
    if len(pending) == 1 and not completed and not failed:
        result = json.dumps({
            "action_id": pending[0]["action_id"],
            "reasoning": "Only one pending action available",
        })
        state["scout_result_raw"] = result
        return _make_skip_response(result)

    if not pending:
        result = json.dumps({
            "action_id": None,
            "reasoning": "No pending actions available",
        })
        state["scout_result_raw"] = result
        return _make_skip_response(result)

    known_text = "\n".join(
        f'- {sid}: "{desc}"' for sid, desc in known_screens.items()
    ) if known_screens else "(none — this is the first screen)"

    context_msg = (
        f"<context>\n"
        f"<current_screen>{screen_description}</current_screen>\n"
        f"<known_screens>{known_text}</known_screens>\n"
        f"</context>\n\n"
        f"<completed_actions>\n"
        f"{_format_actions(completed, include_status=True)}\n"
        f"</completed_actions>\n\n"
        f"<failed_actions>\n"
        f"{_format_actions(failed, include_status=True)}\n"
        f"</failed_actions>\n\n"
        f"<pending_actions>\n"
        f"{_format_actions(pending)}\n"
        f"</pending_actions>\n\n"
        f"<orchestrator_guidance>\n"
        f"{scout_guidance or '(no specific guidance)'}\n"
        f"</orchestrator_guidance>"
    )

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None  # proceed with LLM call


scout_agent = Agent(
    name="scout",
    description="Picks the best next action from pending actions.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=SCOUT_INSTRUCTION,
    output_key="scout_result_raw",
    before_model_callback=_scout_before_model_callback,
)
