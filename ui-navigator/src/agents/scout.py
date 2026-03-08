import json

from src.config import config
from src.utils import gemini_client, strip_code_fences

SCOUT_PROMPT = """<role>You are the action selection tactician for an autonomous UI exploration agent. You pick the single best pending action to execute next.</role>

<context>
<current_screen>{screen_description}</current_screen>
<known_screens>{known_screens}</known_screens>
</context>

<completed_actions>
{completed_actions}
</completed_actions>

<failed_actions>
{failed_actions}
</failed_actions>

<pending_actions>
{pending_actions}
</pending_actions>

<orchestrator_guidance>
{scout_guidance}
</orchestrator_guidance>

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


async def recommend_next_action(
    screen_actions: list[dict],
    screen_description: str,
    known_screens: dict[str, str],
    scout_guidance: str = "",
) -> dict:
    """Pick the best next action from all screen actions.

    Receives ALL actions (pending, completed, failed) so it can reason
    about what was already tried and make informed decisions.

    Args:
        screen_actions: ALL action dicts for this screen (any status).
        screen_description: description of the current screen.
        known_screens: {screen_id: description} of all discovered screens.
        scout_guidance: strategic hint from the Orchestrator.

    Returns:
        {"action_id": str, "reasoning": str} or
        {"action_id": None, "reasoning": "..."} if no actions available.
    """
    pending_actions = [a for a in screen_actions if a.get("status") == "pending"]
    completed_actions = [a for a in screen_actions if a.get("status") == "completed"]
    failed_actions = [a for a in screen_actions if a.get("status") == "failed"]

    if not pending_actions:
        return {"action_id": None, "reasoning": "No pending actions available"}

    if len(pending_actions) == 1 and not completed_actions and not failed_actions:
        return {
            "action_id": pending_actions[0]["action_id"],
            "reasoning": "Only one pending action available",
        }

    known_text = "\n".join(
        f"- {sid}: \"{desc}\"" for sid, desc in known_screens.items()
    ) if known_screens else "(none — this is the first screen)"

    prompt = SCOUT_PROMPT.format(
        screen_description=screen_description,
        known_screens=known_text,
        completed_actions=_format_actions(completed_actions, include_status=True),
        failed_actions=_format_actions(failed_actions, include_status=True),
        pending_actions=_format_actions(pending_actions),
        scout_guidance=scout_guidance or "(no specific guidance — use your best judgment)",
    )

    response = gemini_client.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
    )

    raw = strip_code_fences(response.text)
    try:
        result = json.loads(raw)
        chosen_id = result.get("action_id")
        # Validate the chosen action_id exists in pending actions
        valid_ids = {a["action_id"] for a in pending_actions}
        if chosen_id not in valid_ids:
            return {
                "action_id": pending_actions[0]["action_id"],
                "reasoning": f"Scout returned invalid action_id '{chosen_id}', falling back to first action",
            }
        return {
            "action_id": chosen_id,
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "action_id": pending_actions[0]["action_id"],
            "reasoning": "Failed to parse Scout response, falling back to first action",
        }
