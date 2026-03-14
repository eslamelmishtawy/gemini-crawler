"""Iteration Agent — LlmAgent that reasons about exploration state and
transfers to the appropriate sub-agent.

Replaces the old reasoner + manual if/elif branching with ADK's native
transfer_to_agent mechanism.  The LLM reads exploration state (via
before_model_callback), reasons strategically, then transfers to one of:

    - explore_current   → execute next action on current screen
    - navigate_to       → replay path to a different screen
    - stop_exploration  → escalate to exit the LoopAgent

For navigate_to, the LLM first calls the set_navigation_target tool
to record the target screen_id in session state.
"""

from google.adk.agents import Agent
from google.genai import types

from src.config import config
from src.utils import (
    build_screens_summary,
    build_nav_graph,
    build_history,
    count_consecutive_failures,
)
from src.tools.firestore_tools import (
    get_all_screens,
    get_nav_edges,
    get_actions_by_screen,
    update_screen,
)


# -------------------------------------------------------------------
# Tool: set navigation target before transferring to navigate_to
# -------------------------------------------------------------------

def set_navigation_target(target_screen_id: str, tool_context) -> str:
    """Set the target screen for navigation.

    Call this BEFORE transferring to the navigate_to agent.

    Args:
        target_screen_id: The screen_id to navigate to. Must be a
            valid screen_id from the application map.

    Returns:
        Confirmation message.
    """
    tool_context.state["navigate_target_screen_id"] = target_screen_id
    return (
        f"Navigation target set to {target_screen_id}. "
        "Now transfer to navigate_to."
    )


# -------------------------------------------------------------------
# Prompt — same strategic reasoning, but decisions are agent transfers
# -------------------------------------------------------------------

ITERATION_INSTRUCTION = """\
<role>
You are the strategic decision-maker for an autonomous UI exploration \
agent. You see the full state of the exploration and decide what to \
do next at the SCREEN and FLOW level.
</role>

<context>
You do NOT pick individual actions or interact with UI elements — \
the explore_current agent handles that. You think strategically: \
which areas of the application remain unexplored, whether progress \
is being made, and when to move on or stop.
</context>

<decisions>
Based on your analysis, take EXACTLY ONE action:

1. **Continue exploring the current screen** — if it still has \
valuable unexplored actions, transfer to the "explore_current" agent.

2. **Navigate to a different screen** — if the current screen is \
exhausted or a more promising screen exists, FIRST call \
set_navigation_target(target_screen_id="<screen_id>") with a valid \
screen_id from the application map, THEN transfer to the \
"navigate_to" agent.

3. **Stop exploration** — if the application has been sufficiently \
mapped or exploration is irreversibly stuck, transfer to the \
"stop_exploration" agent.
</decisions>

<guidance_rules>
When transferring to explore_current, your reasoning serves as \
strategic guidance for the scout. Describe WHAT AREA to explore \
and WHY, not which specific button to click.
</guidance_rules>"""


# -------------------------------------------------------------------
# before_model_callback — inject exploration state as user message
# -------------------------------------------------------------------

def _iteration_before_model_callback(callback_context, llm_request):
    """Build exploration context from live Firestore + session state."""
    state = callback_context.state
    platform = state.get("platform", "web")
    current_screen_id = state.get("current_screen_id", "")
    current_description = state.get("current_description", "")
    known_screens = state.get("known_screens", {})
    history = state.get("history", [])

    run_id = state["run_id"]

    iteration_num = state.get("iteration_number", 0) + 1
    state["iteration_number"] = iteration_num

    all_screens = get_all_screens(run_id)
    nav_edges = get_nav_edges(run_id)
    screen_actions = get_actions_by_screen(run_id, current_screen_id)

    # Determine current screen status
    current_status = "analyzed"
    for s in all_screens:
        if s["screen_id"] == current_screen_id:
            current_status = s.get("status", "analyzed")
            break

    pending = [a for a in screen_actions if a.get("status") == "pending"]
    completed = [a for a in screen_actions if a.get("status") == "completed"]
    failed = [a for a in screen_actions if a.get("status") == "failed"]
    skipped = [a for a in screen_actions if a.get("status") == "skipped"]
    explored_count = sum(
        1 for s in all_screens if s.get("status") == "fully_explored"
    )
    consecutive_failures = count_consecutive_failures(history)

    # Mark screen fully explored if no pending actions
    if not pending:
        update_screen(run_id, current_screen_id, {"status": "fully_explored"})

    # Build situation string
    if not pending:
        situation = (
            "Current screen has no pending actions. "
            "Navigate to another screen or stop."
        )
    elif consecutive_failures >= 3:
        situation = (
            f"{consecutive_failures} consecutive actions failed. "
            "May need to change strategy or abandon screen."
        )
    else:
        situation = "Ready to execute next action on current screen."

    # Store scout guidance from LLM reasoning for explore_current
    # (will be populated by the LLM's response naturally)

    context_msg = f"""\
<platform>{platform}</platform>

<application_map>
{build_screens_summary(all_screens)}
</application_map>

<navigation_graph>
{build_nav_graph(nav_edges)}
</navigation_graph>

<current_position>
Screen: {current_screen_id}
Description: "{current_description}"
Status: {current_status}
Pending actions: {len(pending)} | Completed: {len(completed)} \
| Failed: {len(failed)} | Skipped: {len(skipped)}
</current_position>

<exploration_history count="{min(len(history), 15)}">
{build_history(history)}
</exploration_history>

<stats>
Iteration: {iteration_num} of {config.MAX_ITERATIONS}
Screens discovered: {len(all_screens)} | Fully explored: {explored_count}
Actions executed: {state.get('total_actions_executed', 0)}
Consecutive failures: {consecutive_failures}
</stats>

<situation>{situation}</situation>

Reason through the exploration state, then take your action."""

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


# -------------------------------------------------------------------
# The iteration agent — note: sub_agents are wired in agent_definitions
# -------------------------------------------------------------------

iteration_agent = Agent(
    name="iteration",
    description=(
        "Strategic decision-maker that reasons about exploration "
        "state and transfers to the appropriate action agent."
    ),
    model=config.GEMINI_FLASH_MODEL,
    instruction=ITERATION_INSTRUCTION,
    tools=[set_navigation_target],
    before_model_callback=_iteration_before_model_callback,
)
