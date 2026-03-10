"""Reporter Agent — LLM-powered exploration summary and insights.

Called by the orchestrator after the exploration loop completes.
Reads all data from Firestore and session state, builds a structured
context, and asks the LLM for a detailed analysis report.
"""

from google.adk.agents import Agent
from google.genai import types

from src.config import config
from src.tools.firestore_tools import (
    get_all_screens,
    get_nav_edges,
    get_actions_by_screen,
)


REPORTER_INSTRUCTION = """\
<role>
You are an expert QA analyst reviewing the results of an autonomous \
UI exploration session. Produce a detailed, well-structured report.
</role>

<output_format>
Respond with a single JSON object (no markdown fences, no extra text). \
Use this exact schema:

{
  "executive_summary": "One paragraph: what was explored, how well \
the agent performed, and overall coverage quality.",

  "stats": {
    "screens_discovered": <int>,
    "screens_fully_explored": <int>,
    "screens_remaining": <int>,
    "actions_completed": <int>,
    "actions_failed": <int>,
    "actions_skipped": <int>,
    "actions_pending": <int>,
    "nav_edges": <int>,
    "iterations_used": <int>,
    "iterations_max": <int>,
    "time_elapsed_seconds": <number>,
    "coverage_percent": <number>
  },

  "navigation_graph": {
    "edges": [
      {
        "from_screen": "<screen_id>",
        "from_description": "<description>",
        "to_screen": "<screen_id>",
        "to_description": "<description>",
        "via_action": "<action_text>",
        "action_type": "<type>"
      }
    ],
    "orphan_screens": ["<screen_ids with no incoming edges>"],
    "dead_end_screens": ["<screen_ids with no outgoing edges>"]
  },

  "screens": [
    {
      "screen_id": "<id>",
      "description": "<description>",
      "status": "<fully_explored|analyzed>",
      "coverage_percent": <number>,
      "actions": {
        "completed": <int>,
        "failed": <int>,
        "skipped": <int>,
        "pending": <int>
      },
      "failed_actions": [
        {
          "action_id": "<id>",
          "type": "<type>",
          "text": "<element text>",
          "error": "<error message>"
        }
      ],
      "pending_actions": [
        {
          "action_id": "<id>",
          "type": "<type>",
          "text": "<element text>"
        }
      ]
    }
  ],

  "failure_analysis": {
    "error_patterns": [
      {
        "pattern": "<error type/pattern>",
        "count": <int>,
        "affected_actions": ["<action descriptions>"],
        "root_cause": "<analysis>"
      }
    ],
    "most_problematic_screens": ["<screen_ids>"]
  },

  "coverage_assessment": {
    "overall_percent": <number>,
    "well_covered_areas": ["<descriptions>"],
    "under_explored_areas": [
      {
        "screen_id": "<id>",
        "description": "<description>",
        "reason": "<why it needs more exploration>"
      }
    ]
  },

  "iteration_budget": {
    "exploration_complete": <boolean>,
    "iterations_used": <int>,
    "current_max": <int>,
    "suggested_max_iterations": <int or null if complete>,
    "pending_actionable_actions": <int>,
    "avg_iterations_per_action": <number>,
    "rationale": "<explanation of the estimate>"
  },

  "recommendations": [
    "<specific, actionable recommendation>"
  ]
}
</output_format>

<guidelines>
- Be specific — reference actual screen IDs, action texts, and errors
- Be honest about gaps and failures
- Provide actionable insights, not vague observations
- When calculating coverage, exclude assertion actions (they are \
informational only and auto-skipped)
- Ensure all numbers are accurate — compute them from the provided data
- The JSON must be valid and parseable
</guidelines>"""


def _reporter_before_model_callback(callback_context, llm_request):
    """Build full exploration data context for the reporter LLM."""
    state = callback_context.state

    all_screens = get_all_screens()
    nav_edges = get_nav_edges()
    known_screens = state.get("known_screens", {})
    history = state.get("history", [])

    # --- Build per-screen action details ---
    screen_details = []
    total_actions = {"completed": 0, "failed": 0, "skipped": 0, "pending": 0}

    for screen in all_screens:
        sid = screen.get("screen_id", "")
        actions = get_actions_by_screen(sid)
        by_status = {"completed": [], "failed": [], "skipped": [], "pending": []}

        for a in actions:
            status = a.get("status", "pending")
            entry = {
                "action_id": a.get("action_id", ""),
                "type": a.get("type", ""),
                "text": a.get("element_info", {}).get("text", ""),
                "status": status,
            }
            if status == "failed":
                entry["error"] = a.get("error", "")
                entry["outcome"] = a.get("outcome", {})
            if status == "pending":
                entry["type_detail"] = a.get("type", "")
            by_status.get(status, by_status["pending"]).append(entry)

        for s in by_status:
            total_actions[s] += len(by_status[s])

        # Coverage excluding assertions
        actionable = [a for a in actions if a.get("type") != "assertion"]
        actionable_completed = [
            a for a in actionable if a.get("status") == "completed"
        ]
        coverage = (
            f"{len(actionable_completed)}/{len(actionable)} "
            f"({100 * len(actionable_completed) / len(actionable):.0f}%)"
            if actionable else "N/A (no actionable actions)"
        )

        screen_details.append(
            f"### Screen: {sid}\n"
            f"  Description: {known_screens.get(sid, screen.get('description', 'N/A'))}\n"
            f"  Status: {screen.get('status', 'unknown')}\n"
            f"  URL: {screen.get('url_or_activity', 'N/A')}\n"
            f"  Actions Summary: {screen.get('actions_summary', {})}\n"
            f"  Actionable Coverage: {coverage}\n"
            f"  Completed: {_format_actions(by_status['completed'])}\n"
            f"  Failed: {_format_actions(by_status['failed'], include_error=True)}\n"
            f"  Skipped: {_format_actions(by_status['skipped'])}\n"
            f"  Pending: {_format_actions(by_status['pending'])}\n"
        )

    # --- Build edge list ---
    edge_lines = []
    for e in nav_edges:
        edge_lines.append(
            f"  {e.get('from_screen', '?')} → {e.get('to_screen', '?')} "
            f"(via \"{e.get('action_text', '')}\" [{e.get('action_type', '')}])"
        )

    # --- Build history summary ---
    history_lines = []
    for i, h in enumerate(history, 1):
        line = (
            f"  {i}. [{h.get('action_type', '')}] "
            f"\"{h.get('action_text', '')}\" → {h.get('verdict', '?')}"
        )
        if h.get("error"):
            line += f" (error: {h['error'][:80]})"
        if h.get("new_screen"):
            line += f" → discovered {h['new_screen']}"
        history_lines.append(line)

    # --- Assemble context ---
    fully_explored = sum(
        1 for s in all_screens if s.get("status") == "fully_explored"
    )
    iteration_num = state.get("iteration_number", "?")
    elapsed = state.get("exploration_result", {}).get("elapsed_seconds", "?")
    target_url = state.get("target_url", "?")

    context_msg = f"""\
<exploration_data>

<target>{target_url}</target>

<overview>
Screens discovered: {len(all_screens)}
Screens fully explored: {fully_explored}
Screens remaining: {len(all_screens) - fully_explored}
Total actions — completed: {total_actions['completed']}, \
failed: {total_actions['failed']}, \
skipped: {total_actions['skipped']}, \
pending: {total_actions['pending']}
Navigation edges: {len(nav_edges)}
Iterations used: {iteration_num} of {config.MAX_ITERATIONS}
Time elapsed: {elapsed}s
</overview>

<navigation_graph>
{chr(10).join(edge_lines) if edge_lines else "No edges recorded."}
</navigation_graph>

<screens>
{chr(10).join(screen_details)}
</screens>

<exploration_history>
{chr(10).join(history_lines) if history_lines else "No history recorded."}
</exploration_history>

</exploration_data>

Analyze the exploration data above and produce your detailed report."""

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


def _format_actions(actions, include_error=False):
    """Format a list of actions for display."""
    if not actions:
        return "none"
    lines = []
    for a in actions:
        line = f"{a.get('type', '?')}:\"{a.get('text', '')}\""
        if include_error and a.get("error"):
            line += f" — ERROR: {a['error'][:100]}"
        lines.append(line)
    return "; ".join(lines)


def _reporter_after_model_callback(callback_context, llm_response):
    """Parse the JSON report and store structured data in state."""
    import json
    from src.utils import strip_code_fences

    raw = ""
    if llm_response.content and llm_response.content.parts:
        raw = llm_response.content.parts[0].text or ""

    clean = strip_code_fences(raw.strip())
    try:
        report = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        report = {"raw": raw, "parse_error": True}

    callback_context.state["exploration_report_json"] = report


reporter_agent = Agent(
    name="reporter",
    description="Generates a detailed exploration summary report.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=REPORTER_INSTRUCTION,
    output_key="exploration_report",
    before_model_callback=_reporter_before_model_callback,
    after_model_callback=_reporter_after_model_callback,
)
