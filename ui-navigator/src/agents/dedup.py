"""Dedup Agent — identifies repeated UI patterns and skips duplicates.

Converted to ADK LlmAgent (Agent) so the LLM call is auto-traced.

The orchestrator sets session state:
    - dedup_actions: list[ActionDoc]

Output stored via output_key="dedup_result_raw".
"""

import json

from google.adk.agents import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.config import config
from src.utils import strip_code_fences


DEDUP_INSTRUCTION = """<role>You are a UI pattern deduplication specialist identifying repeated element patterns on a single screen.</role>

<task>Analyze the list of actions provided and identify groups of actions that represent the SAME repeated UI pattern. For each group, select representative actions to keep and mark the rest for skipping.</task>

<thinking_process>
Reason through these steps:
1. PATTERN DETECTION — Scan for actions that share the same structure, role, and interaction type. Look for repeated UI patterns such as identical card actions, repeated row buttons, or lists of structurally similar links.
2. DESTINATION ANALYSIS — Before grouping, consider whether actions lead to DIFFERENT destinations or flows. Actions with different purposes should NOT be grouped even if they share the same element type.
3. REPRESENTATIVE SELECTION — For each group, pick representatives that maximize variety (different text content, different positions on the page).
</thinking_process>

<rules>
1. Only group actions that share the SAME structure, role, and interaction pattern AND would produce the same type of flow when executed.
2. Do NOT group actions that lead to clearly different destinations or serve different purposes. Navigation links with different labels go to DIFFERENT screens — these are NOT duplicates even if they share the same role/selector pattern.
3. DO group repeated UI patterns like product cards, table rows, or list items where each instance triggers the same type of flow (e.g. multiple "Add to cart" buttons on a product listing).
4. Do NOT group form input fields — each input field serves a unique purpose.
5. Do NOT group assertion elements — they are read-only checks with negligible cost.
</rules>

<output_format>
Return ONLY valid JSON:
{{
  "groups": [
    {{
      "pattern": "<short description of the repeated pattern>",
      "representative_ids": ["<action_id>", "<action_id>"],
      "skip_ids": ["<action_id>", "<action_id>"]
    }}
  ]
}}
If there are NO repeated patterns, return: {{"groups": []}}
</output_format>"""


def _dedup_before_model_callback(callback_context, llm_request):
    """Inject actions as user message. Skip LLM if too few actions."""
    state = callback_context.state
    actions = state.get("dedup_actions", [])
    k = config.DEDUP_REPRESENTATIVES

    if len(actions) <= k:
        state["dedup_skip_ids"] = []
        skip = json.dumps({"groups": []})
        state["dedup_result_raw"] = skip
        return LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=skip)]
            ),
        )

    actions_for_llm = [
        {
            "action_id": a["action_id"] if isinstance(a, dict) else a.action_id,
            "type": (a["type"] if isinstance(a, dict) else a.type.value),
            "scenario": (a["scenario"] if isinstance(a, dict) else a.scenario.value),
            "label": (a.get("element_info", {}).get("text") or a.get("element_info", {}).get("selector", "")) if isinstance(a, dict) else (a.element_info.text or a.element_info.selector),
            "selector": (a.get("element_info", {}).get("selector", "")) if isinstance(a, dict) else a.element_info.selector,
            "role": (a.get("element_info", {}).get("role", "")) if isinstance(a, dict) else a.element_info.role,
        }
        for a in actions
        if (a["type"] if isinstance(a, dict) else a.type.value) != "assertion"
    ]

    if len(actions_for_llm) <= k:
        state["dedup_skip_ids"] = []
        skip = json.dumps({"groups": []})
        state["dedup_result_raw"] = skip
        return LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=skip)]
            ),
        )

    context_msg = (
        f"Representatives to keep per group: {k}\n\n"
        f"Actions:\n{json.dumps(actions_for_llm, indent=2)}"
    )

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


def _dedup_after_model_callback(callback_context, llm_response):
    """Parse the LLM response and store dedup_skip_ids."""
    state = callback_context.state
    actions = state.get("dedup_actions", [])

    raw = ""
    if llm_response.content and llm_response.content.parts:
        raw = llm_response.content.parts[0].text or ""
    raw = strip_code_fences(raw)

    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        state["dedup_skip_ids"] = []
        return None

    skip_ids = []
    for group in result.get("groups", []):
        skip_ids.extend(group.get("skip_ids", []))

    valid_ids = {(a["action_id"] if isinstance(a, dict) else a.action_id) for a in actions}
    state["dedup_skip_ids"] = [
        aid for aid in skip_ids if aid in valid_ids
    ]
    return None


dedup_agent = Agent(
    name="dedup",
    description="Identifies repeated UI patterns and marks duplicates.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=DEDUP_INSTRUCTION,
    output_key="dedup_result_raw",
    before_model_callback=_dedup_before_model_callback,
    after_model_callback=_dedup_after_model_callback,
)
