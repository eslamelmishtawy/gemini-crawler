"""Dedup Agent — identifies repeated UI patterns and skips duplicates.

Runs after Page Analyzer (which provides the FULL element list) and before
Scout.  Uses an LLM to group actions that would lead to the same flow
(e.g., six identical product cards), keeps K representatives per group,
and marks the rest as skipped.

The logic is entirely generic — no website-specific heuristics.
"""

import json

from src.config import config
from src.utils import gemini_client, strip_code_fences
from src.models.action import ActionDoc

DEDUP_PROMPT = """<role>You are a UI pattern deduplication specialist identifying repeated element patterns on a single screen.</role>

<task>Analyze the list of actions below and identify groups of actions that represent the SAME repeated UI pattern. For each group, select {k} representative actions to keep and mark the rest for skipping.</task>

<actions>
{actions_json}
</actions>

<thinking_process>
Reason through these steps:
1. PATTERN DETECTION — Scan for actions that share the same structure, role, and interaction type. Look for repeated UI patterns such as identical card actions, repeated row buttons, or lists of structurally similar links.
2. DESTINATION ANALYSIS — Before grouping, consider whether actions lead to DIFFERENT destinations or flows. Actions with different purposes should NOT be grouped even if they share the same element type.
3. REPRESENTATIVE SELECTION — For each group, pick {k} representatives that maximize variety (different text content, different positions on the page).
</thinking_process>

<rules>
1. Only group actions that share the SAME structure, role, and interaction pattern AND would produce the same type of flow when executed.
2. Do NOT group actions that lead to clearly different destinations or serve different purposes.
3. Do NOT group form input fields — each input field serves a unique purpose.
4. Do NOT group assertion elements — they are read-only checks with negligible cost.
5. Only group actions of the SAME type (click with click, fill with fill).
6. Actions that do not belong to any repeated pattern should NOT appear in any group.
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


async def dedup_actions(actions: list[ActionDoc]) -> list[str]:
    """Identify duplicate actions and return the action_ids to skip.

    Args:
        actions: Full list of ActionDocs from Page Analyzer.

    Returns:
        List of action_ids that should be marked as skipped.
    """
    if len(actions) <= config.DEDUP_REPRESENTATIVES:
        return []

    # Build concise JSON for the LLM — only fields relevant to dedup
    actions_for_llm = [
        {
            "action_id": a.action_id,
            "type": a.type.value,
            "scenario": a.scenario.value,
            "label": a.element_info.text or a.element_info.selector,
            "selector": a.element_info.selector,
            "role": a.element_info.role,
        }
        for a in actions
        if a.type.value != "assertion"  # skip assertions — rule 4
    ]

    if len(actions_for_llm) <= config.DEDUP_REPRESENTATIVES:
        return []

    prompt = DEDUP_PROMPT.format(
        actions_json=json.dumps(actions_for_llm, indent=2),
        k=config.DEDUP_REPRESENTATIVES,
    )

    response = await gemini_client.aio.models.generate_content(
        model=config.GEMINI_LITE_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )

    raw = strip_code_fences(response.text)
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        return []

    skip_ids = []
    for group in result.get("groups", []):
        ids = group.get("skip_ids", [])
        pattern = group.get("pattern", "repeated pattern")
        print(f"  Dedup: skipping {len(ids)} actions — {pattern}")
        skip_ids.extend(ids)

    # Validate: only skip IDs that actually exist in our action list
    valid_ids = {a.action_id for a in actions}
    skip_ids = [aid for aid in skip_ids if aid in valid_ids]

    return skip_ids
