"""ExploreCurrentAgent — executes one action on the current screen.

Runs the full pipeline for a single exploration step:
    scout → (data_provider) → navigator → sentinel_compare
    → (page_analyzer → sentinel_match → dedup) on new screen

Reads/writes session state for cross-iteration persistence.
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types

from src.agents.sentinel import StateVerdict
from src.models.action import ActionDoc, ActionType
from src.tools.firestore_tools import (
    create_screen, create_actions_batch, update_action_status,
    update_action_fill_value, mark_actions_skipped,
    get_actions_by_screen, create_nav_edge,
)
from src.tools.gcs_tools import upload_screenshot
from src.utils import strip_code_fences, bytes_to_state

import json


class ExploreCurrentAgent(BaseAgent):
    """Execute one action on the current screen and handle the result."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        session = state["session"]
        current_screen_id = state["current_screen_id"]
        current_description = state["current_description"]
        known_screens = state["known_screens"]
        history = state["history"]

        screen_actions = get_actions_by_screen(current_screen_id)

        # --- Skip assertion actions (no UI interaction) ---
        assertion_ids = [
            a["action_id"] for a in screen_actions
            if a.get("type") == ActionType.ASSERTION.value
            and a.get("status") == "pending"
        ]
        if assertion_ids:
            mark_actions_skipped(assertion_ids)
        screen_actions = [
            a for a in screen_actions
            if a.get("type") != ActionType.ASSERTION.value
        ]

        # --- Scout picks next action ---
        state["scout_actions"] = screen_actions
        state["scout_description"] = current_description
        state["scout_known_screens"] = known_screens
        state["scout_guidance"] = state.get("iteration_scout_guidance", "")

        async for event in self._invoke("scout", ctx):
            yield event

        scout_result = _parse_scout(state.get("scout_result_raw", ""))
        if not scout_result["action_id"]:
            return

        chosen_id = scout_result["action_id"]
        action_dict = next(
            (a for a in screen_actions if a["action_id"] == chosen_id),
            None,
        )
        if not action_dict:
            return
        action = ActionDoc(**action_dict)

        # --- Data Provider for FILL actions ---
        if action.type == ActionType.FILL and not action.fill_value:
            state["dp_action"] = action
            state["dp_description"] = current_description
            state["dp_target_url"] = state.get("target_url", "")

            async for event in self._invoke("data_provider", ctx):
                yield event

            dp = state["dp_result"]  # dict
            action = ActionDoc(**dp)
            update_action_fill_value(action.action_id, action.fill_value)

        # --- Navigator executes ---
        screenshot_before = await session.screenshot()
        state["nav_session"] = session
        state["nav_action"] = action
        state["nav_result"] = None
        state["_nav_attempt"] = 0
        state["_nav_last_error"] = None

        async for event in self._invoke("navigator", ctx):
            yield event

        nav_result = state["nav_result"]
        state["total_actions_executed"] = (
            state.get("total_actions_executed", 0) + 1
        )

        if not nav_result["success"]:
            update_action_status(
                action.action_id, "failed",
                outcome={
                    "result": "execution_failed",
                    "error_message": nav_result["action"]["error"] or "",
                },
                error=nav_result["action"]["error"],
                executed_at=(
                    nav_result["action"]["executed_at"]
                ),
            )
            history.append({
                "action_type": action.type.value,
                "action_text": action.element_info.text,
                "verdict": "failed",
                "error": nav_result["action"]["error"],
            })
            return

        # --- Assertion actions: skip sentinel (no browser action occurred) ---
        if action.type == ActionType.ASSERTION:
            update_action_status(
                action.action_id, "completed",
                outcome={"result": "same_screen"},
                executed_at=nav_result["action"].get("executed_at", ""),
            )
            history.append({
                "action_type": action.type.value,
                "action_text": action.element_info.text,
                "verdict": "same_screen",
            })
            return

        # --- Sentinel compares before/after ---
        state["sentinel_screenshot_before"] = bytes_to_state(screenshot_before)
        state["sentinel_screenshot_after"] = bytes_to_state(nav_result["screenshot_after"]) if isinstance(nav_result["screenshot_after"], bytes) else nav_result["screenshot_after"]

        async for event in self._invoke("sentinel_compare", ctx):
            yield event

        verdict = state["sentinel_verdict"]
        history_entry = {
            "action_type": action.type.value,
            "action_text": action.element_info.text,
            "verdict": verdict["verdict"],
        }

        if verdict["verdict"] in (
            StateVerdict.SAME_SCREEN,
            StateVerdict.ERROR,
            StateVerdict.MODAL,
        ):
            update_action_status(
                action.action_id, "completed",
                outcome={"result": verdict["verdict"]},
                executed_at=(
                    nav_result["action"]["executed_at"]
                ),
            )
            history.append(history_entry)
            return

        if verdict["verdict"] != StateVerdict.NEW_SCREEN:
            update_action_status(
                action.action_id, "completed",
                outcome={"result": verdict["verdict"]},
                executed_at=(
                    nav_result["action"]["executed_at"]
                ),
            )
            history_entry["verdict"] = "failed"
            history.append(history_entry)
            return

        # --- NEW SCREEN: analyze, match, register ---
        async for event in self._handle_new_screen(
            ctx, session, action, nav_result,
            current_screen_id, known_screens, history, history_entry,
        ):
            yield event

    # ------------------------------------------------------------------

    async def _handle_new_screen(
        self, ctx, session, action, nav_result,
        current_screen_id, known_screens, history, history_entry,
    ):
        """Page analyzer → sentinel match → register/dedup."""
        state = ctx.session.state

        new_screenshot = await session.screenshot()
        new_page_source = await session.get_page_source()
        new_url = await session.get_current_url()

        state["pa_screenshot"] = bytes_to_state(new_screenshot)
        state["pa_page_source"] = new_page_source

        async for event in self._invoke("page_analyzer", ctx):
            yield event

        new_screen = state["pa_screen"]
        new_desc = state["pa_description"]
        new_actions = state["pa_actions"]

        # --- Screen Matcher ---
        state["sentinel_new_desc"] = new_desc
        state["sentinel_known_screens"] = known_screens

        async for event in self._invoke("sentinel_match", ctx):
            yield event

        match_result = state["sentinel_match"]

        if match_result["match"]:
            matched_id = match_result["screen_id"]
            if matched_id != current_screen_id:
                _create_edge(
                    current_screen_id, matched_id,
                    action.action_id, action.type.value,
                    action.element_info.text,
                )
                state["total_edges"] = state.get("total_edges", 0) + 1

            update_action_status(
                action.action_id, "completed",
                outcome={
                    "result": "new_screen",
                    "target_screen": matched_id,
                },
                executed_at=(
                    nav_result["action"]["executed_at"]
                ),
            )
            history_entry["new_screen"] = f"{matched_id} (known)"
            history.append(history_entry)
            state["current_screen_id"] = matched_id
            state["current_description"] = known_screens[matched_id]
            return

        # --- Register new screen ---
        new_screen["url_or_activity"] = new_url
        new_screen["screenshot_url"] = upload_screenshot(
            new_screenshot, new_screen["screen_id"],
        )
        known_screens[new_screen["screen_id"]] = new_desc
        create_screen(new_screen)
        if new_actions:
            create_actions_batch(new_actions)

        # --- Dedup ---
        state["dedup_actions"] = new_actions

        async for event in self._invoke("dedup", ctx):
            yield event

        skip_ids = state["dedup_skip_ids"]
        if skip_ids:
            mark_actions_skipped(skip_ids)

        _create_edge(
            current_screen_id, new_screen["screen_id"],
            action.action_id, action.type.value,
            action.element_info.text,
        )
        state["total_edges"] = state.get("total_edges", 0) + 1

        update_action_status(
            action.action_id, "completed",
            outcome={
                "result": "new_screen",
                "target_screen": new_screen["screen_id"],
            },
            executed_at=(
                nav_result["action"]["executed_at"]
            ),
        )
        history_entry["new_screen"] = new_screen["screen_id"]
        history.append(history_entry)
        state["current_screen_id"] = new_screen["screen_id"]
        state["current_description"] = new_desc

    # ------------------------------------------------------------------

    async def _invoke(
        self, name: str, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        agent = next(
            (a for a in self.sub_agents if a.name == name), None
        )
        if agent is None:
            raise ValueError(f"Sub-agent '{name}' not found")
        deferred = []
        async for event in agent.run_async(ctx):
            # Buffer escalate events so the inner LoopAgent (navigator)
            # sees escalate and exits its loop, but we don't yield them
            # upstream where the exploration_loop would also exit.
            if event.actions.escalate:
                deferred.append(event)
            else:
                yield event
        # Re-yield with escalate stripped
        for event in deferred:
            event.actions.escalate = False
            yield event


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_scout(raw: str) -> dict:
    raw = strip_code_fences(raw.strip()) if raw else ""
    try:
        result = json.loads(raw)
        return {
            "action_id": result.get("action_id"),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, AttributeError, ValueError):
        return {"action_id": None, "reasoning": "Failed to parse"}


def _create_edge(
    from_screen: str, to_screen: str,
    via_action: str, action_type: str, action_text: str,
):
    edge_id = f"edge_{uuid.uuid4().hex[:8]}"
    create_nav_edge({
        "edge_id": edge_id,
        "from_screen": from_screen,
        "to_screen": to_screen,
        "via_action": via_action,
        "action_type": action_type,
        "action_text": action_text,
        "created_at": datetime.utcnow().isoformat(),
    })
