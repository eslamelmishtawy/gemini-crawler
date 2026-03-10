"""OrchestratorAgent — thin BaseAgent for setup, teardown, and the
exploration LoopAgent.

Responsibilities:
    1. Parse user intent (via intent_parser sub-agent)
    2. Start browser session and analyze the first screen
    3. Initialize shared session state
    4. Delegate to the exploration_loop (LoopAgent)
    5. Teardown and report results

All per-iteration logic lives in iteration_agent + its sub-agents.
"""

import json
import time
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types

from src.utils import strip_code_fences, bytes_to_state
from src.sandbox.session_manager import SessionManager
from src.tools.firestore_tools import (
    create_screen, create_actions_batch,
    mark_actions_skipped, clear_all,
)
from src.tools.gcs_tools import upload_screenshot


class OrchestratorAgent(BaseAgent):
    """Thin ADK BaseAgent — setup/teardown + LoopAgent delegation.

    Sub-agents (wired in agent_definitions.py):
        - intent_parser
        - page_analyzer  (for first screen)
        - dedup          (for first screen)
        - exploration_loop (LoopAgent wrapping the iteration agent)
    """

    platform: str = "web"
    headless: bool = True

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # --- 1. Parse intent ---
        async for event in self._invoke("intent_parser", ctx):
            yield event

        state = ctx.session.state
        intent = self._extract_intent(state.get("intent_result", ""))

        if intent["action"] != "explore" or not intent.get("url"):
            yield self._msg(
                ctx,
                intent.get("reply", "Please provide a URL to explore."),
            )
            return

        target_url = intent["url"]

        # --- 2. Setup browser session ---
        clear_all()
        start_time = time.time()

        session = SessionManager(
            platform=self.platform, headless=self.headless,
        )
        await session.start()
        await session.goto(target_url)

        try:
            # --- 3. Analyze first screen ---
            async for event in self._analyze_first_screen(ctx, session):
                yield event

            screen_id = state["_first_screen_id"]
            screen_desc = state["_first_screen_desc"]

            yield self._msg(
                ctx,
                f"First screen: {screen_id} — {screen_desc}",
            )

            # --- 4. Initialize shared state for the loop ---
            state["session"] = session
            state["target_url"] = target_url
            state["platform"] = self.platform
            state["known_screens"] = {screen_id: screen_desc}
            state["history"] = []
            state["total_actions_executed"] = 0
            state["total_edges"] = 0
            state["current_screen_id"] = screen_id
            state["current_description"] = screen_desc
            state["first_screen_id"] = screen_id

            # --- 5. Run the exploration loop ---
            async for event in self._invoke("exploration_loop", ctx):
                yield event

        finally:
            await session.close()

        # --- 6. Report results ---
        elapsed = time.time() - start_time
        known_screens = state.get("known_screens", {})
        total_actions = state.get("total_actions_executed", 0)
        total_edges = state.get("total_edges", 0)

        state["exploration_result"] = {
            "screens_discovered": len(known_screens),
            "actions_executed": total_actions,
            "nav_edges": total_edges,
            "known_screens": known_screens,
            "history": state.get("history", []),
            "target_url": target_url,
            "elapsed_seconds": round(elapsed, 1),
        }

        yield self._msg(
            ctx,
            f"Exploration complete: {len(known_screens)} screens, "
            f"{total_actions} actions, {total_edges} edges "
            f"in {elapsed:.1f}s",
        )

        # --- 7. Generate detailed report ---
        async for event in self._invoke("reporter", ctx):
            yield event

    # ------------------------------------------------------------------
    # First-screen analysis (page_analyzer + dedup)
    # ------------------------------------------------------------------

    async def _analyze_first_screen(
        self, ctx: InvocationContext, session: SessionManager,
    ) -> AsyncGenerator[Event, None]:
        """Analyze first screen, yielding events so stateDelta is applied."""
        state = ctx.session.state

        screenshot = await session.screenshot()
        page_source = await session.get_page_source()
        current_url = await session.get_current_url()

        state["pa_screenshot"] = bytes_to_state(screenshot)
        state["pa_page_source"] = page_source

        async for event in self._invoke("page_analyzer", ctx):
            yield event

        screen = state["pa_screen"]  # dict from merger
        screen_desc = state["pa_description"]
        actions = state["pa_actions"]  # list of dicts from merger

        screen["url_or_activity"] = current_url
        screen["screenshot_url"] = upload_screenshot(
            screenshot, screen["screen_id"],
        )

        create_screen(screen)
        if actions:
            create_actions_batch(actions)

        # Dedup
        state["dedup_actions"] = actions
        async for event in self._invoke("dedup", ctx):
            yield event

        skip_ids = state.get("dedup_skip_ids", [])
        if skip_ids:
            mark_actions_skipped(skip_ids)

        state["_first_screen_id"] = screen["screen_id"]
        state["_first_screen_desc"] = screen_desc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _invoke(self, name, ctx):
        agent = next(
            (a for a in self.sub_agents if a.name == name), None
        )
        if agent is None:
            raise ValueError(f"Sub-agent '{name}' not found")
        async for event in agent.run_async(ctx):
            yield event

    def _msg(self, ctx, text):
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=text)]
            ),
        )

    def _extract_intent(self, raw: str) -> dict:
        raw = strip_code_fences(raw.strip()) if raw else ""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, AttributeError, ValueError):
            if not raw:
                return {
                    "action": "decline",
                    "reply": "Please provide a URL to explore.",
                }
            return {"action": "decline", "reply": raw}
