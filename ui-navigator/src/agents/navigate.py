"""NavigateToAgent — replays known actions to reach a target screen.

Reads navigate_target_screen_id from session state (set by the
iteration_agent's set_navigation_target tool), then BFS-finds a path
and replays each action via the navigator sub-agent.
"""

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types

from src.models.action import ActionDoc
from src.tools.firestore_tools import get_action, get_nav_edges
from src.utils import find_path


class NavigateToAgent(BaseAgent):
    """Replay the known action chain to reach a target screen."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        target_id = state.get("navigate_target_screen_id")
        first_screen_id = state.get("first_screen_id")
        target_url = state.get("target_url")
        session = state["session"]
        known_screens = state["known_screens"]

        run_id = state["run_id"]

        if not target_id or target_id not in known_screens:
            yield self._msg(ctx, f"Invalid target: {target_id}")
            return

        nav_edges = get_nav_edges(run_id)
        path = find_path(nav_edges, first_screen_id, target_id)
        if path is None:
            yield self._msg(
                ctx,
                f"No path from {first_screen_id} to {target_id}",
            )
            return

        # Reset to start and replay
        await session.goto(target_url)

        for edge in path:
            action_dict = get_action(run_id, edge["via_action"])
            if not action_dict:
                yield self._msg(
                    ctx,
                    f"Replay failed: action {edge['via_action']} not found",
                )
                return

            action = ActionDoc(**action_dict)

            state["nav_session"] = session
            state["nav_action"] = action
            state["nav_result"] = None
            state["_nav_attempt"] = 0
            state["_nav_last_error"] = None

            async for event in self._invoke("navigator", ctx):
                yield event

            nav_result = state["nav_result"]
            if not nav_result["success"]:
                yield self._msg(
                    ctx,
                    f"Replay failed at {action.type.value} "
                    f'"{action.element_info.text}"',
                )
                return

        # Successfully navigated
        state["current_screen_id"] = target_id
        state["current_description"] = known_screens.get(target_id, "")

        yield self._msg(ctx, f"Navigated to {target_id}")

    # ------------------------------------------------------------------

    async def _invoke(self, name, ctx):
        agent = next(
            (a for a in self.sub_agents if a.name == name), None
        )
        if agent is None:
            raise ValueError(f"Sub-agent '{name}' not found")
        deferred = []
        async for event in agent.run_async(ctx):
            if event.actions.escalate:
                deferred.append(event)
            else:
                yield event
        for event in deferred:
            event.actions.escalate = False
            yield event

    def _msg(self, ctx, text):
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=text)]
            ),
        )
