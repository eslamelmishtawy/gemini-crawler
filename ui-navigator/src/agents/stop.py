"""StopAgent — signals the LoopAgent to exit via escalate.

Used by the iteration_agent when exploration is complete or stuck.
The LoopAgent terminates when any sub-agent yields an Event with
escalate=True.
"""

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types


class StopExplorationAgent(BaseAgent):
    """Escalates to exit the exploration LoopAgent."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                parts=[types.Part(text="Exploration stopped.")]
            ),
            actions=EventActions(escalate=True),
        )


stop_agent = StopExplorationAgent(
    name="stop_exploration",
    description=(
        "Stop the exploration. Use when the application has been "
        "sufficiently mapped or exploration is irreversibly stuck."
    ),
)
