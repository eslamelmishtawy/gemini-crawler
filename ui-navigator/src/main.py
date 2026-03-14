"""Entry point for the UI Navigator ADK agent.

Usage:
    python -m src.main                          # uses TARGET_URL from .env
    python -m src.main https://example.com      # override URL
    adk web src/agents                          # ADK Dev UI with graph + traces
"""

import asyncio
import sys

from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

from src.agents.agent_definitions import root_agent
from src.config import config


def create_runner() -> InMemoryRunner:
    """Create an ADK InMemoryRunner with the root agent."""
    return InMemoryRunner(agent=root_agent, app_name="ui-navigator")


async def run_exploration(
    target_url: str | None = None,
    platform: str = "web",
    headless: bool = True,
) -> dict:
    """Run the exploration loop via ADK runner.

    Args:
        target_url: URL to explore. Defaults to config.TARGET_URL.
        platform: "web" (default) or mobile platform.
        headless: Run browser headlessly.

    Returns:
        Exploration result dict from session state.
    """
    url = target_url or config.TARGET_URL

    # Configure the orchestrator
    root_agent.target_url = url
    root_agent.platform = platform
    root_agent.headless = headless

    runner = create_runner()

    print(f"Starting exploration of {url}")
    print(f"Platform: {platform}, Headless: {headless}")
    print("=" * 60)

    # With 50 iterations × ~10 LLM calls each, the default 500 limit is too low.
    run_config = RunConfig(max_llm_calls=5000)

    async for event in runner.run_async(
        user_id="explorer",
        session_id="session_1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=f"Explore {url}")],
        ),
        run_config=run_config,
    ):
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                print(f"[{event.author}] {text}")

    # Retrieve result from session state
    session = await runner.session_service.get_session(
        app_name="ui-navigator",
        user_id="explorer",
        session_id="session_1",
    )
    return session.state.get("exploration_result", {})


def main():
    target_url = sys.argv[1] if len(sys.argv) > 1 else None
    result = asyncio.run(run_exploration(target_url=target_url))
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
