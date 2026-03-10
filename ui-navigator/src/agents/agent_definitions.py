"""ADK agent tree — wires all agents into the root agent hierarchy.

This is the entry point for `adk web` and the InMemoryRunner.
All agents appear in the ADK graph and trace.

Agent tree:
    orchestrator (BaseAgent — setup/teardown)
    ├── intent_parser (Agent)
    ├── page_analyzer (SequentialAgent — first screen only)
    │   ├── parallel_extraction (ParallelAgent)
    │   │   ├── vision_analyzer (Agent)
    │   │   └── dom_analyzer (Agent)
    │   └── merger (Agent)
    ├── dedup (Agent — first screen only)
    ├── exploration_loop (LoopAgent)
    │   └── iteration (Agent — reasons + transfers via transfer_to_agent)
    │       ├── explore_current (BaseAgent)
    │       │   ├── scout (Agent)
    │       │   ├── data_provider (Agent)
    │       │   ├── navigator (LoopAgent)
    │       │   │   └── code_gen_and_execute (Agent)
    │       │   ├── sentinel_compare (Agent)
    │       │   ├── page_analyzer (SequentialAgent)
    │       │   │   ├── parallel_extraction (ParallelAgent)
    │       │   │   │   ├── vision_analyzer (Agent)
    │       │   │   │   └── dom_analyzer (Agent)
    │       │   │   └── merger (Agent)
    │       │   ├── sentinel_match (Agent)
    │       │   └── dedup (Agent)
    │       ├── navigate_to (BaseAgent)
    │       │   └── navigator (LoopAgent)
    │       │       └── code_gen_and_execute (Agent)
    │       └── stop_exploration (BaseAgent — escalate)
    └── reporter (Agent — post-exploration analysis)
"""

from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent

from src.config import config

# --- Leaf agents (originals, used for first-screen analysis) ---
from src.agents.page_analyzer import page_analyzer_agent
from src.agents.dedup import dedup_agent
from src.agents.intent_parser import intent_parser_agent
from src.agents.scout import scout_agent
from src.agents.data_provider import data_provider_agent

# --- Prompts and callbacks for creating duplicate instances ---
from src.agents.navigator import (
    NAVIGATOR_INSTRUCTION,
    _nav_before_model_callback,
    execute_playwright_code,
)
from src.agents.page_analyzer import (
    VISION_PROMPT,
    _vision_before_model_callback,
    DOM_INSTRUCTION,
    _dom_before_model_callback,
    MERGE_INSTRUCTION,
    _merger_before_model_callback,
    _merger_after_model_callback,
)
from src.agents.sentinel import (
    COMPARE_PROMPT,
    _compare_before_model_callback,
    _compare_after_model_callback,
    MATCH_INSTRUCTION,
    _match_before_model_callback,
    _match_after_model_callback,
)
from src.agents.dedup import (
    DEDUP_INSTRUCTION,
    _dedup_before_model_callback,
    _dedup_after_model_callback,
)

# --- New agents ---
from src.agents.iteration import iteration_agent
from src.agents.explore import ExploreCurrentAgent
from src.agents.navigate import NavigateToAgent
from src.agents.stop import stop_agent
from src.agents.reporter import reporter_agent
from src.agents.orchestrator_agent import OrchestratorAgent


# ===================================================================
# Build duplicate instances for explore_current sub-tree
# (ADK requires unique agent instances per tree position)
# ===================================================================

# -- Navigator (explore) --
navigator_explore = LoopAgent(
    name="navigator",
    description="Generates and executes Playwright code with retries.",
    max_iterations=config.ACTION_RETRY_LIMIT,
    sub_agents=[
        Agent(
            name="code_gen_and_execute",
            description="Generates and executes Playwright code.",
            model=config.GEMINI_FLASH_MODEL,
            instruction=NAVIGATOR_INSTRUCTION,
            tools=[execute_playwright_code],
            before_model_callback=_nav_before_model_callback,
        ),
    ],
)

# -- Navigator (replay for navigate_to) --
navigator_replay = LoopAgent(
    name="navigator",
    description="Generates and executes Playwright code with retries.",
    max_iterations=config.ACTION_RETRY_LIMIT,
    sub_agents=[
        Agent(
            name="code_gen_and_execute",
            description="Generates and executes Playwright code.",
            model=config.GEMINI_FLASH_MODEL,
            instruction=NAVIGATOR_INSTRUCTION,
            tools=[execute_playwright_code],
            before_model_callback=_nav_before_model_callback,
        ),
    ],
)

# -- Page analyzer (for new screen discovery in explore_current) --
page_analyzer_explore = SequentialAgent(
    name="page_analyzer",
    description="Dual vision + DOM analysis to extract UI elements.",
    sub_agents=[
        ParallelAgent(
            name="parallel_extraction",
            description="Runs vision and DOM extraction concurrently.",
            sub_agents=[
                Agent(
                    name="vision_analyzer",
                    description="Extracts UI elements from screenshot.",
                    model=config.GEMINI_FLASH_MODEL,
                    instruction=VISION_PROMPT,
                    output_key="pa_vision_raw",
                    before_model_callback=_vision_before_model_callback,
                ),
                Agent(
                    name="dom_analyzer",
                    description="Extracts elements from HTML DOM.",
                    model=config.GEMINI_FLASH_MODEL,
                    instruction=DOM_INSTRUCTION,
                    output_key="pa_dom_raw",
                    before_model_callback=_dom_before_model_callback,
                ),
            ],
        ),
        Agent(
            name="merger",
            description="Merges vision and DOM analysis.",
            model=config.GEMINI_LITE_MODEL,
            instruction=MERGE_INSTRUCTION,
            output_key="pa_merge_raw",
            before_model_callback=_merger_before_model_callback,
            after_model_callback=_merger_after_model_callback,
        ),
    ],
)

# -- Sentinel compare (for explore_current) --
sentinel_compare_explore = Agent(
    name="sentinel_compare",
    description="Detects state changes via screenshot comparison.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=COMPARE_PROMPT,
    output_key="sentinel_compare_raw",
    before_model_callback=_compare_before_model_callback,
    after_model_callback=_compare_after_model_callback,
)

# -- Sentinel match (for explore_current) --
sentinel_match_explore = Agent(
    name="sentinel_match",
    description="Matches new screens against known screens.",
    model=config.GEMINI_LITE_MODEL,
    instruction=MATCH_INSTRUCTION,
    output_key="sentinel_match_raw",
    before_model_callback=_match_before_model_callback,
    after_model_callback=_match_after_model_callback,
)

# -- Dedup (for explore_current) --
dedup_explore = Agent(
    name="dedup",
    description="Identifies repeated UI patterns and marks duplicates.",
    model=config.GEMINI_LITE_MODEL,
    instruction=DEDUP_INSTRUCTION,
    output_key="dedup_result_raw",
    before_model_callback=_dedup_before_model_callback,
    after_model_callback=_dedup_after_model_callback,
)


# ===================================================================
# Assemble the agent tree
# ===================================================================

explore_current_agent = ExploreCurrentAgent(
    name="explore_current",
    description=(
        "Execute the next action on the current screen. "
        "Handles the full pipeline: scout picks action, "
        "data_provider fills values, navigator executes, "
        "sentinel verifies the result."
    ),
    sub_agents=[
        scout_agent,
        data_provider_agent,
        navigator_explore,
        sentinel_compare_explore,
        page_analyzer_explore,
        sentinel_match_explore,
        dedup_explore,
    ],
)

navigate_to_agent = NavigateToAgent(
    name="navigate_to",
    description=(
        "Navigate to a different screen that needs exploration. "
        "Replays the known action chain to reach the target. "
        "Use when the current screen is exhausted or a more "
        "promising screen exists."
    ),
    sub_agents=[navigator_replay],
)

# Wire sub_agents onto the iteration agent
iteration_agent.sub_agents = [
    explore_current_agent,
    navigate_to_agent,
    stop_agent,
]

exploration_loop = LoopAgent(
    name="exploration_loop",
    description="Main exploration loop — iterates until done or max reached.",
    max_iterations=config.MAX_ITERATIONS,
    sub_agents=[iteration_agent],
)

root_agent = OrchestratorAgent(
    name="orchestrator",
    description=(
        "Strategic decision-maker that drives "
        "the autonomous UI exploration loop."
    ),
    sub_agents=[
        intent_parser_agent,
        page_analyzer_agent,
        dedup_agent,
        exploration_loop,
        reporter_agent,
    ],
)
