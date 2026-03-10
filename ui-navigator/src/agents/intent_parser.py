"""IntentParserAgent — LlmAgent that determines user intent.

Uses ADK's LlmAgent so tracing, graph, and state are handled
automatically by the framework.
"""

from google.adk.agents import Agent

from src.config import config

INTENT_INSTRUCTION = """You are an intent parser for a UI exploration agent.

Determine if the user wants to explore a web application.
If yes, extract the URL they want explored.
If no URL is provided but intent is clear, ask for one.

Return ONLY valid JSON:
{"action": "explore", "url": "<the URL>"}
or
{"action": "decline", "reply": "<friendly response explaining what you can do and asking for a URL>"}"""


intent_parser_agent = Agent(
    name="intent_parser",
    description="Parses user messages to determine exploration intent and extract URLs.",
    model=config.GEMINI_LITE_MODEL,
    instruction=INTENT_INSTRUCTION,
    output_key="intent_result",
)
