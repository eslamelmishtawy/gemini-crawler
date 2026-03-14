"""ADK entry point for the UI Navigator agent.

This module is discovered by `adk web agents/` and exposes root_agent.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` imports work
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.agent_definitions import root_agent

__all__ = ["root_agent"]
