import base64

from google import genai
from src.config import config


def bytes_to_state(data: bytes) -> str:
    """Encode bytes as base64 string for ADK state storage."""
    if not data:
        return ""
    return base64.b64encode(data).decode("ascii")


def state_to_bytes(data) -> bytes:
    """Decode base64 string from ADK state back to bytes."""
    if not data:
        return b""
    if isinstance(data, bytes):
        return data
    return base64.b64decode(data)

# Shared Gemini client — single instance across all agents
gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# State builders (used by iteration_agent's before_model_callback)
# ---------------------------------------------------------------------------

def build_screens_summary(all_screens: list[dict]) -> str:
    if not all_screens:
        return "(no screens discovered yet)"
    lines = []
    for s in all_screens:
        status = s.get("status", "unknown")
        desc = s.get("screen_description", "no description")
        summary = s.get("actions_summary", {})
        lines.append(
            f"- {s['screen_id']}: \"{desc}\" [{status.upper()}] "
            f"— {summary.get('total', 0)} actions "
            f"({summary.get('pending', 0)} pending, "
            f"{summary.get('completed', 0)} completed, "
            f"{summary.get('failed', 0)} failed)"
        )
    return "\n".join(lines)


def build_nav_graph(edges: list[dict]) -> str:
    if not edges:
        return "(no navigation edges yet — first screen)"
    lines = []
    for e in edges:
        lines.append(
            f"- {e['from_screen']} "
            f"→({e.get('action_type', '?')} "
            f"\"{e.get('action_text', '?')}\")→ "
            f"{e['to_screen']}"
        )
    return "\n".join(lines)


def build_history(history: list[dict]) -> str:
    if not history:
        return "(exploration just started)"
    lines = []
    for i, step in enumerate(history[-15:], 1):
        line = (
            f"{i}. {step['action_type']} "
            f"\"{step['action_text']}\" → {step['verdict']}"
        )
        if step.get("new_screen"):
            line += f" (discovered: {step['new_screen']})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path-finding helpers
# ---------------------------------------------------------------------------

def count_consecutive_failures(history: list[dict]) -> int:
    """Count how many of the last steps were failures."""
    count = 0
    for step in reversed(history):
        if step.get("verdict") == "failed":
            count += 1
        else:
            break
    return count


def find_path(
    nav_edges: list[dict],
    from_screen: str,
    to_screen: str,
) -> list[dict] | None:
    """BFS on nav_edges to find shortest path between two screens."""
    if from_screen == to_screen:
        return []

    adj: dict[str, list[dict]] = {}
    for edge in nav_edges:
        adj.setdefault(edge["from_screen"], []).append(edge)

    visited = {from_screen}
    queue = [(from_screen, [])]

    while queue:
        current, path = queue.pop(0)
        for edge in adj.get(current, []):
            next_screen = edge["to_screen"]
            if next_screen in visited:
                continue
            new_path = path + [edge]
            if next_screen == to_screen:
                return new_path
            visited.add(next_screen)
            queue.append((next_screen, new_path))

    return None
