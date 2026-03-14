"""Reporter Agent — LLM-powered exploration summary and insights.

Called by the orchestrator after the exploration loop completes.
Reads all data from Firestore and session state, builds a structured
context, and asks the LLM for a detailed human-readable analysis report.
Also generates a navigation graph visualization (PNG).
"""

from google.adk.agents import Agent
from google.genai import types

from src.config import config
from src.tools.firestore_tools import (
    get_all_screens,
    get_nav_edges,
    get_actions_by_screen,
)


REPORTER_INSTRUCTION = """\
<role>
You are an expert QA analyst reviewing the results of an autonomous \
UI exploration session. Produce a detailed, well-structured report \
in human-readable markdown format.
</role>

<output_format>
Write the report in clean markdown with the following sections. \
Use headers, tables, bullet points, and bold text for readability.

# Exploration Report

## Executive Summary
One paragraph: what was explored, how well the agent performed, \
overall coverage quality, and whether the exploration was interrupted.

## Statistics
A markdown table with key metrics: screens discovered/explored/remaining, \
actions completed/failed/skipped/pending, nav edges, iterations used vs max, \
time elapsed, and overall coverage percent.

## Navigation Graph
Describe the navigation flow as a list of edges: \
"Screen A → Screen B (via action text)". \
Call out orphan screens (no incoming edges) and dead-end screens \
(no outgoing edges).

## Screen Details
For each screen, include:
- Screen name and URL
- Status (fully_explored / analyzed)
- Coverage percent (excluding assertions)
- Summary of completed, failed, skipped, pending actions
- List failed actions with their error messages
- List pending actions that were not reached

Use a subsection (###) per screen.

## Failure Analysis
- Group errors by pattern/type
- Count occurrences and list affected actions
- Analyze root causes
- Highlight the most problematic screens

## Coverage Assessment
- Overall coverage percentage
- Well-covered areas
- Under-explored areas with reasons

## Iteration Budget
- Was exploration complete?
- Iterations used vs max
- If incomplete: estimate how many more iterations needed and why
- Average iterations per action

## Recommendations
Specific, actionable recommendations for improving coverage or fixing issues.
</output_format>

<guidelines>
- Use screen names/descriptions — NEVER use internal IDs in the report
- Be specific — reference actual screen names, action texts, and errors
- Be honest about gaps and failures
- Provide actionable insights, not vague observations
- When calculating coverage, exclude assertion actions (they are \
informational only and auto-skipped)
- Ensure all numbers are accurate — compute them from the provided data
- Use markdown tables where they improve readability
- If the exploration was interrupted, clearly note this and assess \
what was achieved before the interruption
</guidelines>"""


def _reporter_before_model_callback(callback_context, llm_request):
    """Build full exploration data context for the reporter LLM."""
    state = callback_context.state

    run_id = state.get("run_id", "")
    all_screens = get_all_screens(run_id)
    nav_edges = get_nav_edges(run_id)
    known_screens = state.get("known_screens", {})
    history = state.get("history", [])

    # --- Build per-screen action details ---
    screen_details = []
    total_actions = {"completed": 0, "failed": 0, "skipped": 0, "pending": 0}

    # Build screen_id → description lookup
    sid_to_name = {}
    for sid, desc in known_screens.items():
        sid_to_name[sid] = desc
    for screen in all_screens:
        sid = screen.get("screen_id", "")
        if sid not in sid_to_name:
            sid_to_name[sid] = screen.get("description", sid)

    for screen in all_screens:
        sid = screen.get("screen_id", "")
        screen_name = sid_to_name.get(sid, sid)
        actions = get_actions_by_screen(run_id, sid)
        by_status = {"completed": [], "failed": [], "skipped": [], "pending": []}

        for a in actions:
            status = a.get("status", "pending")
            entry = {
                "type": a.get("type", ""),
                "text": a.get("element_info", {}).get("text", ""),
                "status": status,
            }
            if status == "failed":
                entry["error"] = a.get("error", "")
            if status == "pending":
                entry["type_detail"] = a.get("type", "")
            by_status.get(status, by_status["pending"]).append(entry)

        for s in by_status:
            total_actions[s] += len(by_status[s])

        # Coverage excluding assertions
        actionable = [a for a in actions if a.get("type") != "assertion"]
        actionable_completed = [
            a for a in actionable if a.get("status") == "completed"
        ]
        coverage = (
            f"{len(actionable_completed)}/{len(actionable)} "
            f"({100 * len(actionable_completed) / len(actionable):.0f}%)"
            if actionable else "N/A (no actionable actions)"
        )

        screen_details.append(
            f"### {screen_name}\n"
            f"  Status: {screen.get('status', 'unknown')}\n"
            f"  URL: {screen.get('url_or_activity', 'N/A')}\n"
            f"  Actionable Coverage: {coverage}\n"
            f"  Completed: {_format_actions(by_status['completed'])}\n"
            f"  Failed: {_format_actions(by_status['failed'], include_error=True)}\n"
            f"  Skipped: {_format_actions(by_status['skipped'])}\n"
            f"  Pending: {_format_actions(by_status['pending'])}\n"
        )

    # --- Build edge list ---
    edge_lines = []
    for e in nav_edges:
        from_name = sid_to_name.get(e.get("from_screen", "?"), e.get("from_screen", "?"))
        to_name = sid_to_name.get(e.get("to_screen", "?"), e.get("to_screen", "?"))
        edge_lines.append(
            f"  {from_name} → {to_name} "
            f"(via \"{e.get('action_text', '')}\")"
        )

    # --- Build history summary ---
    history_lines = []
    for i, h in enumerate(history, 1):
        line = (
            f"  {i}. [{h.get('action_type', '')}] "
            f"\"{h.get('action_text', '')}\" → {h.get('verdict', '?')}"
        )
        if h.get("error"):
            line += f" (error: {h['error'][:80]})"
        if h.get("new_screen"):
            new_sid = h["new_screen"].split(" (known)")[0]  # strip "(known)" suffix
            new_name = sid_to_name.get(new_sid, h["new_screen"])
            line += f" → discovered \"{new_name}\""
        history_lines.append(line)

    # --- Assemble context ---
    fully_explored = sum(
        1 for s in all_screens if s.get("status") == "fully_explored"
    )
    iteration_num = state.get("iteration_number", "?")
    elapsed = state.get("exploration_result", {}).get("elapsed_seconds", "?")
    target_url = state.get("target_url", "?")
    interrupted = state.get("exploration_result", {}).get("interrupted", False)
    interrupt_reason = state.get("exploration_result", {}).get("interrupt_reason", "")

    interrupt_note = ""
    if interrupted:
        interrupt_note = f"\nNOTE: Exploration was INTERRUPTED. Reason: {interrupt_reason}\n"

    context_msg = f"""\
<exploration_data>

<target>{target_url}</target>
{interrupt_note}
<overview>
Screens discovered: {len(all_screens)}
Screens fully explored: {fully_explored}
Screens remaining: {len(all_screens) - fully_explored}
Total actions — completed: {total_actions['completed']}, \
failed: {total_actions['failed']}, \
skipped: {total_actions['skipped']}, \
pending: {total_actions['pending']}
Navigation edges: {len(nav_edges)}
Iterations used: {iteration_num} of {config.MAX_ITERATIONS}
Time elapsed: {elapsed}s
</overview>

<navigation_graph>
{chr(10).join(edge_lines) if edge_lines else "No edges recorded."}
</navigation_graph>

<screens>
{chr(10).join(screen_details)}
</screens>

<exploration_history>
{chr(10).join(history_lines) if history_lines else "No history recorded."}
</exploration_history>

</exploration_data>

Analyze the exploration data above and produce your detailed markdown report."""

    llm_request.contents = [
        types.Content(
            role="user", parts=[types.Part(text=context_msg)]
        )
    ]
    return None


def _format_actions(actions, include_error=False):
    """Format a list of actions for display."""
    if not actions:
        return "none"
    lines = []
    for a in actions:
        line = f"{a.get('type', '?')}:\"{a.get('text', '')}\""
        if include_error and a.get("error"):
            line += f" — ERROR: {a['error'][:100]}"
        lines.append(line)
    return "; ".join(lines)


def _reporter_after_model_callback(callback_context, llm_response):
    """Store the markdown report and generate navigation graph PNG."""
    raw = ""
    if llm_response.content and llm_response.content.parts:
        raw = llm_response.content.parts[0].text or ""

    # Generate navigation graph — save as file and embed in report
    try:
        run_id = callback_context.state.get("run_id", "")
        graph_result = _generate_nav_graph(run_id)
        if graph_result:
            graph_path, graph_b64 = graph_result
            callback_context.state["nav_graph_path"] = graph_path
            graph_md = (
                "\n\n## Navigation Graph Visualization\n\n"
                f"Navigation graph saved to: `{graph_path}`\n\n"
                f"![Navigation Graph](data:image/png;base64,{graph_b64})\n"
            )
            raw = raw + graph_md
    except Exception as e:
        raw = raw + f"\n\n*Navigation graph generation failed: {e}*\n"

    callback_context.state["exploration_report"] = raw


def _generate_nav_graph(run_id: str) -> tuple[str, str] | None:
    """Build a directed navigation graph. Returns (file_path, base64_str) or None."""
    import os
    import io
    import base64
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nav_edges = get_nav_edges(run_id)
    all_screens = get_all_screens(run_id)

    G = nx.DiGraph()

    # Add all screens as nodes
    screen_labels = {}
    screen_statuses = {}
    for s in all_screens:
        sid = s.get("screen_id", "unknown")
        desc = s.get("description", sid)
        status = s.get("status", "analyzed")
        # Use description as the label, no IDs
        label = desc[:50] + "..." if len(desc) > 50 else desc
        screen_labels[sid] = label
        screen_statuses[sid] = status
        G.add_node(sid)

    # Add edges
    edge_labels = {}
    for e in nav_edges:
        src = e.get("from_screen", "?")
        dst = e.get("to_screen", "?")
        action_text = e.get("action_text", "")
        label = action_text[:25] + "..." if len(action_text) > 25 else action_text
        G.add_edge(src, dst)
        edge_labels[(src, dst)] = label

    if not G.nodes:
        return None

    # Layout
    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(G.nodes) * 2.5), max(8, len(G.nodes) * 1.5)))

    try:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    except Exception:
        pos = nx.shell_layout(G)

    # Color nodes by status
    node_colors = []
    for node in G.nodes:
        status = screen_statuses.get(node, "analyzed")
        if status == "fully_explored":
            node_colors.append("#4CAF50")  # green
        else:
            node_colors.append("#FF9800")  # orange for remaining

    # Draw
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=3000,
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: screen_labels.get(n, n) for n in G.nodes},
        font_size=7,
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#666666",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        width=1.5,
    )
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=6,
        font_color="#333333",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Fully Explored"),
        Patch(facecolor="#FF9800", label="Partially Explored"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title("Navigation Graph", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    # Save to file
    output_path = os.path.join(os.getcwd(), f"nav_graph_{run_id}.png")
    fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight", facecolor="white")

    # Also capture as base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    return output_path, b64


reporter_agent = Agent(
    name="reporter",
    description="Generates a detailed exploration summary report.",
    model=config.GEMINI_FLASH_MODEL,
    instruction=REPORTER_INSTRUCTION,
    output_key="exploration_report",
    before_model_callback=_reporter_before_model_callback,
    after_model_callback=_reporter_after_model_callback,
)
