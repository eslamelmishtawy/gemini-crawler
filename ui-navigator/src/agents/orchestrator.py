import json
import time
import uuid
from datetime import datetime

from src.config import config
from src.utils import gemini_client, strip_code_fences
from src.sandbox.session_manager import SessionManager
from src.agents.page_analyzer import analyze_screen
from src.agents.navigator import navigate_action
from src.agents.data_provider import provide_fill_value
from src.agents.sentinel import compare_states, match_screen, StateVerdict
from src.agents.scout import recommend_next_action
from src.agents.dedup import dedup_actions
from src.models.action import ActionDoc, ActionType
from src.tools.firestore_tools import (
    create_screen, create_actions_batch, update_action_status,
    update_action_fill_value, mark_actions_skipped, get_action,
    get_actions_by_screen, get_all_screens, get_nav_edges,
    create_nav_edge, update_screen,
)
from src.tools.gcs_tools import upload_screenshot



# ---------------------------------------------------------------------------
# Orchestrator Prompt — the brain that sees EVERYTHING and decides strategy
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """<role>You are the strategic decision-maker for an autonomous UI exploration agent. You see the full state of the exploration and decide what to do next at the SCREEN and FLOW level.</role>

<context>
You do NOT pick individual actions or interact with UI elements — the Scout handles that.
You think strategically: which areas of the application remain unexplored, whether progress is being made, and when to move on or stop.
</context>

<platform>{platform}</platform>

<application_map>
{screens_summary}
</application_map>

<navigation_graph>
{nav_graph}
</navigation_graph>

<current_position>
Screen: {current_screen_id}
Description: "{current_description}"
Status: {current_status}
Pending actions: {pending_count} | Completed: {completed_count} | Failed: {failed_count} | Skipped: {skipped_count}
</current_position>

<exploration_history count="{history_count}">
{exploration_history}
</exploration_history>

<stats>
Screens discovered: {total_screens} | Fully explored: {explored_count}
Actions executed: {total_actions_executed}
Consecutive failures: {consecutive_failures}
</stats>

<situation>{situation}</situation>

<thinking_process>
Reason through these steps in order:

1. APPLICATION MODEL — What type of application is this? Based on screen descriptions and navigation graph, build a mental model of its structure. What sections likely exist that have not been discovered yet?

2. COVERAGE ANALYSIS — How thoroughly has the application been mapped? Are there screens with many untried actions? Are there discovered screens that have not been visited? Are there obvious flows that remain unexplored?

3. PROGRESS EVALUATION — Examine the exploration history. Is real progress being made (new screens discovered, deeper navigation)? Or is the exploration stuck in a loop (same screens, repeated failures, circular navigation)?

4. CURRENT SCREEN ASSESSMENT — Is this screen still worth exploring? How many pending actions remain? Have recent actions here been productive or wasteful?

5. STRATEGIC DECISION — Given the above analysis, determine the best course of action.
</thinking_process>

<decisions>
Choose exactly one:
- "explore_current" — This screen still has valuable unexplored actions. Continue here.
- "navigate_to" — Leave this screen and go to a specific screen that needs exploration. You MUST set "target_screen_id" to a valid screen_id from the application map. Choose the screen that fills the biggest gap in coverage.
- "stop" — The application has been sufficiently mapped, or exploration is irreversibly stuck.
</decisions>

<guidance_rules>
Your scout_guidance should be HIGH-LEVEL strategic direction, not action-level instructions.
Describe WHAT AREA to explore and WHY, not which specific button to click.
</guidance_rules>

<output_format>
Return ONLY valid JSON:
{{
    "decision": "<explore_current | navigate_to | stop>",
    "target_screen_id": "<required if navigate_to>",
    "scout_guidance": "<high-level strategic direction for the Scout>",
    "reasoning": "<your full reasoning covering application model, coverage, progress, and decision>"
}}
</output_format>"""



# ---------------------------------------------------------------------------
# State builder — reads Firestore to build complete snapshot
# ---------------------------------------------------------------------------

def _build_screens_summary(all_screens: list[dict]) -> str:
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


def _build_nav_graph(edges: list[dict]) -> str:
    if not edges:
        return "(no navigation edges yet — first screen)"
    lines = []
    for e in edges:
        lines.append(
            f"- {e['from_screen']} →({e.get('action_type', '?')} \"{e.get('action_text', '?')}\")→ {e['to_screen']}"
        )
    return "\n".join(lines)


def _build_history(history: list[dict]) -> str:
    if not history:
        return "(exploration just started)"
    lines = []
    for i, step in enumerate(history[-15:], 1):
        line = f"{i}. {step['action_type']} \"{step['action_text']}\" → {step['verdict']}"
        if step.get("new_screen"):
            line += f" (discovered: {step['new_screen']})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator reasoning call
# ---------------------------------------------------------------------------

async def _reason(
    platform: str,
    current_screen_id: str,
    current_description: str,
    current_status: str,
    all_screens: list[dict],
    nav_edges: list[dict],
    screen_actions: list[dict],
    history: list[dict],
    total_actions_executed: int,
    consecutive_failures: int,
    situation: str,
) -> dict:
    """Ask Orchestrator to reason about the full state and decide next step."""

    pending = [a for a in screen_actions if a.get("status") == "pending"]
    completed = [a for a in screen_actions if a.get("status") == "completed"]
    failed = [a for a in screen_actions if a.get("status") == "failed"]
    skipped = [a for a in screen_actions if a.get("status") == "skipped"]
    explored_count = sum(1 for s in all_screens if s.get("status") == "fully_explored")

    prompt = ORCHESTRATOR_PROMPT.format(
        platform=platform,
        screens_summary=_build_screens_summary(all_screens),
        nav_graph=_build_nav_graph(nav_edges),
        current_screen_id=current_screen_id,
        current_description=current_description,
        current_status=current_status,
        pending_count=len(pending),
        completed_count=len(completed),
        failed_count=len(failed),
        skipped_count=len(skipped),
        exploration_history=_build_history(history),
        history_count=min(len(history), 15),
        total_screens=len(all_screens),
        explored_count=explored_count,
        total_actions_executed=total_actions_executed,
        consecutive_failures=consecutive_failures,
        situation=situation,
    )

    response = gemini_client.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
    )

    raw = strip_code_fences(response.text)
    try:
        result = json.loads(raw)
        decision = result.get("decision", "explore_current")
        if decision not in ("explore_current", "navigate_to", "stop"):
            decision = "explore_current"
        return {
            "decision": decision,
            "target_screen_id": result.get("target_screen_id"),
            "scout_guidance": result.get("scout_guidance", ""),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "decision": "explore_current",
            "target_screen_id": None,
            "scout_guidance": "",
            "reasoning": "Failed to parse orchestrator response",
        }


# ---------------------------------------------------------------------------
# Main exploration loop
# ---------------------------------------------------------------------------

async def explore(
    target_url: str,
    platform: str = "web",
    headless: bool = True,
) -> dict:
    """Run autonomous exploration on a target URL.

    The Orchestrator reads full state from Firestore before every decision,
    delegates to specialized nodes, and tracks exploration history.

    Returns:
        {
            "screens_discovered": int,
            "actions_executed": int,
            "nav_edges": int,
            "known_screens": dict[str, str],
        }
    """
    start_time = time.time()
    session = SessionManager(platform=platform, headless=headless)
    await session.start()
    await session.goto(target_url)

    # Exploration state (in-memory, also reflected in Firestore)
    known_screens: dict[str, str] = {}  # {screen_id: description}
    history: list[dict] = []  # exploration steps log
    total_actions_executed = 0
    total_edges = 0

    try:
        # --- Step 1: Analyze the first screen ---
        current_screen_id, current_description = await _analyze_and_register(
            session, known_screens
        )
        first_screen_id = current_screen_id

        # --- Main exploration loop ---
        iteration = 0
        while iteration < config.MAX_ITERATIONS:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{config.MAX_ITERATIONS} ---")

            try:
                # Read full state from Firestore (one read per collection)
                all_screens = get_all_screens()
                nav_edges = get_nav_edges()
                screen_actions = get_actions_by_screen(current_screen_id)

                current_status = "analyzed"
                for s in all_screens:
                    if s["screen_id"] == current_screen_id:
                        current_status = s.get("status", "analyzed")
                        break

                # Derive pending from the single read
                pending = [a for a in screen_actions if a.get("status") == "pending"]
                consecutive_failures = _count_consecutive_failures(history)

                # Mark fully explored if no pending actions
                if not pending:
                    update_screen(current_screen_id, {"status": "fully_explored"})
                    print(f"Screen {current_screen_id} fully explored (no pending actions)")

                # Determine situation
                if not pending:
                    situation = "Current screen has no pending actions. Navigate to another screen or stop."
                elif consecutive_failures >= 3:
                    situation = f"{consecutive_failures} consecutive actions failed. May need to change strategy or abandon screen."
                else:
                    situation = "Ready to execute next action on current screen."

                # --- Orchestrator reasons about full state ---
                decision = await _reason(
                    platform=platform,
                    current_screen_id=current_screen_id,
                    current_description=current_description,
                    current_status=current_status,
                    all_screens=all_screens,
                    nav_edges=nav_edges,
                    screen_actions=screen_actions,
                    history=history,
                    total_actions_executed=total_actions_executed,
                    consecutive_failures=consecutive_failures,
                    situation=situation,
                )

                print(f"Orchestrator: {decision['decision']} — {decision['reasoning']}")

                # --- Execute decision ---
                if decision["decision"] == "stop":
                    print("Orchestrator decided to stop exploration.")
                    break

                elif decision["decision"] == "navigate_to":
                    target_id = decision.get("target_screen_id")
                    if not target_id or target_id not in known_screens:
                        print(f"Invalid navigate_to target: {target_id}. Stopping.")
                        break

                    result = await _navigate_to_screen(
                        session, target_id, first_screen_id,
                        target_url, nav_edges, known_screens,
                    )
                    if result is None:
                        print(f"Failed to navigate to {target_id}. Stopping.")
                        break

                    current_screen_id, current_description = result
                    print(f"Navigated to {current_screen_id}")
                    continue

                elif decision["decision"] == "explore_current":
                    # --- Scout picks the next action (sees ALL actions for context) ---
                    scout_result = await recommend_next_action(
                        screen_actions=screen_actions,
                        screen_description=current_description,
                        known_screens=known_screens,
                        scout_guidance=decision.get("scout_guidance", ""),
                    )

                    if not scout_result["action_id"]:
                        print("Scout: no action to recommend")
                        continue

                    chosen_id = scout_result["action_id"]
                    print(f"Scout: {chosen_id} — {scout_result['reasoning']}")

                    # Rebuild ActionDoc from the already-fetched data
                    action_dict = next(
                        (a for a in screen_actions if a["action_id"] == chosen_id),
                        None,
                    )
                    if not action_dict:
                        print(f"WARNING: Scout chose {chosen_id} but not found in screen_actions")
                        continue
                    action = ActionDoc(**action_dict)

                    # --- Data Provider for FILL actions ---
                    if action.type == ActionType.FILL and not action.fill_value:
                        action = await provide_fill_value(action, current_description, target_url)
                        # Persist fill_value so replay navigation can reuse it
                        update_action_fill_value(action.action_id, action.fill_value)
                        print(f"Data Provider: fill_value = \"{action.fill_value}\"")

                    # --- Navigator executes ---
                    screenshot_before = await session.screenshot()
                    print(f"Navigator: {action.type.value} on \"{action.element_info.text}\"")
                    nav_result = await navigate_action(session, action)

                    total_actions_executed += 1

                    if not nav_result["success"]:
                        print(f"FAILED: {nav_result['action'].error}")
                        update_action_status(action.action_id, "failed", {
                            "result": "execution_failed",
                            "error_message": nav_result["action"].error or "",
                        })
                        history.append({
                            "action_type": action.type.value,
                            "action_text": action.element_info.text,
                            "verdict": "failed",
                            "error": nav_result["action"].error,
                        })
                        continue

                    print(f"Success (attempt {nav_result['attempts']})")

                    # --- Sentinel compares before/after ---
                    screenshot_after = nav_result["screenshot_after"]
                    verdict = await compare_states(screenshot_before, screenshot_after)
                    print(f"Sentinel: {verdict['verdict']} — {verdict['reason']}")

                    history_entry = {
                        "action_type": action.type.value,
                        "action_text": action.element_info.text,
                        "verdict": verdict["verdict"],
                    }

                    if verdict["verdict"] in (StateVerdict.SAME_SCREEN, StateVerdict.ERROR, StateVerdict.MODAL):
                        update_action_status(action.action_id, "completed", {
                            "result": verdict["verdict"],
                        })
                        history.append(history_entry)
                        continue

                    if verdict["verdict"] == StateVerdict.NEW_SCREEN:
                        # --- Page Analyzer on new screen ---
                        new_screenshot = await session.screenshot()
                        new_page_source = await session.get_page_source()
                        new_url = await session.get_current_url()
                        new_result = await analyze_screen(new_screenshot, new_page_source)
                        new_desc = new_result["screen_description"]

                        # --- Screen Matcher ---
                        match_result = await match_screen(new_desc, known_screens)

                        if match_result["match"]:
                            matched_id = match_result["screen_id"]
                            print(f"Screen Matcher: MATCH -> {matched_id}")

                            if matched_id != current_screen_id:
                                _create_edge(
                                    current_screen_id, matched_id,
                                    action.action_id, action.type.value, action.element_info.text,
                                )
                                total_edges += 1
                            else:
                                print(f"WARNING: Self-loop detected, skipping edge")

                            update_action_status(action.action_id, "completed", {
                                "result": "new_screen",
                                "target_screen": matched_id,
                            })
                            history_entry["new_screen"] = f"{matched_id} (known)"
                            history.append(history_entry)

                            current_screen_id = matched_id
                            current_description = known_screens[matched_id]
                            continue

                        else:
                            new_screen = new_result["screen"]
                            new_actions = new_result["actions"]

                            new_screen.url_or_activity = new_url
                            new_screen.screenshot_url = upload_screenshot(
                                new_screenshot, new_screen.screen_id,
                            )
                            known_screens[new_screen.screen_id] = new_desc
                            create_screen(new_screen.model_dump(mode="json"))
                            if new_actions:
                                action_dicts = [a.model_dump(mode="json") for a in new_actions]
                                create_actions_batch(action_dicts)

                            # Dedup new screen's actions
                            skip_ids = await dedup_actions(new_actions)
                            if skip_ids:
                                mark_actions_skipped(skip_ids)
                                print(f"Dedup: skipped {len(skip_ids)}/{len(new_actions)} duplicate actions")

                            print(f"Screen Matcher: NEW -> {new_screen.screen_id}")
                            print(f"Description: {new_desc}")
                            print(f"Actions: {len(new_actions)}")

                            _create_edge(
                                current_screen_id, new_screen.screen_id,
                                action.action_id, action.type.value, action.element_info.text,
                            )
                            total_edges += 1

                            update_action_status(action.action_id, "completed", {
                                "result": "new_screen",
                                "target_screen": new_screen.screen_id,
                            })
                            history_entry["new_screen"] = new_screen.screen_id
                            history.append(history_entry)

                            current_screen_id = new_screen.screen_id
                            current_description = new_desc
                            continue

                    # Verdict was not one of the known enum values — log and continue
                    print(f"WARNING: Unknown Sentinel verdict '{verdict['verdict']}', treating as error")
                    update_action_status(action.action_id, "completed", {
                        "result": verdict["verdict"],
                    })
                    history_entry["verdict"] = "failed"
                    history.append(history_entry)
                    continue

                else:
                    print(f"Unknown decision: {decision['decision']}, stopping.")
                    break

            except Exception as e:
                print(f"ERROR in iteration {iteration}: {e}")
                history.append({
                    "action_type": "error",
                    "action_text": str(e),
                    "verdict": "failed",
                })
                continue

        else:
            print(f"MAX_ITERATIONS ({config.MAX_ITERATIONS}) reached. Stopping.")

    finally:
        await session.close()

    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETE")
    print(f"Screens discovered: {len(known_screens)}")
    print(f"Actions executed: {total_actions_executed}")
    print(f"Nav edges: {total_edges}")
    for sid, desc in known_screens.items():
        print(f"  {sid}: {desc}")
    print(f"{'='*60}")

    elapsed = time.time() - start_time
    print(f"Exploration took {elapsed:.1f}s ({elapsed/60:.1f}m)")

    return {
        "screens_discovered": len(known_screens),
        "actions_executed": total_actions_executed,
        "nav_edges": total_edges,
        "known_screens": known_screens,
        "history": history,
        "iterations": iteration,
        "hit_max_iterations": iteration >= config.MAX_ITERATIONS,
        "target_url": target_url,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def _analyze_and_register(
    session: SessionManager,
    known_screens: dict[str, str],
) -> tuple[str, str]:
    """Analyze current screen and register it. Returns (screen_id, description).

    Used only for the very first screen where there are no known screens to match against.
    """
    screenshot = await session.screenshot()
    page_source = await session.get_page_source()
    current_url = await session.get_current_url()

    print("\n--- Analyzing screen... ---")
    result = await analyze_screen(screenshot, page_source)

    screen = result["screen"]
    screen_desc = result["screen_description"]
    actions = result["actions"]

    screen.url_or_activity = current_url
    screen.screenshot_url = upload_screenshot(screenshot, screen.screen_id)
    known_screens[screen.screen_id] = screen_desc

    create_screen(screen.model_dump(mode="json"))
    if actions:
        action_dicts = [a.model_dump(mode="json") for a in actions]
        create_actions_batch(action_dicts)

    # Dedup: identify repeated patterns and skip duplicates
    skip_ids = await dedup_actions(actions)
    if skip_ids:
        mark_actions_skipped(skip_ids)
        print(f"Dedup: skipped {len(skip_ids)}/{len(actions)} duplicate actions")

    print(f"Screen: {screen.screen_id}")
    print(f"Description: {screen_desc}")
    print(f"Actions found: {len(actions)} ({len(actions) - len(skip_ids)} active)")

    return screen.screen_id, screen_desc


def _create_edge(from_screen: str, to_screen: str, via_action: str, action_type: str, action_text: str):
    """Create a nav edge in Firestore."""
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
    print(f"Nav edge: {from_screen} → {to_screen}")


def _count_consecutive_failures(history: list[dict]) -> int:
    """Count how many of the last steps were failures."""
    count = 0
    for step in reversed(history):
        if step.get("verdict") == "failed":
            count += 1
        else:
            break
    return count


def _find_path(nav_edges: list[dict], from_screen: str, to_screen: str) -> list[dict] | None:
    """BFS on nav_edges to find shortest action chain between two screens.

    Returns list of edge dicts in order, or None if no path exists.
    """
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


async def _navigate_to_screen(
    session: SessionManager,
    target_screen_id: str,
    first_screen_id: str,
    target_url: str,
    nav_edges: list[dict],
    known_screens: dict[str, str],
) -> tuple[str, str] | None:
    """Replay the known action chain to reach a target screen.

    1. Find path from first_screen to target via nav_edges
    2. goto(target_url) to reset browser
    3. Re-execute each action along the path
    4. Return (screen_id, description) or None on failure
    """
    path = _find_path(nav_edges, first_screen_id, target_screen_id)
    if path is None:
        print(f"No path found from {first_screen_id} to {target_screen_id}")
        return None

    print(f"Replaying {len(path)} steps to reach {target_screen_id}")

    await session.goto(target_url)

    for edge in path:
        action_dict = get_action(edge["via_action"])
        if not action_dict:
            print(f"  Replay failed: action {edge['via_action']} not found")
            return None

        action = ActionDoc(**action_dict)
        print(f"  Replay: {action.type.value} \"{action.element_info.text}\"")

        nav_result = await navigate_action(session, action)
        if not nav_result["success"]:
            print(f"  Replay failed: {nav_result['action'].error}")
            return None

    return target_screen_id, known_screens.get(target_screen_id, "")
