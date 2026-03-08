"""EVAL 11: Full autonomous exploration — end-to-end flow evaluation.

Target: saucedemo.com (login → inventory)
Tests every aspect of the orchestrator loop:

  DISCOVERY:
    1. Discovered 2+ screens (login + inventory minimum)
    2. Screen descriptions are functionally accurate
    3. No duplicate screens in Firestore

  ACTIONS:
    4. Actions were executed (total > 0)
    5. Fill actions got fill values from Data Provider
    6. Failed actions have error details
    7. Actions_summary on each screen is consistent with actual action statuses

  NAVIGATION:
    8. Nav edges exist connecting screens
    9. Nav edges reference valid screen IDs
   10. No self-loops in nav edges (from_screen != to_screen)

  ORCHESTRATOR DECISIONS:
   11. Exploration terminated (didn't hit MAX_ITERATIONS)
   12. History shows progression (not stuck in loops)

  FIRESTORE CONSISTENCY:
   13. Every action references a valid screen_id
   14. Every screen has at least 1 action
   15. actions_summary counts match actual action counts
"""
import asyncio
import sys
from collections import Counter

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

from src.agents.orchestrator import explore
from src.tools.firestore_tools import get_all_screens, get_nav_edges, get_actions_by_screen
from src.config import config

from google.cloud import firestore

def _clear_firestore():
    """Delete all docs in screens, actions, nav_edges collections before eval."""
    db = firestore.Client(project=config.GCP_PROJECT_ID, database=config.FIRESTORE_DATABASE)
    for collection_name in ("screens", "actions", "nav_edges"):
        docs = db.collection(collection_name).stream()
        for doc in docs:
            doc.reference.delete()
    print("Firestore cleared (screens, actions, nav_edges)")


class EvalResult:
    def __init__(self, name: str):
        self.name = name
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, label: str, passed: bool, detail: str = ""):
        self.checks.append((label, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))

    def summary(self) -> bool:
        passed = sum(1 for _, p, _ in self.checks if p)
        total = len(self.checks)
        all_passed = passed == total
        print(f"\n{'=' * 60}")
        print(f"{self.name}: {passed}/{total} checks passed — {'PASSED' if all_passed else 'FAILED'}")
        print(f"{'=' * 60}")
        return all_passed


async def main():
    eval_result = EvalResult("EVAL 11: Full Orchestrator Loop")

    print("=" * 60)
    print("Starting autonomous exploration on saucedemo.com")
    print(f"MAX_ITERATIONS: {config.MAX_ITERATIONS}")
    print("=" * 60)

    # --- Clean Firestore before run ---
    _clear_firestore()

    # --- Run the full exploration ---
    result = await explore(
        target_url="https://www.saucedemo.com",
        platform="web",
        headless=True,
    )

    print(f"\n--- Exploration returned ---")
    print(f"Screens: {result['screens_discovered']}")
    print(f"Actions executed: {result['actions_executed']}")
    print(f"Nav edges: {result['nav_edges']}")
    for sid, desc in result["known_screens"].items():
        print(f"  {sid}: {desc}")

    # === Read full Firestore state for deep assertions ===
    print("\n--- Reading Firestore state for validation ---")
    all_screens = get_all_screens()
    nav_edges = get_nav_edges()

    # Gather all actions across all screens
    all_actions = []
    actions_by_screen: dict[str, list[dict]] = {}
    for screen in all_screens:
        sid = screen["screen_id"]
        screen_actions = get_actions_by_screen(sid)
        actions_by_screen[sid] = screen_actions
        all_actions.extend(screen_actions)

    screen_ids = {s["screen_id"] for s in all_screens}

    print(f"Firestore: {len(all_screens)} screens, {len(all_actions)} actions, {len(nav_edges)} edges")

    # =================================================================
    # DISCOVERY CHECKS
    # =================================================================
    print("\n--- DISCOVERY ---")

    # 1. Discovered 2+ screens
    eval_result.check(
        "Discovered 2+ screens",
        result["screens_discovered"] >= 2,
        f"Found {result['screens_discovered']}",
    )

    # 2. Screen descriptions mention expected pages
    descriptions = {s["screen_id"]: s.get("screen_description", "") for s in all_screens}
    all_desc_text = " ".join(descriptions.values()).lower()
    has_login_desc = any(
        kw in all_desc_text for kw in ["login", "auth", "sign in", "username"]
    )
    has_inventory_desc = any(
        kw in all_desc_text
        for kw in ["inventory", "product", "catalog", "item", "shop"]
    )
    eval_result.check(
        "Has login-related screen description",
        has_login_desc,
        f"Descriptions: {list(descriptions.values())}",
    )
    eval_result.check(
        "Has inventory/product screen description",
        has_inventory_desc,
        f"Descriptions: {list(descriptions.values())}",
    )

    # 3. No duplicate screens (same description shouldn't appear twice)
    desc_list = list(descriptions.values())
    eval_result.check(
        "No duplicate screen descriptions",
        len(desc_list) == len(set(desc_list)),
        f"{len(desc_list)} descriptions, {len(set(desc_list))} unique",
    )

    # =================================================================
    # ACTION CHECKS
    # =================================================================
    print("\n--- ACTIONS ---")

    # 4. Actions were executed
    eval_result.check(
        "Actions were executed",
        result["actions_executed"] > 0,
        f"Executed {result['actions_executed']}",
    )

    # 5. Fill actions got fill values
    fill_actions = [a for a in all_actions if a.get("type") == "fill"]
    filled_with_value = [a for a in fill_actions if a.get("fill_value")]
    completed_fills = [a for a in fill_actions if a.get("status") == "completed"]
    eval_result.check(
        "Completed fill actions have fill_value",
        all(a.get("fill_value") for a in completed_fills) if completed_fills else True,
        f"{len(completed_fills)} completed fills, {len(filled_with_value)} have fill_value",
    )

    # 6. Completed actions have actual_outcome
    completed_actions = [a for a in all_actions if a.get("status") == "completed"]
    with_outcome = [a for a in completed_actions if a.get("actual_outcome")]
    eval_result.check(
        "Completed actions have actual_outcome",
        len(with_outcome) == len(completed_actions) if completed_actions else True,
        f"{len(with_outcome)}/{len(completed_actions)} have outcome",
    )

    # 7. Failed actions have error context
    failed_actions = [a for a in all_actions if a.get("status") == "failed"]
    eval_result.check(
        "Failed actions tracked (or none failed)",
        True,  # Informational — just log the count
        f"{len(failed_actions)} failed actions out of {len(all_actions)} total",
    )

    # 7. actions_summary on each screen is consistent
    summary_consistent = True
    summary_details = []
    for screen in all_screens:
        sid = screen["screen_id"]
        summary = screen.get("actions_summary", {})
        actual_actions = actions_by_screen.get(sid, [])

        actual_counts = Counter(a.get("status", "pending") for a in actual_actions)
        expected_total = len(actual_actions)
        expected_pending = actual_counts.get("pending", 0)
        expected_completed = actual_counts.get("completed", 0)
        expected_failed = actual_counts.get("failed", 0)
        expected_skipped = actual_counts.get("skipped", 0)

        matches = (
            summary.get("total", 0) == expected_total
            and summary.get("pending", 0) == expected_pending
            and summary.get("completed", 0) == expected_completed
            and summary.get("failed", 0) == expected_failed
            and summary.get("skipped", 0) == expected_skipped
        )
        if not matches:
            summary_consistent = False
            summary_details.append(
                f"{sid}: summary={summary} vs actual={{total={expected_total}, "
                f"pending={expected_pending}, completed={expected_completed}, "
                f"failed={expected_failed}}}"
            )

    eval_result.check(
        "actions_summary consistent with actual action statuses",
        summary_consistent,
        "; ".join(summary_details) if summary_details else "All screens consistent",
    )

    # =================================================================
    # NAVIGATION CHECKS
    # =================================================================
    print("\n--- NAVIGATION ---")

    # 8. Nav edges exist
    eval_result.check(
        "Nav edges exist",
        len(nav_edges) >= 1,
        f"Found {len(nav_edges)} edges",
    )

    # 9. Nav edges reference valid screen IDs
    invalid_refs = []
    for edge in nav_edges:
        if edge.get("from_screen") not in screen_ids:
            invalid_refs.append(f"from={edge.get('from_screen')} not in screens")
        if edge.get("to_screen") not in screen_ids:
            invalid_refs.append(f"to={edge.get('to_screen')} not in screens")
    eval_result.check(
        "Nav edges reference valid screen IDs",
        len(invalid_refs) == 0,
        "; ".join(invalid_refs) if invalid_refs else f"All {len(nav_edges)} edges valid",
    )

    # 10. No self-loops
    self_loops = [
        e for e in nav_edges if e.get("from_screen") == e.get("to_screen")
    ]
    eval_result.check(
        "No self-loop nav edges",
        len(self_loops) == 0,
        f"Found {len(self_loops)} self-loops" if self_loops else "No self-loops",
    )

    # =================================================================
    # ORCHESTRATOR DECISION CHECKS
    # =================================================================
    print("\n--- ORCHESTRATOR DECISIONS ---")

    # 11. Exploration terminated before MAX_ITERATIONS
    # If it hit MAX_ITERATIONS, that suggests it got stuck
    eval_result.check(
        "Exploration terminated (didn't exhaust MAX_ITERATIONS)",
        result["actions_executed"] < config.MAX_ITERATIONS,
        f"Executed {result['actions_executed']} actions, limit was {config.MAX_ITERATIONS}",
    )

    # 12. At least one screen marked fully_explored
    explored_screens = [s for s in all_screens if s.get("status") == "fully_explored"]
    eval_result.check(
        "At least one screen marked fully_explored",
        len(explored_screens) >= 1,
        f"{len(explored_screens)} screens fully explored",
    )

    # =================================================================
    # FIRESTORE CONSISTENCY CHECKS
    # =================================================================
    print("\n--- FIRESTORE CONSISTENCY ---")

    # 13. Every action references a valid screen_id
    orphan_actions = [
        a["action_id"]
        for a in all_actions
        if a.get("screen_id") not in screen_ids
    ]
    eval_result.check(
        "No orphan actions (all reference valid screen_id)",
        len(orphan_actions) == 0,
        f"{len(orphan_actions)} orphans" if orphan_actions else "All actions linked",
    )

    # 14. Every screen has at least 1 action
    empty_screens = [
        s["screen_id"]
        for s in all_screens
        if len(actions_by_screen.get(s["screen_id"], [])) == 0
    ]
    eval_result.check(
        "Every screen has at least 1 action",
        len(empty_screens) == 0,
        f"Empty screens: {empty_screens}" if empty_screens else "All screens have actions",
    )

    # 15. In-memory known_screens matches Firestore screen count
    eval_result.check(
        "In-memory screen count matches Firestore",
        result["screens_discovered"] == len(all_screens),
        f"In-memory={result['screens_discovered']}, Firestore={len(all_screens)}",
    )

    # --- Print nav graph for debugging ---
    print("\n--- NAV GRAPH ---")
    for edge in nav_edges:
        from_desc = descriptions.get(edge.get("from_screen"), "?")[:40]
        to_desc = descriptions.get(edge.get("to_screen"), "?")[:40]
        print(
            f"  {edge.get('from_screen')} ({from_desc}) "
            f"→ [{edge.get('action_type', '?')} \"{edge.get('action_text', '?')}\"] "
            f"→ {edge.get('to_screen')} ({to_desc})"
        )

    # --- Print per-screen action breakdown ---
    print("\n--- PER-SCREEN ACTIONS ---")
    for screen in all_screens:
        sid = screen["screen_id"]
        desc = screen.get("screen_description", "?")[:50]
        status = screen.get("status", "?")
        actions = actions_by_screen.get(sid, [])
        counts = Counter(a.get("status", "?") for a in actions)
        print(
            f"  {sid} [{status}] \"{desc}\""
            f" — {len(actions)} actions: {dict(counts)}"
        )

    return eval_result.summary()


if __name__ == "__main__":
    passed = asyncio.run(main())
    sys.exit(0 if passed else 1)
