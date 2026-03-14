"""Firestore tools — all data scoped under runs/{run_id}/...

Collections:
    runs/{run_id}                  — run metadata (url, domain, status, timestamps)
    runs/{run_id}/screens/{sid}    — screen documents
    runs/{run_id}/actions/{aid}    — action documents
    runs/{run_id}/nav_edges/{eid}  — navigation edge documents
"""

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from src.config import config

db = firestore.Client(
    project=config.GCP_PROJECT_ID, database=config.FIRESTORE_DATABASE,
)


# ===================================================================
# Run management
# ===================================================================

def create_run(run_id: str, target_url: str, platform: str = "web") -> str:
    """Create a new run document. Returns run_id."""
    domain = urlparse(target_url).netloc or "unknown"
    domain = re.sub(r"^www\.", "", domain)

    doc_ref = db.collection("runs").document(run_id)
    doc_ref.set({
        "run_id": run_id,
        "target_url": target_url,
        "domain": domain,
        "platform": platform,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "stats": {},
    })
    return run_id


def update_run_status(
    run_id: str,
    status: str,
    stats: dict | None = None,
    interrupted: bool = False,
    interrupt_reason: str = "",
) -> None:
    """Update run status (running -> completed / interrupted)."""
    updates: dict = {
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "interrupted": interrupted,
        "interrupt_reason": interrupt_reason,
    }
    if stats:
        updates["stats"] = stats
    db.collection("runs").document(run_id).update(updates)


def get_run(run_id: str) -> dict | None:
    """Get a run document by ID."""
    doc = db.collection("runs").document(run_id).get()
    return doc.to_dict() if doc.exists else None


def get_runs_by_domain(domain: str, limit: int = 20) -> list[dict]:
    """Get all runs for a domain, ordered by most recent first."""
    domain = re.sub(r"^www\.", "", domain)
    docs = (
        db.collection("runs")
        .where(filter=FieldFilter("domain", "==", domain))
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [doc.to_dict() for doc in docs]


def get_all_runs(limit: int = 50) -> list[dict]:
    """Get all runs, ordered by most recent first."""
    docs = (
        db.collection("runs")
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [doc.to_dict() for doc in docs]


# ===================================================================
# Helper — subcollection reference
# ===================================================================

def _screens_col(run_id: str):
    return db.collection("runs").document(run_id).collection("screens")


def _actions_col(run_id: str):
    return db.collection("runs").document(run_id).collection("actions")


def _nav_edges_col(run_id: str):
    return db.collection("runs").document(run_id).collection("nav_edges")


# ===================================================================
# Screen Registry
# ===================================================================

def create_screen(run_id: str, screen_data: dict) -> str:
    """Create a new screen document. Returns screen_id."""
    doc_ref = _screens_col(run_id).document(screen_data["screen_id"])
    doc_ref.set(screen_data)
    return screen_data["screen_id"]


def get_screen(run_id: str, screen_id: str) -> dict | None:
    """Get a screen by ID."""
    doc = _screens_col(run_id).document(screen_id).get()
    return doc.to_dict() if doc.exists else None


def update_screen(run_id: str, screen_id: str, updates: dict) -> None:
    """Update fields on a screen document."""
    _screens_col(run_id).document(screen_id).update(updates)


def get_all_screens(run_id: str) -> list[dict]:
    """Get all screen documents for a run."""
    docs = _screens_col(run_id).stream()
    return [doc.to_dict() for doc in docs]


# ===================================================================
# Action Tracker
# ===================================================================

def create_action(run_id: str, action_data: dict) -> str:
    """Create a new action document. Returns action_id."""
    doc_ref = _actions_col(run_id).document(action_data["action_id"])
    doc_ref.set(action_data)
    return action_data["action_id"]


def create_actions_batch(run_id: str, actions: list[dict]) -> int:
    """Create multiple actions in a batch. Returns count created."""
    batch = db.batch()
    for action_data in actions:
        doc_ref = _actions_col(run_id).document(action_data["action_id"])
        batch.set(doc_ref, action_data)
    batch.commit()
    return len(actions)


def get_pending_actions(run_id: str, screen_id: str) -> list[dict]:
    """Get all pending actions for a screen."""
    docs = (
        _actions_col(run_id)
        .where(filter=FieldFilter("screen_id", "==", screen_id))
        .where(filter=FieldFilter("status", "==", "pending"))
        .stream()
    )
    return [doc.to_dict() for doc in docs]


def get_actions_by_screen(run_id: str, screen_id: str) -> list[dict]:
    """Get ALL actions for a screen (any status)."""
    docs = (
        _actions_col(run_id)
        .where(filter=FieldFilter("screen_id", "==", screen_id))
        .stream()
    )
    return [doc.to_dict() for doc in docs]


def mark_actions_skipped(run_id: str, action_ids: list[str]) -> int:
    """Bulk-skip actions. Returns count skipped."""
    if not action_ids:
        return 0
    batch = db.batch()
    screen_ids = set()
    for action_id in action_ids:
        doc_ref = _actions_col(run_id).document(action_id)
        batch.update(doc_ref, {"status": "skipped"})
        doc = doc_ref.get()
        if doc.exists:
            sid = doc.to_dict().get("screen_id")
            if sid:
                screen_ids.add(sid)
    batch.commit()
    for sid in screen_ids:
        _refresh_screen_summary(run_id, sid)
    return len(action_ids)


def get_action(run_id: str, action_id: str) -> dict | None:
    """Get a single action by ID."""
    doc = _actions_col(run_id).document(action_id).get()
    return doc.to_dict() if doc.exists else None


def update_action_fill_value(run_id: str, action_id: str, fill_value: str) -> None:
    """Persist the fill_value generated by Data Provider."""
    _actions_col(run_id).document(action_id).update({"fill_value": fill_value})


def update_action_status(
    run_id: str,
    action_id: str,
    status: str,
    outcome: dict | None = None,
    error: str | None = None,
    executed_at: str | None = None,
) -> None:
    """Update an action's status and optionally its outcome, error, and executed_at.

    Also refreshes the parent screen's actions_summary.
    """
    updates = {"status": status}
    if outcome:
        updates["actual_outcome"] = outcome
    if error is not None:
        updates["error"] = error
    if executed_at is not None:
        updates["executed_at"] = executed_at
    _actions_col(run_id).document(action_id).update(updates)

    # Refresh parent screen's actions_summary
    action_doc = _actions_col(run_id).document(action_id).get()
    if action_doc.exists:
        screen_id = action_doc.to_dict().get("screen_id")
        if screen_id:
            _refresh_screen_summary(run_id, screen_id)


def _refresh_screen_summary(run_id: str, screen_id: str) -> None:
    """Recount action statuses and update the screen doc's actions_summary."""
    actions = list(
        _actions_col(run_id)
        .where(filter=FieldFilter("screen_id", "==", screen_id))
        .stream()
    )
    summary = {"total": len(actions), "pending": 0, "completed": 0, "failed": 0, "skipped": 0}
    for a in actions:
        s = a.to_dict().get("status", "pending")
        if s in summary:
            summary[s] += 1
    _screens_col(run_id).document(screen_id).update({"actions_summary": summary})


# ===================================================================
# Nav Graph
# ===================================================================

def create_nav_edge(run_id: str, edge_data: dict) -> str:
    """Create a navigation edge. Returns edge_id."""
    doc_ref = _nav_edges_col(run_id).document(edge_data["edge_id"])
    doc_ref.set(edge_data)
    return edge_data["edge_id"]


def get_nav_edges(run_id: str) -> list[dict]:
    """Get all navigation edges for a run."""
    docs = _nav_edges_col(run_id).stream()
    return [doc.to_dict() for doc in docs]
