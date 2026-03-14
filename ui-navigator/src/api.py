"""FastAPI wrapper for Cloud Run deployment.

Endpoints:
    POST /explore     — Start autonomous exploration of a URL
    GET  /health      — Health check
    GET  /runs        — List recent runs
    GET  /runs/{id}   — Get run details with screens/actions/edges
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="UI Navigator",
    description="Autonomous AI agent that explores and tests web applications",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class ExploreRequest(BaseModel):
    url: str
    platform: str = "web"
    headless: bool = True


class ExploreResponse(BaseModel):
    run_id: str
    screens_discovered: int
    actions_executed: int
    nav_edges: int
    elapsed_seconds: float
    target_url: str
    interrupted: bool
    interrupt_reason: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ui-navigator"}


@app.post("/explore", response_model=ExploreResponse)
async def explore(req: ExploreRequest):
    """Start an autonomous exploration run."""
    # Lazy import — agent tree is heavy and should not block startup
    from src.main import run_exploration

    try:
        result = await run_exploration(
            target_url=req.url,
            platform=req.platform,
            headless=req.headless,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ExploreResponse(
        run_id=result.get("run_id", ""),
        screens_discovered=result.get("screens_discovered", 0),
        actions_executed=result.get("actions_executed", 0),
        nav_edges=result.get("nav_edges", 0),
        elapsed_seconds=result.get("elapsed_seconds", 0),
        target_url=result.get("target_url", req.url),
        interrupted=result.get("interrupted", False),
        interrupt_reason=result.get("interrupt_reason", ""),
    )


@app.get("/runs")
async def list_runs(limit: int = 20):
    """List recent exploration runs."""
    from src.tools.firestore_tools import get_all_runs
    return get_all_runs(limit=limit)


@app.get("/runs/{run_id}")
async def get_run_detail(run_id: str):
    """Get full details for a run including screens, actions, and edges."""
    from src.tools.firestore_tools import (
        get_run, get_all_screens, get_nav_edges, get_actions_by_screen,
    )

    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    screens = get_all_screens(run_id)
    edges = get_nav_edges(run_id)

    actions_by_screen = {}
    for screen in screens:
        sid = screen["screen_id"]
        actions_by_screen[sid] = get_actions_by_screen(run_id, sid)

    return {
        "run": run,
        "screens": screens,
        "nav_edges": edges,
        "actions_by_screen": actions_by_screen,
    }
