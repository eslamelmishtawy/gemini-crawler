from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ScreenStatus(str, Enum):
    DISCOVERED = "discovered"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    FULLY_EXPLORED = "fully_explored"


class ScreenDoc(BaseModel):
    screen_id: str
    screen_description: str = ""
    url_or_activity: str = ""
    platform: str = "web"
    resolution: dict = Field(default_factory=lambda: {"width": 1280, "height": 720})
    status: ScreenStatus = ScreenStatus.DISCOVERED
    actions_summary: dict = Field(default_factory=lambda: {
        "total": 0, "pending": 0, "completed": 0, "failed": 0, "skipped": 0,
    })
    screenshot_url: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_visited: datetime = Field(default_factory=datetime.utcnow)
