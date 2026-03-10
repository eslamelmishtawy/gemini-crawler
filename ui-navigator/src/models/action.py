from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    CLICK = "click"
    FILL = "fill"
    SCROLL = "scroll"
    SWIPE = "swipe"
    TAP = "tap"
    SELECT = "select"
    TOGGLE = "toggle"
    NAVIGATE = "navigate"
    ASSERTION = "assertion"


class ActionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionScenario(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ElementInfo(BaseModel):
    selector: str = ""
    text: str = ""
    role: str = ""
    bounds: Optional[dict] = None
    visible: bool = True


class ActionOutcome(BaseModel):
    result: str = ""
    target_screen: Optional[str] = None
    error_message: str = ""


class ActionDoc(BaseModel):
    action_id: str
    screen_id: str
    type: ActionType
    scenario: ActionScenario = ActionScenario.NEUTRAL
    element_info: ElementInfo = Field(default_factory=ElementInfo)
    fill_value: str = ""
    status: ActionStatus = ActionStatus.PENDING
    actual_outcome: Optional[ActionOutcome] = None
    error: Optional[str] = None
    retry_count: int = 0
    discovered_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    executed_at: Optional[str] = None
