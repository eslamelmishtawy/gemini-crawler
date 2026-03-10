from pydantic import BaseModel, Field
from datetime import datetime


class NavEdge(BaseModel):
    edge_id: str
    from_screen: str
    to_screen: str
    via_action: str
    action_type: str = ""
    action_text: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
