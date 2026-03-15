from dataclasses import dataclass, field


@dataclass
class Token:
    case_id: str
    token_id: str
    events: list[str]
    start_time: float = 0.0
    end_time: float = 0.0
    cycle_time: float = 0.0
    activity_history: list = field(default_factory=list)