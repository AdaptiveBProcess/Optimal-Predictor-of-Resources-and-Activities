from dataclasses import dataclass, field


@dataclass
class Case:
    case_id: str
    events: list[str]
    start_time: float = 0.0
    end_time: float = 0.0
    cycle_time: float = 0.0
    activity_history: list = field(default_factory=list)

    @property
    def current_activity(self):
        return self.activity_history[-1] if self.activity_history else None