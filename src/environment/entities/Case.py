from dataclasses import dataclass


@dataclass
class Case:
    case_id: str
    events: list[str]
    start_time: float = 0.0
    end_time: float = 0.0
    cycle_time: float = 0.0