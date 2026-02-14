from dataclasses import dataclass


@dataclass
class Case:
    case_id: str
    events: list[str]