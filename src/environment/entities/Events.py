from dataclasses import dataclass

@dataclass
class Event:
    case_id: str
    start_timestamp: str
    end_timestamp: str
    activity: str
    resource: str