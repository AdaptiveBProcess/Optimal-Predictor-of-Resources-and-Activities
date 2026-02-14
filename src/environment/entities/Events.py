from dataclasses import dataclass

@dataclass
class Event:
    start_timestamp: str
    end_timestamp: str
    activity: str
    resource: str
    case_id: str