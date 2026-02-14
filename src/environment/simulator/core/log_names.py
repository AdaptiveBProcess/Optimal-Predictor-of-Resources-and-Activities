from dataclasses import dataclass

@dataclass
class LogColumnNames:
    case_id: str
    activity: str
    resource: str
    start_timestamp: str
    end_timestamp: str