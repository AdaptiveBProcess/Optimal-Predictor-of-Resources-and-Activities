from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessModel:
    activities: set[str]
    start_activities: set[str]
    end_activities: set[str]
    resources: set[str] | None = None
