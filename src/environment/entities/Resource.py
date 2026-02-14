from dataclasses import dataclass, field
from typing import Set

@dataclass(frozen=True)
class Resource:
    id: str
    name: str | None = None
    skills: Set[str] = field(default_factory=set)
    capacity: int = 1

    def __post_init__(self):
        # normalize name
        object.__setattr__(self, "name", self.name or self.id)

    def can_execute(self, activity) -> bool:
        return activity.name in self.skills

    def __str__(self):
        return f"Resource(id={self.id}, skills={len(self.skills)})"
