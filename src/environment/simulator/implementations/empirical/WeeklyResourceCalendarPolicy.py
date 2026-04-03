from datetime import datetime, timedelta
import numpy as np
from environment.simulator.policies.CalendarPolicy import CalendarPolicy


class WeeklyResourceCalendarPolicy(CalendarPolicy):

    def __init__(
        self,
        resource_availability: dict,   # resource_id -> (7, 24) bool np.ndarray
        global_availability: np.ndarray,
        start_timestamp: str,
    ):
        self.resource_availability = resource_availability
        self.global_availability = global_availability
        self.start_ts = datetime.fromisoformat(start_timestamp).timestamp()

        self.global_counter = 0
        self.resources_counter = 0
    def _matrix(self, resource_id=None) -> np.ndarray:
        if resource_id and resource_id in self.resource_availability:
            self.resources_counter += 1
            return self.resource_availability[resource_id]
        self.global_counter += 1
        return self.global_availability

    def is_working_time(self, t: float, resource_id=None) -> bool:
        dt = datetime.fromtimestamp(self.start_ts + t)
        return bool(self._matrix(resource_id)[dt.weekday(), dt.hour])

    def next_working_time(self, t: float, resource_id=None) -> float:
        matrix = self._matrix(resource_id)
        dt = datetime.fromtimestamp(self.start_ts + t)
        for _ in range(7 * 24):
            if matrix[dt.weekday(), dt.hour]:
                return dt.timestamp() - self.start_ts
            dt += timedelta(hours=1)
        raise RuntimeError("Calendar has no working hours")
