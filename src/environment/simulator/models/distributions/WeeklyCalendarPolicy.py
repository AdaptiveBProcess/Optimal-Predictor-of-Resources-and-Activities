from datetime import datetime, timedelta
from environment.simulator.policies.CalendarPolicy import CalendarPolicy

class WeeklyCalendarPolicy(CalendarPolicy):

    def __init__(self, availability, start_timestamp: str):
        """
        availability: np.ndarray[7, 24] of bool
        """
        self.availability = availability
        self.start_ts = datetime.fromisoformat(start_timestamp).timestamp()

    def is_working_time(self, t: float) -> bool:
        dt = datetime.fromtimestamp(self.start_ts + t)
        return self.availability[dt.weekday(), dt.hour]

    def next_working_time(self, t: float) -> float:
        time = self.start_ts + t
        dt = datetime.fromtimestamp(time)

        for _ in range(7 * 24):
            if self.availability[dt.weekday(), dt.hour]:
                return (dt.timestamp() - self.start_ts)
            dt += timedelta(hours=1)

        raise RuntimeError("Calendar has no working hours")

    def __str__(self) -> str:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        lines = ["WeeklyCalendarPolicy"]

        for d in range(7):
            row = []
            for h in range(24):
                row.append("X" if self.availability[d, h] else ".")
            lines.append(f"{days[d]}: " + " ".join(row))

        return "\n".join(lines)

